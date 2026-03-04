from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
from jax import lax, tree_util

print = partial(print, flush=True)


def occ_columns_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
    """
    Complex exponential parametrization of a Slater determinant.
    """
    theta = jnp.asarray(theta)
    n_occ = theta.shape[0]
    U, s, Vh = jnp.linalg.svd(theta, full_matrices=False)
    V = jnp.conj(Vh.T)
    c = jnp.cos(s)
    sn = jnp.sin(s)
    I_occ = jnp.eye(n_occ, dtype=theta.dtype)
    U_oo = I_occ + U @ jnp.diag(c - 1.0) @ jnp.conj(U.T)
    U_vo = V @ jnp.diag(sn) @ jnp.conj(U.T)
    C = jnp.vstack([U_oo, U_vo])
    return C


def theta_from_occ_columns(C: jnp.ndarray) -> jnp.ndarray:
    """
    Recover a complex Thouless matrix theta from a Slater determinant.
    """
    C = jnp.asarray(C)
    n_occ = C.shape[1]
    C_occ = C[:n_occ, :]
    C_vir = C[n_occ:, :]
    X = C_vir @ jnp.linalg.inv(C_occ)
    Ux, t, Vxh = jnp.linalg.svd(X, full_matrices=False)
    Vx = jnp.conj(Vxh.T)
    s = jnp.arctan(t)
    theta = Vx @ jnp.diag(s) @ jnp.conj(Ux.T)
    return theta


def _pack_psi(psi: jnp.ndarray) -> jnp.ndarray:
    """
    Flatten a complex psi into a real parameter vector
    [Re(psi), Im(psi)].
    """
    psi = jnp.asarray(psi)
    re = jnp.real(psi).ravel()
    im = jnp.imag(psi).ravel()
    return jnp.concatenate([re, im])


def _unpack_psi(theta: jnp.ndarray, shape) -> jnp.ndarray:
    """
    Map a real parameter vector back to complex psi with given shape.
    """
    theta = jnp.asarray(theta)
    n = int(np.prod(shape))
    re = theta[:n].reshape(shape)
    im = theta[n:].reshape(shape)
    return re + 1.0j * im


def apply_sz_projector(
    input_ket: jnp.ndarray, m: float, n_grid: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return sz projected determinants and coefficients for a single ket."""
    norb = input_ket.shape[0] // 2

    gammas = 2.0 * jnp.pi * jnp.arange(n_grid) / n_grid
    exp_factors = jnp.exp(1.0j * gammas / 2.0)
    exp_neg_factors = jnp.exp(-1.0j * gammas / 2.0)
    m_factors = jnp.exp(-1.0j * gammas * m) / n_grid

    def projector_for_gamma(ig):
        rot_diag = jnp.concatenate(
            [jnp.ones(norb) * exp_factors[ig], jnp.ones(norb) * exp_neg_factors[ig]]
        )
        return input_ket * rot_diag[:, None], m_factors[ig]

    return jax.vmap(projector_for_gamma, in_axes=0, out_axes=(0, 0))(jnp.arange(n_grid))


def build_rotchol(psi_t_conj: jnp.ndarray, chol: jnp.ndarray) -> jnp.ndarray:
    """
    Precompute the Cholesky-rotated matrices used for two-body energy evaluation.

    Args:
        psi_t_conj: Trial determinant conjugate transpose, shape (n_occ, 2 * n_orb).
        chol: Cholesky tensors, shape (n_chol, n_orb, n_orb).

    Returns:
        rotchol: Array with shape (2, n_chol, n_occ, n_orb) containing alpha/beta blocks.
    """
    _, norb, _ = chol.shape

    psi_alpha = psi_t_conj[:, :norb]
    psi_beta = psi_t_conj[:, norb:]

    rot_alpha = jnp.einsum("ij,njk->nik", psi_alpha, chol)
    rot_beta = jnp.einsum("ij,njk->nik", psi_beta, chol)

    rotchol = jnp.stack([rot_alpha, rot_beta], axis=0)
    return rotchol


def _evaluate_linear_expansion_energy(
    psi: jnp.ndarray, kets: jnp.ndarray, coeffs: jnp.ndarray, ham: hamiltonian
) -> jnp.ndarray:
    """Compute projected energy for a linear combination of determinants."""
    context = ham.prepare(psi)
    psi_t_conj = psi.T.conj()

    def compute_overlap(ket):
        ovlp_mat = psi_t_conj @ ket
        return jsp.linalg.det(ovlp_mat)

    def compute_energy(ket):
        return ham.mixed_energy(psi, ket, context)

    overlaps = jax.vmap(compute_overlap)(kets)
    energies = jax.vmap(compute_energy)(kets)

    weighted_terms = coeffs * overlaps * energies
    enum = jnp.sum(weighted_terms)
    ovlp = jnp.sum(coeffs * overlaps)

    return (enum / ovlp).real


def get_energy_chol(
    bra: jnp.ndarray,
    ket: jnp.ndarray,
    h1: jnp.ndarray,
    rotchol: jnp.ndarray,
    enuc: float,
) -> jnp.ndarray:
    """Return the mixed estimator <bra|H|ket>/<bra|ket> for a pair of determinants."""
    norb = h1[0].shape[0]
    n_chol = rotchol[0].shape[0]

    # Calculate Green's function.
    Ghalf = ket @ jnp.linalg.inv(bra.T.conj() @ ket)
    Ghalfa = Ghalf[:norb]
    Ghalfb = Ghalf[norb:]
    G = Ghalf @ bra.T.conj()

    # Core energy.
    energy = enuc

    # One-body energy,
    e1 = jnp.trace(h1[0] @ G[:norb, :norb]) + jnp.trace(h1[1] @ G[norb:, norb:])
    energy += e1

    # Two-body energy.
    def scan_fun(i, carry):
        ej, ek = carry
        W0 = rotchol[0, i] @ Ghalfa
        W1 = rotchol[1, i] @ Ghalfb
        ej += 0.5 * (jnp.trace(W0) + jnp.trace(W1)) ** 2
        ek -= 0.5 * (
            jnp.sum(W0 * W0.T) + jnp.sum(W0 * W1.T) + jnp.sum(W1 * W0.T) + jnp.sum(W1 * W1.T)
        )
        return ej, ek

    init = (0.0 + 0.0j, 0.0 + 0.0j)

    ej, ek = jax.lax.fori_loop(0, n_chol, scan_fun, init)
    energy += ej + ek
    return energy


def _mixed_green(bra: jnp.ndarray, ket: jnp.ndarray) -> jnp.ndarray:
    return (ket @ jnp.linalg.inv(bra.T.conj() @ ket) @ bra.T.conj()).T


def get_energy_hubbard(
    bra: jnp.ndarray,
    ket: jnp.ndarray,
    h1: jnp.ndarray,
    u: float,
    enuc: float = 0.0,
) -> jnp.ndarray:
    """Return the mixed estimator <bra|H|ket>/<bra|ket> for a pair of determinants."""
    norb = h1[0].shape[0]
    green = _mixed_green(bra, ket)
    energy_1 = jnp.sum(green[:norb, :norb] * h1[0]) + jnp.sum(green[norb:, norb:] * h1[1])
    energy_2 = u * (
        jnp.sum(green[:norb, :norb].diagonal() * green[norb:, norb:].diagonal())
        - jnp.sum(green[:norb, norb:].diagonal() * green[norb:, :norb].diagonal())
    )
    return energy_1 + energy_2 + enuc


class hamiltonian:
    """Abstract Hamiltonian interface."""

    def prepare(self, bra: jnp.ndarray) -> dict:
        """Precompute any data derived from the bra determinant."""
        return {}

    def mixed_energy(
        self, bra: jnp.ndarray, ket: jnp.ndarray, context: dict | None = None
    ) -> jnp.ndarray:
        """Return <bra|H|ket>/<bra|ket> using optional cached context."""
        raise NotImplementedError


@dataclass
class cholesky_hamiltonian(hamiltonian):
    """Hamiltonian defined by one-body integrals, Cholesky vectors, and core energy."""

    h1: jnp.ndarray
    chol: jnp.ndarray
    enuc: float

    def prepare(self, bra: jnp.ndarray) -> dict:
        norb = self.h1[0].shape[0]
        psi_t_conj = bra.T.conj()
        chol = self.chol.reshape((-1, norb, norb))
        rotchol = build_rotchol(psi_t_conj, chol)
        return {"rotchol": rotchol}

    def mixed_energy(
        self, bra: jnp.ndarray, ket: jnp.ndarray, context: dict | None = None
    ) -> jnp.ndarray:
        assert context is not None, "cholesky_hamiltonian requires cached context."
        return get_energy_chol(bra, ket, self.h1, context["rotchol"], self.enuc)


@dataclass
class hubbard_hamiltonian(hamiltonian):
    """Hubbard Hamiltonian with on-site interaction U."""

    h1: jnp.ndarray
    u: float
    enuc: float = 0.0

    def mixed_energy(
        self, bra: jnp.ndarray, ket: jnp.ndarray, context: dict | None = None
    ) -> jnp.ndarray:
        return get_energy_hubbard(bra, ket, self.h1, self.u, self.enuc)


def _spin_rot_elements(alpha: float, beta: float, gamma: float) -> tuple[jnp.ndarray, ...]:
    """
    Spin-1/2 rotation (z-y-z Euler angles):
    U = e^{-i alpha sigma_z/2} e^{-i beta sigma_y/2} e^{-i gamma sigma_z/2}
      = [[e^{-i(alpha+gamma)/2} cos(beta/2),   -e^{-i(alpha-gamma)/2} sin(beta/2)],
         [e^{ i(alpha-gamma)/2} sin(beta/2),    e^{ i(alpha+gamma)/2} cos(beta/2)]]
    Returns the four scalar elements u11, u12, u21, u22.
    """
    half = 0.5
    cb = jnp.cos(beta * half)
    sb = jnp.sin(beta * half)
    e_p = jnp.exp(-0.5j * (alpha + gamma))
    e_m = jnp.exp(-0.5j * (alpha - gamma))
    u11 = e_p * cb
    u12 = -e_m * sb
    u21 = jnp.conj(e_m) * sb
    u22 = jnp.conj(e_p) * cb
    return u11, u12, u21, u22


def apply_s0_projector(
    input_ket: jnp.ndarray, n_alpha: int = 8, n_beta: int = 8, n_gamma: int = 8
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    SU(2) projection specialized to S=0 on a general (noncollinear) GHF state.

    Args:
      input_ket : jax.Array, shape (2*norb, nocc)   (GHF columns are occupied spinors)
      n_alpha, n_beta, n_gamma : Euler angle quadrature counts

    Returns:
      kets   : (Ngrid, 2*norb, nocc) rotated kets R(omega)|phi⟩
      coeffs : (Ngrid,) complex weights including (1/8pi^2) * sin beta * volume element
    """
    norb = input_ket.shape[0] // 2

    # uniform alpha, gamma in [0, 2pi), midpoint beta in (0, pi) with sin beta weight
    dalpha = 2.0 * jnp.pi / n_alpha
    dgamma = 2.0 * jnp.pi / n_gamma
    alpha = jnp.arange(n_alpha) * dalpha
    gamma = jnp.arange(n_gamma) * dgamma
    beta_edges = jnp.linspace(0.0, jnp.pi, n_beta + 1)
    beta = 0.5 * (beta_edges[:-1] + beta_edges[1:])
    dbeta = jnp.pi / n_beta

    A, B, G = jnp.meshgrid(alpha, beta, gamma, indexing="ij")
    A = A.reshape(-1)
    B = B.reshape(-1)
    G = G.reshape(-1)

    prefac = (1.0 / (8.0 * jnp.pi**2)) * dalpha * dbeta * dgamma

    def rotate_one(angles):
        a, b, g = angles
        u11, u12, u21, u22 = _spin_rot_elements(a, b, g)
        alpha_block = u11 * input_ket[:norb, :] + u12 * input_ket[norb:, :]
        beta_block = u21 * input_ket[:norb, :] + u22 * input_ket[norb:, :]
        ket_rot = jnp.vstack([alpha_block, beta_block])
        weight = prefac * jnp.sin(b)
        return ket_rot, weight

    kets, coeffs = jax.vmap(rotate_one)(jnp.stack([A, B, G], axis=1))
    return kets, coeffs


def _gauss_legendre_nodes_weights(n: int):
    """
    Golub–Welsch for Gauss–Legendre on [-1, 1].
    Returns nodes x (ascending) and weights w.
    """
    assert n > 0, "Number of Gauss–Legendre points must be positive."
    k = jnp.arange(1, n)
    b = k / jnp.sqrt(4.0 * k * k - 1.0)  # off-diagonals
    # symmetric tridiagonal Jacobi matrix
    J = jnp.zeros((n, n))
    J = J.at[jnp.arange(n - 1), jnp.arange(1, n)].set(b)
    J = J.at[jnp.arange(1, n), jnp.arange(n - 1)].set(b)
    x, V = jnp.linalg.eigh(J)  # x: nodes; V: eigenvectors
    w = 2.0 * (V[0, :] ** 2)  # weights
    return x, w


def apply_s0_projector_gauss(
    input_ket: jnp.ndarray,
    n_alpha: int = 8,
    n_beta: int = 16,
    n_gamma: int = 8,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    S=0 projector with Gauss–Legendre quadrature in beta (via t = cos beta).
    alpha, gamma use uniform trapezoidal grids.

    Returns:
      kets   : (n_alpha * n_beta * n_gamma, 2*norb, nocc)
      coeffs : (n_alpha * n_beta * n_gamma,) complex weights
               equal to (1/8pi^2) * (2pi/n_alpha) * (2pi/n_gamma) * w_beta[j]
    """
    norb = input_ket.shape[0] // 2

    # alpha, gamma: uniform on [0, 2pi)
    dalpha = (2.0 * jnp.pi) / n_alpha
    dgamma = (2.0 * jnp.pi) / n_gamma
    alpha = jnp.arange(n_alpha) * dalpha
    gamma = jnp.arange(n_gamma) * dgamma

    # beta: Gauss–Legendre on t = cos beta in [-1, 1]
    t, w_beta = _gauss_legendre_nodes_weights(n_beta)
    beta = jnp.arccos(t)  # maps [-1,1] → [0,pi]

    # full tensor grid, then flatten
    A, B, G = jnp.meshgrid(alpha, beta, gamma, indexing="ij")
    angles = jnp.stack([A.ravel(), B.ravel(), G.ravel()], axis=1)

    # weights: (1/8pi^2) * dalpha * dgamma * w_beta[j]
    W = (1.0 / (8.0 * jnp.pi**2)) * dalpha * dgamma
    weights = (W * jnp.tile(w_beta[None, :, None], (n_alpha, 1, n_gamma))).ravel()

    up = input_ket[:norb, :]
    dn = input_ket[norb:, :]

    def rotate_one(ang):
        a, b, g = ang
        u11, u12, u21, u22 = _spin_rot_elements(a, b, g)
        alpha_block = u11 * up + u12 * dn
        beta_block = u21 * up + u22 * dn
        return jnp.vstack([alpha_block, beta_block])

    kets = jax.vmap(rotate_one)(angles)
    return kets, weights


def apply_s0_projector_gauss_reduced(
    input_ket: jnp.ndarray,
    n_beta: int = 16,
    n_gamma: int = 8,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reduced S=0 projector for bra-side use when [O, Sz] = 0 and m = 0.

    The alpha Euler angle integral is trivial (gives factor 2pi), leaving
    only n_beta * n_gamma grid points instead of n_alpha * n_beta * n_gamma.
    Weights become (1/4pi) * (2pi/n_gamma) * w_beta[j].

    Returns:
      kets   : (n_beta * n_gamma, 2*norb, nocc)
      coeffs : (n_beta * n_gamma,) real weights
    """
    norb = input_ket.shape[0] // 2

    dgamma = (2.0 * jnp.pi) / n_gamma
    gamma = jnp.arange(n_gamma) * dgamma

    t, w_beta = _gauss_legendre_nodes_weights(n_beta)
    beta = jnp.arccos(t)

    B, G = jnp.meshgrid(beta, gamma, indexing="ij")
    angles = jnp.stack([B.ravel(), G.ravel()], axis=1)

    W = (1.0 / (4.0 * jnp.pi)) * dgamma
    weights = (W * jnp.tile(w_beta[:, None], (1, n_gamma))).ravel()

    up = input_ket[:norb, :]
    dn = input_ket[norb:, :]

    def rotate_one(ang):
        b, g = ang
        u11, u12, u21, u22 = _spin_rot_elements(0.0, b, g)
        alpha_block = u11 * up + u12 * dn
        beta_block = u21 * up + u22 * dn
        return jnp.vstack([alpha_block, beta_block])

    kets = jax.vmap(rotate_one)(angles)
    return kets, weights


def _apply_pg_to_ket(ket: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
    """U acts in orbital space on both spin blocks."""
    norb = ket.shape[0] // 2
    up = U @ ket[:norb, :]
    dn = U @ ket[norb:, :]
    return jnp.vstack([up, dn])


class projector:
    """Abstract projector that maps a batch of determinants to a new linear combination."""

    def apply(self, kets: jnp.ndarray, coeffs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError


@dataclass
class sz_projector(projector):
    """Project onto fixed Sz using discrete rotations."""

    m: float = 0.0
    n_grid: int = 8

    def apply(self, kets: jnp.ndarray, coeffs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def rotate_single(ket: jnp.ndarray):
            return apply_sz_projector(ket, self.m, self.n_grid)

        rot_kets, rot_coeffs = jax.vmap(rotate_single, in_axes=0)(kets)
        batch, nterms, norb2, nocc = rot_kets.shape
        new_kets = rot_kets.reshape(batch * nterms, norb2, nocc)
        new_coeffs = (coeffs[:, None] * rot_coeffs).reshape(batch * nterms)
        return new_kets, new_coeffs


@dataclass
class singlet_projector(projector):
    """Project onto total spin S=0 using SU(2) quadrature."""

    n_alpha: int = 8
    n_beta: int = 8
    n_gamma: int = 8

    def apply(self, kets: jnp.ndarray, coeffs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def rotate_single(ket):
            return apply_s0_projector_gauss(
                ket, n_alpha=self.n_alpha, n_beta=self.n_beta, n_gamma=self.n_gamma
            )

        rot_kets, rot_coeffs = jax.vmap(rotate_single, in_axes=0)(kets)
        batch, nterms, norb2, nocc = rot_kets.shape
        new_kets = rot_kets.reshape(batch * nterms, norb2, nocc)
        new_coeffs = (coeffs[:, None] * rot_coeffs).reshape(batch * nterms)
        return new_kets, new_coeffs


@dataclass
class reduced_singlet_projector(projector):
    """Reduced S=0 projector (beta, gamma only) for bra-side use when [O, Sz] = 0."""

    n_beta: int = 8
    n_gamma: int = 8

    def apply(self, kets: jnp.ndarray, coeffs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def rotate_single(ket):
            return apply_s0_projector_gauss_reduced(ket, n_beta=self.n_beta, n_gamma=self.n_gamma)

        rot_kets, rot_coeffs = jax.vmap(rotate_single, in_axes=0)(kets)
        batch, nterms, norb2, nocc = rot_kets.shape
        new_kets = rot_kets.reshape(batch * nterms, norb2, nocc)
        new_coeffs = (coeffs[:, None] * rot_coeffs).reshape(batch * nterms)
        return new_kets, new_coeffs


@dataclass
class point_group_projector(projector):
    """Project onto a point-group irrep defined by orbital operators."""

    pg_ops: list | jnp.ndarray
    pg_chars: list | None = None

    def __post_init__(self):
        if self.pg_chars is None:
            chars = jnp.ones(len(self.pg_ops), dtype=jnp.complex128)
        else:
            chars = jnp.asarray(self.pg_chars, dtype=jnp.complex128)

        self.pg_ops = jnp.array([jnp.asarray(U) for U in self.pg_ops])
        self._chars = chars
        self._norm = 1.0 / len(self.pg_ops)

    def apply(self, kets: jnp.ndarray, coeffs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:

        def apply_one_op(U, kets_batch):
            return jax.vmap(_apply_pg_to_ket, in_axes=(0, None))(kets_batch, U)

        rotated = jax.vmap(apply_one_op, in_axes=(0, None))(self.pg_ops, kets)
        new_kets = rotated.reshape(-1, *kets.shape[1:])
        weight_factors = (jnp.conj(self._chars) * self._norm)[:, None]
        new_coeffs = (weight_factors * coeffs[None, :]).reshape(-1)
        return new_kets, new_coeffs


@dataclass
class k_projector(projector):
    """Project complex conjugation symmetry."""

    parity: int = 1

    def apply(self, kets: jnp.ndarray, coeffs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        kets_conj = jnp.conj(kets)
        new_kets = jnp.concatenate([kets, kets_conj], axis=0)
        new_coeffs = 0.5 * jnp.concatenate([coeffs, self.parity * jnp.conj(coeffs)], axis=0)
        return new_kets, new_coeffs


def _apply_projector_sequence(
    psi: jnp.ndarray, projectors: tuple[projector, ...]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply projectors sequentially to generate a linear combination of Slater determinants.

    Returns:
        Tuple of (kets, coeffs) describing the expanded state.
    """
    kets = psi[jnp.newaxis, ...]
    if jnp.issubdtype(psi.dtype, jnp.complexfloating):
        coeff_dtype = psi.dtype
    else:
        coeff_dtype = jnp.complex128
    coeffs = jnp.array([1.0 + 0.0j], dtype=coeff_dtype)

    for projector in projectors:
        kets, coeffs = projector.apply(kets, coeffs)

    return kets, coeffs


def _split_projectors(
    projectors: tuple[projector, ...], commuting_types: set[type]
) -> tuple[tuple[projector, ...], tuple[projector, ...], bool]:
    """
    Split projectors for asymmetric bra/ket evaluation.

    Projectors whose type is in *commuting_types* commute with the operator
    being measured and are dropped from the bra side (absorbed via P^2 = P).

    Additionally, if sz_projector is in commuting_types but singlet_projector
    is not, the singlet projector on the bra side is replaced by a reduced
    version that omits the alpha Euler angle integral (since e^{i*alpha*Sz}
    commutes through O and is absorbed by P_Sz on the ket side).

    Returns:
        (bra_projectors, ket_projectors, changed) where ket_projectors is the
        full original tuple, bra_projectors has commuting ones removed/reduced,
        and changed indicates whether the bra differs from the ket.
    """
    sz_commutes = sz_projector in commuting_types
    bra_projs = []
    changed = False
    for p in projectors:
        if type(p) in commuting_types:
            changed = True
            continue
        if isinstance(p, singlet_projector) and sz_commutes:
            bra_projs.append(reduced_singlet_projector(n_beta=p.n_beta, n_gamma=p.n_gamma))
            changed = True
        else:
            bra_projs.append(p)
    return tuple(bra_projs), tuple(projectors), changed


def _has_k_projector(projectors: tuple[projector, ...]) -> bool:
    return any(isinstance(p, k_projector) for p in projectors)


def _split_projectors_for_k(
    projectors: tuple[projector, ...],
) -> tuple[tuple[projector, ...], int]:
    """
    Split projectors into:
      - linear/unitary projectors Q (spin, PG, etc.)
      - combined parity for any k_projector present.

    We assume at most a small number of k_projector instances; their parities
    just multiply.
    """
    linear_projectors: list[projector] = []
    parity = 1
    for p in projectors:
        if isinstance(p, k_projector):
            parity *= p.parity
        else:
            linear_projectors.append(p)
    return tuple(linear_projectors), parity


def _evaluate_projected_energy(
    psi: jnp.ndarray, ham: hamiltonian, projectors: tuple[projector, ...] = tuple()
) -> jnp.ndarray:
    """
    Evaluate the projected energy after applying a sequence of projectors.
    """
    if not _has_k_projector(projectors):
        kets, coeffs = _apply_projector_sequence(psi, projectors)
        return _evaluate_linear_expansion_energy(psi, kets, coeffs, ham)

    linear_projectors, parity = _split_projectors_for_k(projectors)

    phi0 = psi
    phi1 = jnp.conj(psi)
    bras = (phi0, phi1)

    if jnp.issubdtype(psi.dtype, jnp.complexfloating):
        cdtype = psi.dtype
    else:
        cdtype = jnp.complex128

    contexts = [ham.prepare(bra) for bra in bras]
    bra_t_conjs = [bra.T.conj() for bra in bras]

    exp_kets = []
    exp_coeffs = []
    for j in range(2):
        kets_j, coeffs_j = _apply_projector_sequence(bras[j], linear_projectors)
        exp_kets.append(kets_j)
        exp_coeffs.append(coeffs_j)

    S = jnp.zeros((2, 2), dtype=cdtype)
    H = jnp.zeros((2, 2), dtype=cdtype)

    for i in range(2):
        bra = bras[i]
        ctx = contexts[i]
        bra_t_conj = bra_t_conjs[i]

        def compute_overlap(ket):
            ovlp_mat = bra_t_conj @ ket
            return jsp.linalg.det(ovlp_mat)

        def compute_energy(ket):
            return ham.mixed_energy(bra, ket, ctx)

        for j in range(2):
            kets_j = exp_kets[j]
            coeffs_j = exp_coeffs[j]

            overlaps_ij = jax.vmap(compute_overlap)(kets_j)
            energies_ij = jax.vmap(compute_energy)(kets_j)

            S_ij = jnp.sum(coeffs_j * overlaps_ij)
            H_ij = jnp.sum(coeffs_j * overlaps_ij * energies_ij)

            S = S.at[i, j].set(S_ij)
            H = H.at[i, j].set(H_ij)

    d = jnp.array([1.0 + 0.0j, float(parity) + 0.0j], dtype=cdtype)

    num = jnp.vdot(d, H @ d)
    den = jnp.vdot(d, S @ d)
    return (num / den).real


# def _evaluate_projected_energy(
#     psi: jnp.ndarray, ham: hamiltonian, projectors: tuple[projector, ...]
# ) -> jnp.ndarray:
#     """Evaluate the projected energy after applying a sequence of projectors."""
#     kets, coeffs = _apply_projector_sequence(psi, projectors)
#     return _evaluate_linear_expansion_energy(psi, kets, coeffs, ham)


def make_energy_fn(ham, projectors, theta_shape):
    """
    Returns E(theta), where theta is a real 1D vector encoding complex psi.
    """

    def energy_fn(theta: jnp.ndarray) -> jnp.ndarray:
        theta = _unpack_psi(theta, theta_shape)
        psi = occ_columns_from_theta(theta)
        E = _evaluate_projected_energy(psi, ham, projectors)
        return jnp.real(E)

    return energy_fn


def calculate_hessian(theta, ham, projectors=tuple()):
    """
    Compute the Hessian of the projected energy with respect to the
    real parameters (Re(theta), Im(theta)) at theta.
    """
    theta = jnp.asarray(theta, dtype=jnp.complex128)
    theta_shape = theta.shape
    theta = _pack_psi(theta)

    energy_fn = make_energy_fn(ham, projectors, theta_shape)
    energy_jit = jax.jit(energy_fn)
    grad_jit = jax.jit(jax.grad(energy_fn))
    hess_jit = jax.jit(jax.hessian(energy_fn))

    E0 = energy_jit(theta)
    grad = grad_jit(theta)
    H = hess_jit(theta)

    return float(np.array(E0)), np.array(grad), np.array(H)


def calculate_projected_energy(
    psi: jnp.ndarray,
    ham: hamiltonian,
    projectors: tuple[projector, ...] = tuple(),
) -> jnp.ndarray:
    """
    Calculate the projected energy of a GHF determinant under optional symmetry projections.

    Args:
        psi: GHF determinant.
        ham: Hamiltonian object implementing the mixed-energy interface.
        projectors: Sequence of projector objects applied to |psi>.

    Returns:
        Projected energy as a scalar jax.Array.
    """

    @jax.jit
    def energy_function():
        return _evaluate_projected_energy(psi, ham, projectors)

    return energy_function()


def optimize(
    psi: jnp.ndarray,
    ham: hamiltonian,
    projectors: tuple[projector, ...] = tuple(),
    maxiter: int = 100,
    grad_conv_tol: float = 1e-5,
    step: float = 0.01,
    printQ: bool = True,
    optimizer_name: str = "lbfgs",
    use_thouless: bool = False,
    param_file: str | None = None,
) -> tuple[float, np.ndarray]:
    """
    Variationally optimize a GHF determinant under optional symmetry projections.

    Args:
        psi: Initial GHF determinant.
        ham: Hamiltonian object implementing the mixed energy interface.
        projectors: Sequence of projector objects applied to |psi>.
        maxiter: Maximum number of optimization iterations.
        grad_conv_tol: Convergence tolerance on gradient norm.
        step: Initial step size for optimizers that require it.
        printQ: Whether to print optimization progress.
        optimizer_name: Name of the Optax optimizer to use ("lbfgs" or "amsgrad").
        use_thouless: Whether to parametrize psi via Thouless rotations.
        param_file: If provided, save optimized parameters to this file.

    Returns:
      (final_energy, psi_opt_np)
    """
    psi0 = jnp.asarray(psi, dtype=jnp.complex128)
    psi_shape = psi0.shape
    theta_shape = None

    if use_thouless:
        psi0 = jnp.linalg.qr(psi0)[0]
        theta0_mat = theta_from_occ_columns(psi0)
        theta_shape = theta0_mat.shape
        theta0 = _pack_psi(theta0_mat)

        def objective_thouless(theta: jnp.ndarray) -> jnp.ndarray:
            current_theta = _unpack_psi(theta, theta_shape)
            current_psi = occ_columns_from_theta(current_theta)
            energy = _evaluate_projected_energy(current_psi, ham, projectors)
            return jnp.real(energy)

        objective_function = objective_thouless
    else:
        theta0 = _pack_psi(psi0)

        def objective_direct(theta: jnp.ndarray) -> jnp.ndarray:
            current_psi = _unpack_psi(theta, psi_shape)
            energy = _evaluate_projected_energy(current_psi, ham, projectors)
            return jnp.real(energy)

        objective_function = objective_direct

    value_fn = jax.jit(objective_function)

    if optimizer_name == "amsgrad":
        optimizer = optax.amsgrad(step, b2=0.99)
        vg = jax.jit(jax.value_and_grad(value_fn))

        def valgrad_amsgrad(theta, opt_state):
            val, grad = vg(theta)
            return val, grad

        def update_amsgrad(grad, opt_state, theta, value):
            updates, new_state = optimizer.update(grad, opt_state, params=theta)
            return updates, new_state

        valgrad_fn = valgrad_amsgrad
        update_fn = update_amsgrad

    elif optimizer_name == "lbfgs":
        optimizer = optax.lbfgs(memory_size=10)
        vgs = optax.value_and_grad_from_state(value_fn)

        def valgrad_lbfgs(theta, opt_state):
            val, grad = vgs(theta, state=opt_state)
            return val, grad

        def update_lbfgs(grad, opt_state, theta, value):
            updates, new_state = optimizer.update(
                grad,
                opt_state,
                params=theta,
                value=value,
                grad=grad,
                value_fn=value_fn,
            )
            return updates, new_state

        valgrad_fn = valgrad_lbfgs
        update_fn = update_lbfgs

    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")

    opt_state = optimizer.init(theta0)
    theta = theta0

    @jax.jit
    def opt_step(theta, opt_state):
        value, grad = valgrad_fn(theta, opt_state)
        grad = tree_util.tree_map(jnp.real, grad)
        updates, opt_state = update_fn(grad, opt_state, theta, value)
        theta = optax.apply_updates(theta, updates)
        grad_norm = optax.global_norm(grad)
        return theta, opt_state, value, grad_norm

    e0 = float(jax.device_get(value_fn(theta)))
    if printQ:
        print(f"\n# Initial projected energy = {e0}")
        print(f"Starting optimization with Optax {optimizer_name}...")

    energy = e0
    for iteration in range(maxiter):
        theta, opt_state, value, grad_norm = opt_step(theta, opt_state)

        energy = float(jax.device_get(value))
        gn = float(jax.device_get(grad_norm))

        if printQ:
            print(f"Iteration {iteration + 1}: Energy = {energy}, Grad norm = {gn}")

        if gn < grad_conv_tol:
            if printQ:
                print("Converged!")
            break

    # unpack final params back to psi
    if use_thouless:
        theta_opt = _unpack_psi(theta, theta_shape)
        psi_opt = occ_columns_from_theta(theta_opt)
    else:
        psi_opt = _unpack_psi(theta, psi_shape)

    psi_opt_np = np.array(jax.device_get(psi_opt))
    if param_file is not None:
        np.savetxt(param_file, psi_opt_np)

    return float(energy), psi_opt_np


def _accumulate_weighted(kets, coeffs, property_fn, n_chunks=1):
    N = kets.shape[0]
    batch_size = (N + n_chunks - 1) // n_chunks

    def row_contribution(ket_j, c_j):
        def col_fn(bra_i, c_i):
            o = jnp.linalg.det(bra_i.T.conj() @ ket_j)
            v = property_fn(bra_i, ket_j)
            w = c_j.conj() * c_i * o
            return w * v, w

        def inner_f(x):
            return col_fn(x[0], x[1])

        w_vals, ws = lax.map(inner_f, (kets, coeffs), batch_size=batch_size)
        return jnp.sum(w_vals, axis=0), jnp.sum(ws)

    def outer_f(x):
        return row_contribution(x[0], x[1])

    nums, denoms = lax.map(outer_f, (kets, coeffs), batch_size=batch_size)
    return (jnp.sum(nums, axis=0) / jnp.sum(denoms)).real


def _accumulate_weighted_asymmetric(bras, bra_coeffs, kets, ket_coeffs, property_fn, n_chunks=1):
    """Accumulate a projected property with different bra and ket expansions.

    Computes  sum_{i,j} c_i^* c_j <bra_i|O|ket_j> / sum_{i,j} c_i^* c_j <bra_i|ket_j>
    where the bra and ket linear combinations may have different sizes.
    """
    N_bra = bras.shape[0]
    N_ket = kets.shape[0]
    bra_batch = (N_bra + n_chunks - 1) // n_chunks
    ket_batch = (N_ket + n_chunks - 1) // n_chunks

    def row_contribution(ket_j, c_j):
        def col_fn(bra_i, c_i):
            o = jnp.linalg.det(bra_i.T.conj() @ ket_j)
            v = property_fn(bra_i, ket_j)
            w = c_i.conj() * c_j * o
            return w * v, w

        def inner_f(x):
            return col_fn(x[0], x[1])

        w_vals, ws = lax.map(inner_f, (bras, bra_coeffs), batch_size=bra_batch)
        return jnp.sum(w_vals, axis=0), jnp.sum(ws)

    def outer_f(x):
        return row_contribution(x[0], x[1])

    nums, denoms = lax.map(outer_f, (kets, ket_coeffs), batch_size=ket_batch)
    return (jnp.sum(nums, axis=0) / jnp.sum(denoms)).real


def _evaluate_projected_property(
    psi: jnp.ndarray,
    projectors: tuple,
    property_fn: callable,
    n_chunks: int = 1,
    commuting_types: set[type] | None = None,
) -> jnp.ndarray:
    if commuting_types is None:
        kets, coeffs = _apply_projector_sequence(psi, projectors)
        return _accumulate_weighted(kets, coeffs, property_fn, n_chunks=n_chunks)

    bra_projs, ket_projs, changed = _split_projectors(projectors, commuting_types)
    if not changed:
        kets, coeffs = _apply_projector_sequence(psi, projectors)
        return _accumulate_weighted(kets, coeffs, property_fn, n_chunks=n_chunks)

    bras, bra_coeffs = _apply_projector_sequence(psi, bra_projs)
    kets, ket_coeffs = _apply_projector_sequence(psi, ket_projs)
    return _accumulate_weighted_asymmetric(
        bras, bra_coeffs, kets, ket_coeffs, property_fn, n_chunks=n_chunks
    )


def calculate_projected_1rdm(
    psi: jnp.ndarray,
    projectors: tuple[projector, ...],
    n_chunks: int = 1,
) -> np.ndarray:
    """
    Calculate the spin-conserving blocks of the projected 1-RDM.

    Returns the up-up and down-down blocks as a (2, norb, norb) array.
    The spin-flip blocks vanish by symmetry for Sz=0 projected states
    and are not computed.  Since each a_{is}^dag a_{js} commutes with
    P_Sz, that projector is absorbed from the bra side.  When a singlet
    projector is present, its alpha integral is also reduced on the bra.

    Args:
        psi: GHF determinant, shape (2*norb, nocc).
        projectors: Sequence of projector objects applied to |psi>.

    Returns:
        Projected 1-RDM spin blocks as a (2, norb, norb) numpy ndarray.
    """
    norb = psi.shape[0] // 2

    def _spin_conserving_green(bra, ket):
        G = _mixed_green(bra, ket)
        return jnp.stack([G[:norb, :norb], G[norb:, norb:]])

    @jax.jit
    def rdm1_function(psi_var: jnp.ndarray) -> jnp.ndarray:
        return _evaluate_projected_property(
            psi_var,
            projectors,
            _spin_conserving_green,
            n_chunks=n_chunks,
            commuting_types={sz_projector},
        )

    rdm1 = rdm1_function(psi)
    return np.array(rdm1)


def calculate_projected_density_correlations(
    psi: jnp.ndarray,
    projectors: tuple[projector, ...],
    n_chunks: int = 1,
) -> np.ndarray:
    """
    Calculate the projected density-density correlation matrix
    of a GHF determinant under optional symmetry projections.

    Args:
        psi: GHF determinant.
        projectors: Sequence of projector objects applied to |psi>.

    Returns:
        Projected density-density correlation matrix as a numpy ndarray.
    """

    @jax.jit
    def density_corr_function(psi_var: jnp.ndarray) -> jnp.ndarray:
        def calc_density_corr(bra, ket):
            green = _mixed_green(bra, ket)
            green_diag = jnp.diagonal(green)
            density_corr = (
                green_diag[:, None] * green_diag[None, :] - green * green.T + jnp.diag(green_diag)
            )
            return density_corr

        return _evaluate_projected_property(
            psi_var,
            projectors,
            calc_density_corr,
            n_chunks=n_chunks,
            commuting_types={sz_projector},
        )

    density_corr = density_corr_function(psi)
    return np.array(density_corr)


def _SzSz_from_G(G):
    n = G.shape[0] // 2
    Guu = G[:n, :n]
    Gdd = G[n:, n:]
    Gud = G[:n, n:]
    Gdu = G[n:, :n]

    def nn_same(Gblk):
        occ = jnp.diag(Gblk)
        return occ[:, None] * occ[None, :] - Gblk * Gblk.T + jnp.diag(occ)

    Euu = nn_same(Guu)
    Edd = nn_same(Gdd)
    Eud = jnp.diag(Guu)[:, None] * jnp.diag(Gdd)[None, :] - Gud * Gdu.T
    Edu = jnp.diag(Gdd)[:, None] * jnp.diag(Guu)[None, :] - Gdu * Gud.T

    SzSz = 0.25 * (Euu + Edd - Eud - Edu)
    return SzSz


def _SpSm_from_G(G):
    n = G.shape[0] // 2
    Guu = G[:n, :n]
    Gdd = G[n:, n:]
    Gud = G[:n, n:]
    Gdu = G[n:, :n]
    coh = jnp.outer(jnp.diag(Gud), jnp.diag(Gdu))
    diag = jnp.diag(jnp.diag(Guu))
    exch = Guu * Gdd.T
    return coh + diag - exch


def _SmSp_from_G(G):
    n = G.shape[0] // 2
    Guu = G[:n, :n]
    Gdd = G[n:, n:]
    Gud = G[:n, n:]
    Gdu = G[n:, :n]
    coh = jnp.outer(jnp.diag(Gdu), jnp.diag(Gud))
    diag = jnp.diag(jnp.diag(Gdd))
    exch = Gdd * Guu.T
    return coh + diag - exch


def _SdotS_from_G(G):
    return _SzSz_from_G(G) + 0.5 * (_SpSm_from_G(G) + _SmSp_from_G(G))


def calculate_projected_sz_correlations(psi, projectors, n_chunks=1):
    @jax.jit
    def evaluate():
        def prop_fn(bra, ket):
            return _SzSz_from_G(_mixed_green(bra, ket))

        return _evaluate_projected_property(
            psi,
            projectors,
            prop_fn,
            n_chunks=n_chunks,
            commuting_types={sz_projector},
        )

    corr = evaluate()
    return np.array(corr)


def calculate_projected_s_correlations(psi, projectors, n_chunks=1):
    @jax.jit
    def evaluate():
        def prop_fn(bra, ket):
            return _SdotS_from_G(_mixed_green(bra, ket))

        return _evaluate_projected_property(
            psi,
            projectors,
            prop_fn,
            n_chunks=n_chunks,
            commuting_types={sz_projector, singlet_projector},
        )

    corr = evaluate()
    return np.array(corr)


def calculate_projected_s2(psi, projectors, n_chunks=1):
    @jax.jit
    def evaluate():
        def prop_fn(bra, ket):
            G = _mixed_green(bra, ket)
            return jnp.sum(_SdotS_from_G(G))

        return _evaluate_projected_property(
            psi,
            projectors,
            prop_fn,
            n_chunks=n_chunks,
            commuting_types={sz_projector, singlet_projector},
        )

    s2 = evaluate()
    return float(s2)
