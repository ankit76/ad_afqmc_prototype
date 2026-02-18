from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GhfTrial:
    """
    Generalized HF trial.
    """
    mo_coeff: jax.Array  # (2*norb, nocc)

    @property
    def norb(self) -> int:
        return int(self.mo_coeff.shape[0] // 2)

    @property
    def nocc(self) -> int:
        return int(self.mo_coeff.shape[1])

    def tree_flatten(self):
        return (self.mo_coeff,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (mo_coeff,) = children
        return cls(mo_coeff=mo_coeff)


def _det(m: jax.Array) -> jax.Array:
    return jnp.linalg.det(m)


def get_rdm1_u(trial_data: GhfTrial) -> jax.Array:
    """
    Return spin-block 1RDM for use by AFQMC propagator code that expects
    (2, norb, norb) for restricted basis Hamiltonians.

    Note: This discards spin offdiagonal blocks in a true GHF density matrix.
    See get_rdm1_g for the full 1RDM matrix.
    """
    c = trial_data.mo_coeff
    dm = c @ c.conj().T  # (2*norb, 2*norb)
    norb = trial_data.norb
    dm_u = dm[:norb, :norb]
    dm_d = dm[norb:, norb:]
    return jnp.stack([dm_u, dm_d], axis=0)  # (2, norb, norb)


def get_rdm1_g(trial_data: GhfTrial) -> jax.Array:
    """
    Full (2*norb, 2*norb) 1RDM.
    """
    c = trial_data.mo_coeff
    dm = c @ c.conj().T  # (2*norb, 2*norb)
    return dm


def overlap_r(walker: jax.Array, trial_data: GhfTrial) -> jax.Array:
    norb = trial_data.norb
    cH = trial_data.mo_coeff.conj().T  # (ne, 2*norb)
    top = cH[:, :norb] @ walker
    bot = cH[:, norb:] @ walker
    m = jnp.hstack([top, bot])  # (ne, 2*nocc)
    return _det(m)


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: GhfTrial) -> jax.Array:
    wu, wd = walker
    norb = trial_data.norb
    cH = trial_data.mo_coeff.conj().T  # (ne, 2*norb)
    top = cH[:, :norb] @ wu
    bot = cH[:, norb:] @ wd
    m = jnp.hstack([top, bot])  # (ne, ne)
    return _det(m)


def overlap_g(walker: jax.Array, trial_data: GhfTrial) -> jax.Array:
    m = trial_data.mo_coeff.conj().T @ walker  # (ne, ne)
    return _det(m)


def _eff_idx(update_indices: jax.Array, norb: int) -> tuple[jax.Array, jax.Array]:
    """
    Returns effective indices in the combined (2*norb) basis.
    """
    spin_i, i = update_indices[0]
    spin_j, j = update_indices[1]
    i_eff = i + (spin_i == 1) * norb
    j_eff = j + (spin_j == 1) * norb
    return i_eff, j_eff


def _ratio_full_rank2(
    G: jax.Array, i: jax.Array, j: jax.Array, u0: jax.Array, u1: jax.Array
) -> jax.Array:
    """
    Determinant-lemma overlap ratio for two diagonal updates
    """
    Gii = G[i, i]
    Gjj = G[j, j]
    Gij = G[i, j]
    Gji = G[j, i]
    return (1.0 + u0 * Gii) * (1.0 + u1 * Gjj) - (u0 * u1) * (Gij * Gji)


def _update_full_rank2(
    G: jax.Array,
    i: jax.Array,
    j: jax.Array,
    u0: jax.Array,
    u1: jax.Array,
    *,
    eps: float = 1.0e-8,
    sanitize: bool = True,
) -> jax.Array:
    """
    SMW update for the two diagonal update
    """
    r = _ratio_full_rank2(G, i, j, u0, u1)
    r_safe = jnp.where(jnp.abs(r) < eps, jnp.asarray(1.0, dtype=r.dtype), r)

    s_i = G[i].at[i].add(-1)
    s_j = G[j].at[j].add(-1)

    col_i = G[:, i]
    col_j = G[:, j]

    Gii = G[i, i]
    Gjj = G[j, j]
    Gij = G[i, j]
    Gji = G[j, i]

    term_i = u1 * (Gij * s_j - Gjj * s_i) - s_i
    term_j = u0 * (Gji * s_i - Gii * s_j) - s_j

    G_new = (
        G
        + (u0 / r_safe) * jnp.outer(col_i, term_i)
        + (u1 / r_safe) * jnp.outer(col_j, term_j)
    )

    if sanitize:
        z = jnp.asarray(0.0, dtype=G_new.dtype)
        G_new = jnp.where(jnp.isfinite(G_new), G_new, z)

    return G_new


def calc_green_u(
    walker: tuple[jax.Array, jax.Array], trial_data: GhfTrial
) -> jax.Array:
    """
    Compute full G for unrestricted walker
    """
    wu, wd = walker
    norb = wu.shape[0]
    nup = wu.shape[1]
    c = trial_data.mo_coeff  # (2*norb, nocc)
    cu = c[:norb, :]  # (norb, nocc)
    cd = c[norb:, :]  # (norb, nocc)
    o_left = cu.conj().T @ wu  # (nocc, nup)
    o_right = cd.conj().T @ wd  # (nocc, ndn)
    o = jnp.concatenate([o_left, o_right], axis=1)
    x = jnp.linalg.solve(o, c.conj().T)  # (nocc, 2*norb)
    top = wu @ x[:nup, :]  # (norb, 2*norb)
    bot = wd @ x[nup:, :]  # (norb, 2*norb)
    G = jnp.concatenate([top, bot], axis=0).T  # (2*norb, 2*norb)
    return G


def calc_green_g(walker: jax.Array, trial_data: GhfTrial) -> jax.Array:
    """
    Compute full G for generalized walker
    """
    c = trial_data.mo_coeff  # (2*norb, nocc)
    o = c.conj().T @ walker  # (nocc, nocc)
    x = jnp.linalg.solve(o, c.conj().T)  # (nocc, 2*norb)
    G = (walker @ x).T  # (2*norb, 2*norb)
    return G


def calc_overlap_ratio(
    greens: jax.Array,
    update_indices: jax.Array,
    update_constants: jax.Array,
) -> jax.Array:
    """
    Overlap ratio.
    update_indices: [[spin_i, i], [spin_j, j]]
    update_constants: shape (2,) update constants (constants - 1)
    """
    norb = greens.shape[0] // 2
    i_eff, j_eff = _eff_idx(update_indices, norb)
    u0, u1 = update_constants[0], update_constants[1]
    return _ratio_full_rank2(greens, i_eff, j_eff, u0, u1)


def update_green(
    greens: jax.Array,
    update_indices: jax.Array,
    update_constants: jax.Array,
) -> jax.Array:
    """
    Update full G for unrestricted/generalized walker
    """
    norb = greens.shape[0] // 2
    i_eff, j_eff = _eff_idx(update_indices, norb)
    u0, u1 = update_constants[0], update_constants[1]
    return _update_full_rank2(greens, i_eff, j_eff, u0, u1)


def make_ghf_trial_ops(sys: System) -> TrialOps:
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        if sys.nup != sys.ndn:
            raise ValueError("restricted walkers require nup == ndn.")
        return TrialOps(
            overlap=overlap_r, 
            get_rdm1=get_rdm1_u
        )

    if wk == "unrestricted":
        return TrialOps(
            overlap=overlap_u,
            get_rdm1=get_rdm1_u,
            calc_green=calc_green_u,
            update_green=update_green,
            calc_overlap_ratio=calc_overlap_ratio,
        )

    if wk == "generalized":
        return TrialOps(
            overlap=overlap_g,
            get_rdm1=get_rdm1_g,
            calc_green=calc_green_g,
            update_green=update_green,
            calc_overlap_ratio=calc_overlap_ratio,
        )

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")
