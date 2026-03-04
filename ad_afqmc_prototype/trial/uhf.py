from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UhfTrial:
    """
    Unrestricted HF trial.
    """
    mo_coeff_a: jax.Array  # (norb, nocc[0])
    mo_coeff_b: jax.Array  # (norb, nocc[1])

    @property
    def norb(self) -> int:
        return int(self.mo_coeff_a.shape[0])

    @property
    def nocc(self) -> tuple[int, int]:
        return (int(self.mo_coeff_a.shape[1]), int(self.mo_coeff_b.shape[1]))

    def tree_flatten(self):
        return (self.mo_coeff_a, self.mo_coeff_b), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        mo_coeff_a, mo_coeff_b = children
        return cls(mo_coeff_a=mo_coeff_a, mo_coeff_b=mo_coeff_b)


def _det(m: jax.Array) -> jax.Array:
    return jnp.linalg.det(m)


def get_rdm1(trial_data: UhfTrial) -> jax.Array:
    cu = trial_data.mo_coeff_a
    cd = trial_data.mo_coeff_b
    dm_u = cu @ cu.conj().T  # (norb, norb)
    dm_d = cd @ cd.conj().T  # (norb, norb)
    return jnp.stack([dm_u, dm_d], axis=0)  # (2, norb, norb)


def overlap_r(walker: jax.Array, trial_data: UhfTrial) -> jax.Array:
    assert trial_data.nocc[0] == trial_data.nocc[1]
    w = walker
    ou = trial_data.mo_coeff_a.conj().T @ w  # (nocc[0], nocc[0])
    od = trial_data.mo_coeff_b.conj().T @ w  # (nocc[1], nocc[1])
    return _det(ou) * _det(od)


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: UhfTrial) -> jax.Array:
    wu, wd = walker
    ou = trial_data.mo_coeff_a.conj().T @ wu  # (nocc[0], nocc[0])
    od = trial_data.mo_coeff_b.conj().T @ wd  # (nocc[1], nocc[1])
    return _det(ou) * _det(od)


def overlap_g(walker: jax.Array, trial_data: UhfTrial) -> jax.Array:
    norb = trial_data.norb
    cuH = trial_data.mo_coeff_a.conj().T  # (nocc[0], norb)
    cdH = trial_data.mo_coeff_b.conj().T  # (nocc[1], norb)
    top = cuH @ walker[:norb, :]  # (nocc[0], sum(nocc))
    bot = cdH @ walker[norb:, :]  # (nocc[1], sum(nocc))
    m = jnp.vstack([top, bot])  # (sum(nocc), sum(nocc))
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
    walker: tuple[jax.Array, jax.Array], trial_data: UhfTrial
) -> jax.Array:
    """
    Compute full G for unrestricted walker
    """
    wu, wd = walker
    norb = wu.shape[0]
    cu = trial_data.mo_coeff_a  # (norb, nocc[0])
    cd = trial_data.mo_coeff_b  # (norb, nocc[1])
    ou = cu.conj().T @ wu  # (nocc[0], nocc[0])
    od = cd.conj().T @ wd  # (nocc[1], nocc[1])
    xu = jnp.linalg.solve(ou, cu.conj().T)  # (nocc[0], norb)
    xd = jnp.linalg.solve(od, cd.conj().T)  # (nocc[1], norb)
    Gu = wu @ xu  # (norb, norb)
    Gd = wd @ xd  # (norb, norb)
    G = jsp.linalg.block_diag(Gu, Gd).T # (2*norb, 2*norb)
    return G


def calc_green_g(walker: jax.Array, trial_data: GhfTrial) -> jax.Array:
    """
    Compute full G for generalized walker
    """
    c = jsp.linalg.block_diag(
        trial_data.mo_coeff_a, 
        trial_data.mo_coeff_b
    ) # (2*norb, nocc[0]+nocc[1])
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


def make_uhf_trial_ops(sys: System) -> TrialOps:
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        if sys.nup != sys.ndn:
            raise ValueError("restricted walkers require nup == ndn.")
        overlap_fn = overlap_r
        get_rdm1_fn = get_rdm1
    elif wk == "unrestricted":
        overlap_fn = overlap_u
        get_rdm1_fn = get_rdm1
    elif wk == "generalized":
        overlap_fn = overlap_g
        get_rdm1_fn = get_rdm1
    else:
        raise ValueError(f"unknown walker_kind: {sys.walker_kind}")

    return TrialOps(
        overlap=overlap_fn,
        get_rdm1=get_rdm1_fn,
    )


def make_uhf_trial_data(data: dict, sys: System) -> UhfTrial:
    if "mo_a" in data and "mo_b" in data:
        mo_a = jnp.asarray(data["mo_a"])
        mo_b = jnp.asarray(data["mo_b"])
    elif "mo" in data:
        mo_a = jnp.asarray(data["mo"])
        mo_b = jnp.asarray(data["mo"])
    else:
        raise KeyError("Failed to find the trial coeff.")

    mo_a = mo_a[:, : sys.nup]
    mo_b = mo_b[:, : sys.ndn]

    return UhfTrial(mo_a, mo_b)
