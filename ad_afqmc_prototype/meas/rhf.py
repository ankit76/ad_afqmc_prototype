from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.rhf import RhfTrial, overlap_g, overlap_r, overlap_u


def _half_green_from_overlap_matrix(w: jax.Array, ovlp_mat: jax.Array) -> jax.Array:
    """
    green_half = (w @ inv(ovlp_mat)).T
    """
    return jnp.linalg.solve(ovlp_mat.T, w.T)


def force_bias_kernel_r(
    walker: jax.Array, ham_data: Any, meas_ctx: RhfMeasCtx, trial_data: RhfTrial
) -> jax.Array:
    m = trial_data.mo_coeff.conj().T @ walker
    g_half = _half_green_from_overlap_matrix(walker, m)  # (nocc, norb)
    # RHF: factor 2 for (up+dn)
    return 2.0 * (meas_ctx.rot_chol_flat @ g_half.reshape(-1))


def energy_kernel_r(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: RhfMeasCtx,
    trial_data: RhfTrial,
    nchol: Optional[int] = None,
) -> jax.Array:
    if nchol is None:
        nchol = ham_data.chol.shape[0]
    m = trial_data.mo_coeff.conj().T @ walker
    g_half = _half_green_from_overlap_matrix(walker, m)  # (nocc, norb)

    e0 = ham_data.h0
    e1 = 2.0 * jnp.sum(g_half * meas_ctx.rot_h1)

    f = jnp.einsum(
        "gij,jk->gik", meas_ctx.rot_chol[:nchol], g_half.T, optimize="optimal"
    )
    c = jax.vmap(jnp.trace)(f)
    exc = jnp.sum(jax.vmap(lambda x: x * x.T)(f))
    e2 = 2.0 * jnp.sum(c * c) - exc

    return e0 + e1 + e2


def force_bias_kernel_u(
    walker: tuple[jax.Array, jax.Array],
    ham_data: Any,
    meas_ctx: RhfMeasCtx,
    trial_data: RhfTrial,
) -> jax.Array:
    wu, wd = walker
    mu = trial_data.mo_coeff.conj().T @ wu
    md = trial_data.mo_coeff.conj().T @ wd
    gu = _half_green_from_overlap_matrix(wu, mu)  # (nocc_a, norb)
    gd = _half_green_from_overlap_matrix(wd, md)  # (nocc_b, norb)
    g = gu + gd
    return meas_ctx.rot_chol_flat @ g.reshape(-1)


def energy_kernel_u(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: RhfMeasCtx,
    trial_data: RhfTrial,
) -> jax.Array:
    wu, wd = walker
    mu = trial_data.mo_coeff.conj().T @ wu
    md = trial_data.mo_coeff.conj().T @ wd
    gu = _half_green_from_overlap_matrix(wu, mu)
    gd = _half_green_from_overlap_matrix(wd, md)

    e0 = ham_data.h0
    e1 = jnp.sum((gu + gd) * meas_ctx.rot_h1)

    f_up = jnp.einsum("gij,jk->gik", meas_ctx.rot_chol, gu.T, optimize="optimal")
    f_dn = jnp.einsum("gij,jk->gik", meas_ctx.rot_chol, gd.T, optimize="optimal")
    c_up = jax.vmap(jnp.trace)(f_up)
    c_dn = jax.vmap(jnp.trace)(f_dn)
    exc_up = jnp.sum(jax.vmap(lambda x: x * x.T)(f_up))
    exc_dn = jnp.sum(jax.vmap(lambda x: x * x.T)(f_dn))

    e2 = (
        jnp.sum(c_up * c_up)
        + jnp.sum(c_dn * c_dn)
        + 2.0 * jnp.sum(c_up * c_dn)
        - exc_up
        - exc_dn
    ) / 2.0

    return e0 + e1 + e2


def force_bias_kernel_g(
    walker: jax.Array, ham_data: Any, meas_ctx: RhfMeasCtx, trial_data: RhfTrial
) -> jax.Array:
    norb, nocc = trial_data.norb, trial_data.nocc
    cH = trial_data.mo_coeff.conj().T
    top = cH @ walker[:norb, :]
    bot = cH @ walker[norb:, :]
    m = jnp.vstack([top, bot])  # (2*nocc, 2*nocc)

    g_half = _half_green_from_overlap_matrix(walker, m)  # (2*nocc, 2*norb)
    g_up = g_half[:nocc, :norb]
    g_dn = g_half[nocc:, norb:]
    g = g_up + g_dn
    return meas_ctx.rot_chol_flat @ g.reshape(-1)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RhfMeasCtx:
    # half-rotated:
    rot_h1: jax.Array  # (nocc, norb)
    rot_chol: jax.Array  # (n_chol, nocc, norb)
    rot_chol_flat: jax.Array  # (n_chol, nocc*norb)

    def tree_flatten(self):
        return (self.rot_h1, self.rot_chol, self.rot_chol_flat), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        rot_h1, rot_chol, rot_chol_flat = children
        return cls(
            rot_h1=rot_h1,
            rot_chol=rot_chol,
            rot_chol_flat=rot_chol_flat,
        )


def build_meas_ctx(ham_data: HamChol, trial_data: RhfTrial) -> RhfMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("RHF MeasOps currently assumes HamChol.basis == 'restricted'.")
    cH = trial_data.mo_coeff.conj().T  # (nocc, norb)
    rot_h1 = cH @ ham_data.h1  # (nocc, norb)
    rot_chol = jnp.einsum("pi,gij->gpj", cH, ham_data.chol, optimize="optimal")
    rot_chol_flat = rot_chol.reshape(rot_chol.shape[0], -1)
    return RhfMeasCtx(rot_h1=rot_h1, rot_chol=rot_chol, rot_chol_flat=rot_chol_flat)


def make_rhf_meas_ops(sys: System) -> MeasOps:
    wk = sys.walker_kind.lower()
    if wk == "restricted":
        return MeasOps(
            overlap=overlap_r,
            build_meas_ctx=build_meas_ctx,
            kernels={k_force_bias: force_bias_kernel_r, k_energy: energy_kernel_r},
        )

    if wk == "unrestricted":
        return MeasOps(
            overlap=overlap_u,
            build_meas_ctx=build_meas_ctx,
            kernels={k_force_bias: force_bias_kernel_u, k_energy: energy_kernel_u},
        )

    if wk == "generalized":
        return MeasOps(
            overlap=overlap_g,
            build_meas_ctx=build_meas_ctx,
            kernels={k_force_bias: force_bias_kernel_g},
        )

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")
