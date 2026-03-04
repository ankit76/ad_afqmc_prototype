from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.levels import LevelPack, LevelSpec
from ..core.ops import MeasOps, k_energy, k_force_bias, o_rdm1
from ..core.system import System
from ..ham.chol import HamChol, slice_ham_level
from ..trial.cisd import CisdTrial
from ..trial.cisd import overlap_r as cisd_overlap_r
from ..trial.cisd import slice_trial_level


def _greens_restricted(walker: jax.Array, nocc: int) -> jax.Array:
    wocc = walker[:nocc, :]  # (nocc, nocc)
    return jnp.linalg.solve(wocc.T, walker.T)  # (nocc, norb)


def _greenp_from_green(green: jax.Array) -> jax.Array:
    """
    green_occ = green[:, nocc:]
    greenp = vstack([green_occ, -I(nvir)])   shape (norb, nvir)
    """
    nocc = green.shape[0]
    nvir = green.shape[1] - nocc
    green_occ = green[:, nocc:]
    return jnp.vstack((green_occ, -jnp.eye(nvir, dtype=green.dtype)))


@dataclass(frozen=True)
class CisdMeasCfg:
    memory_mode: str = "low"  # or Literal["low","high"]
    mixed_real_dtype: jnp.dtype = jnp.float64
    mixed_complex_dtype: jnp.dtype = jnp.complex128
    mixed_real_dtype_testing: jnp.dtype = jnp.float32
    mixed_complex_dtype_testing: jnp.dtype = jnp.complex64


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CisdMeasCtx:
    rot_chol: jax.Array  # (n_chol, nocc, norb)
    lci1: jax.Array  # (n_chol, norb, nocc)
    cfg: CisdMeasCfg  # static

    def tree_flatten(self):
        children = (self.rot_chol, self.lci1)
        aux = (self.cfg,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cfg,) = aux
        rot_chol, lci1 = children
        return cls(rot_chol=rot_chol, lci1=lci1, cfg=cfg)


def slice_meas_ctx_chol(ctx: CisdMeasCtx, nchol_keep: int | None) -> CisdMeasCtx:
    if nchol_keep is None:
        return ctx
    return CisdMeasCtx(
        rot_chol=ctx.rot_chol[:nchol_keep],
        lci1=ctx.lci1[:nchol_keep],
        cfg=ctx.cfg,
    )


def build_meas_ctx(
    ham_data: HamChol, trial_data: CisdTrial, cfg: CisdMeasCfg = CisdMeasCfg()
) -> CisdMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("CISD MeasOps currently assumes HamChol.basis == 'restricted'.")

    chol = ham_data.chol  # (n_chol, norb, norb)
    nocc = trial_data.nocc

    rot_chol = chol[:, :nocc, :]  # (n_chol, nocc, norb)

    lci1 = jnp.einsum(
        "git,pt->gip",
        chol[:, :, nocc:],
        trial_data.ci1,
        optimize="optimal",
    )  # (n_chol, norb, nocc)

    return CisdMeasCtx(rot_chol=rot_chol, lci1=lci1, cfg=cfg)


def make_level_pack(
    *,
    ham_data: HamChol,
    trial_data: CisdTrial,
    level: LevelSpec,
    orb_fullchol_ctx: CisdMeasCtx | None = None,
    orb_fullchol_ham: HamChol | None = None,
    memory_mode: str = "low",
    mixed_precision: bool = True,
) -> LevelPack:
    cfg = CisdMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float32,
        mixed_complex_dtype_testing=jnp.complex64,
    )
    if level.nvir_keep is None:
        trial_orb = trial_data
        norb_keep = None
    else:
        trial_orb = slice_trial_level(trial_data, level.nvir_keep)
        norb_keep = int(trial_data.nocc) + int(level.nvir_keep)

    if orb_fullchol_ham is None:
        ham_orb_fullchol = slice_ham_level(ham_data, norb_keep=norb_keep, nchol_keep=None)
    else:
        ham_orb_fullchol = orb_fullchol_ham

    if orb_fullchol_ctx is None:
        ctx_orb_fullchol = build_meas_ctx(ham_orb_fullchol, trial_orb, cfg=cfg)
    else:
        ctx_orb_fullchol = orb_fullchol_ctx

    if level.nchol_keep is None:
        ham_lvl = ham_orb_fullchol
        ctx_lvl = ctx_orb_fullchol
    else:
        ham_lvl = slice_ham_level(ham_orb_fullchol, norb_keep=None, nchol_keep=level.nchol_keep)
        ctx_lvl = slice_meas_ctx_chol(ctx_orb_fullchol, level.nchol_keep)

    return LevelPack(
        level=level,
        ham_data=ham_lvl,
        trial_data=trial_orb,
        meas_ctx=ctx_lvl,
        norb_keep=norb_keep,
    )


def force_bias_kernel_rw_rh(
    walker: jax.Array, ham_data: HamChol, meas_ctx: CisdMeasCtx, trial_data: CisdTrial
) -> jax.Array:
    ci1, ci2 = trial_data.ci1, trial_data.ci2
    nocc = trial_data.nocc

    green = _greens_restricted(walker, nocc)  # (nocc, norb)
    green_occ = green[:, nocc:]  # (nocc, nvir)
    greenp = _greenp_from_green(green)  # (norb, nvir)

    chol = ham_data.chol
    rot_chol = meas_ctx.rot_chol

    lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
    fb_0 = 2.0 * lg

    # singles
    ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
    ci1gp = jnp.einsum("pt,it->pi", ci1, greenp, optimize="optimal")  # (nocc, norb)
    gci1gp = jnp.einsum("pj,pi->ij", green, ci1gp, optimize="optimal")  # (norb, norb)

    fb_1_1 = 4.0 * ci1g * lg
    fb_1_2 = -2.0 * jnp.einsum(
        "gij,ij->g",
        chol.astype(meas_ctx.cfg.mixed_real_dtype),
        gci1gp.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    fb_1 = fb_1_1 + fb_1_2

    # doubles
    ci2g_c = jnp.einsum(
        "ptqu,pt->qu",
        ci2.astype(meas_ctx.cfg.mixed_real_dtype),
        green_occ.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )  # (nocc, nvir)
    ci2g_e = jnp.einsum(
        "ptqu,pu->qt",
        ci2.astype(meas_ctx.cfg.mixed_real_dtype),
        green_occ.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )  # (nocc, nvir)

    cisd_green_c = (greenp @ ci2g_c.T) @ green  # (norb, norb)
    cisd_green_e = (greenp @ ci2g_e.T) @ green  # (norb, norb)
    cisd_green = -4.0 * cisd_green_c + 2.0 * cisd_green_e

    ci2g = 4.0 * ci2g_c - 2.0 * ci2g_e  # (nocc, nvir)
    gci2g = jnp.einsum("qu,qu->", ci2g, green_occ, optimize="optimal")

    fb_2_1 = lg * gci2g
    fb_2_2 = jnp.einsum(
        "gij,ij->g",
        chol.astype(meas_ctx.cfg.mixed_real_dtype),
        cisd_green.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    fb_2 = fb_2_1 + fb_2_2

    overlap = 1.0 + 2.0 * ci1g + 0.5 * gci2g
    return (fb_0 + fb_1 + fb_2) / overlap


def energy_kernel_rw_rh(
    walker: jax.Array, ham_data: HamChol, meas_ctx: CisdMeasCtx, trial_data: CisdTrial
) -> jax.Array:
    ci1, ci2 = trial_data.ci1, trial_data.ci2
    nocc = trial_data.nocc

    green = _greens_restricted(walker, nocc)  # (nocc, norb)
    green_occ = green[:, nocc:]  # (nocc, nvir)
    greenp = _greenp_from_green(green)  # (norb, nvir)

    h1 = ham_data.h1
    chol = ham_data.chol
    rot_chol = meas_ctx.rot_chol

    # 0 body
    e0 = ham_data.h0

    # 1 body
    hg = jnp.einsum("pj,pj->", h1[:nocc, :], green, optimize="optimal")
    e1_0 = 2.0 * hg

    # singles
    ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
    e1_1_1 = 4.0 * ci1g * hg
    gpci1 = greenp @ ci1.T  # (norb, nocc)
    ci1_green = gpci1 @ green  # (norb, norb)
    e1_1_2 = -2.0 * jnp.einsum("ij,ij->", h1, ci1_green, optimize="optimal")
    e1_1 = e1_1_1 + e1_1_2

    # doubles
    ci2g_c = jnp.einsum(
        "ptqu,pt->qu",
        ci2.astype(meas_ctx.cfg.mixed_real_dtype),
        green_occ.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    ci2g_e = jnp.einsum(
        "ptqu,pu->qt",
        ci2.astype(meas_ctx.cfg.mixed_real_dtype),
        green_occ.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    ci2_green_c = (greenp @ ci2g_c.T) @ green
    ci2_green_e = (greenp @ ci2g_e.T) @ green
    ci2_green = 2.0 * ci2_green_c - 1.0 * ci2_green_e

    ci2g = 2.0 * ci2g_c - 1.0 * ci2g_e
    gci2g = jnp.einsum("qu,qu->", ci2g, green_occ, optimize="optimal")

    e1_2_1 = 2.0 * hg * gci2g
    e1_2_2 = -2.0 * jnp.einsum("ij,ij->", h1, ci2_green, optimize="optimal")
    e1_2 = e1_2_1 + e1_2_2

    e1 = e1_0 + e1_1 + e1_2

    # 2 body
    lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")  # (n_chol,)
    lg1 = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")  # (n_chol,nocc,nocc)

    e2_0_1 = 2.0 * (lg @ lg)
    e2_0_2 = -jnp.sum(lg1 * jnp.swapaxes(lg1, -1, -2))
    e2_0 = e2_0_1 + e2_0_2

    # singles
    e2_1_1 = 2.0 * e2_0 * ci1g
    lci1g = jnp.einsum(
        "gij,ij->g",
        chol.astype(meas_ctx.cfg.mixed_real_dtype),
        ci1_green.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    e2_1_2 = -2.0 * (lci1g @ lg)

    ci1g1 = ci1 @ green_occ.T  # (nocc, nocc)
    e2_1_3_1 = jnp.einsum("gpq,gqr,rp->", lg1, lg1, ci1g1, optimize="optimal")
    lci1g_mat = jnp.einsum("gip,qi->gpq", meas_ctx.lci1, green, optimize="optimal")
    e2_1_3_2 = -jnp.einsum("gpq,gqp->", lci1g_mat, lg1, optimize="optimal")
    e2_1_3 = e2_1_3_1 + e2_1_3_2

    e2_1 = e2_1_1 + 2.0 * (e2_1_2 + e2_1_3)

    # doubles
    e2_2_1 = e2_0 * gci2g
    lci2g = jnp.einsum(
        "gij,ij->g",
        chol.astype(meas_ctx.cfg.mixed_real_dtype),
        ci2_green.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    e2_2_2_1 = -(lci2g @ lg)

    if meas_ctx.cfg.memory_mode == "low":
        dtype_acc = jnp.result_type(walker, ci1, ci2)
        zero = jnp.array(0.0, dtype=dtype_acc)
        ci2_t = ci2.astype(meas_ctx.cfg.mixed_real_dtype)

        def scan_over_chol(carry, x):
            e22_acc, e23_acc = carry
            chol_i, rot_chol_i = x  # (norb,norb), (nocc,norb)

            gl_i = jnp.einsum("pj,ji->pi", green, chol_i, optimize="optimal")
            lci2_green_i = jnp.einsum("pi,ji->pj", rot_chol_i, ci2_green, optimize="optimal")

            e22_acc = e22_acc + 0.5 * jnp.einsum("pi,pi->", gl_i, lci2_green_i, optimize="optimal")

            glgp_i = jnp.einsum("pi,it->pt", gl_i, greenp, optimize="optimal").astype(
                meas_ctx.cfg.mixed_complex_dtype_testing
            )
            l2ci2_1 = jnp.einsum(
                "pt,qu,ptqu->",
                glgp_i,
                glgp_i,
                ci2_t,
                optimize="optimal",
            )
            l2ci2_2 = jnp.einsum(
                "pu,qt,ptqu->",
                glgp_i,
                glgp_i,
                ci2_t,
                optimize="optimal",
            )
            e23_acc = e23_acc + (2.0 * l2ci2_1 - l2ci2_2)

            return (e22_acc, e23_acc), zero

        (e2_2_2_2, e2_2_3), _ = lax.scan(scan_over_chol, (zero, zero), (chol, rot_chol))
    else:
        lci2_green = jnp.einsum("gpi,ji->gpj", rot_chol, ci2_green, optimize="optimal")
        gl = jnp.einsum(
            "pj,gji->gpi",
            green.astype(meas_ctx.cfg.mixed_complex_dtype),
            chol.astype(meas_ctx.cfg.mixed_real_dtype),
            optimize="optimal",
        )
        e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lci2_green, optimize="optimal")

        glgp = jnp.einsum("gpi,it->gpt", gl, greenp, optimize="optimal").astype(
            meas_ctx.cfg.mixed_complex_dtype_testing
        )
        l2ci2_1 = jnp.einsum(
            "gpt,gqu,ptqu->g",
            glgp,
            glgp,
            ci2.astype(meas_ctx.cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        l2ci2_2 = jnp.einsum(
            "gpu,gqt,ptqu->g",
            glgp,
            glgp,
            ci2.astype(meas_ctx.cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        e2_2_3 = 2.0 * l2ci2_1.sum() - l2ci2_2.sum()

    e2_2_2 = 4.0 * (e2_2_2_1 + e2_2_2_2)
    e2_2 = e2_2_1 + e2_2_2 + e2_2_3

    e2 = e2_0 + e2_1 + e2_2

    overlap = 1.0 + 2.0 * ci1g + gci2g
    return (e1 + e2) / overlap + e0


def rdm1_kernel_rw(
    walker: jax.Array, ham_data: jax.Array, meas_ctx: CisdMeasCtx, trial_data: CisdTrial
) -> jax.Array:
    """
    Mixed estimator 1RDM for CISD trial with restricted walker.

    gamma = G^0 - (ci1_green + ci2_green) / Omega

    Returns (2, norb, norb): [alpha, beta] (identical for restricted).
    """
    ci1, ci2 = trial_data.ci1, trial_data.ci2
    nocc = trial_data.nocc
    norb = trial_data.norb

    green = _greens_restricted(walker, nocc)  # (nocc, norb)
    green_occ = green[:, nocc:]  # (nocc, nvir)
    greenp = _greenp_from_green(green)  # (norb, nvir)

    # HF Green's function: G^0[:, :nocc] = green.T, G^0[:, nocc:] = 0
    g0 = jnp.zeros((norb, norb), dtype=green.dtype)
    g0 = g0.at[:, :nocc].set(green.T)

    # overlap components
    ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")

    # singles correction
    ci1_green = (greenp @ ci1.T) @ green  # (norb, norb)

    # doubles correction
    ci2g_c = jnp.einsum(
        "ptqu,pt->qu",
        ci2.astype(meas_ctx.cfg.mixed_real_dtype),
        green_occ.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    ci2g_e = jnp.einsum(
        "ptqu,pu->qt",
        ci2.astype(meas_ctx.cfg.mixed_real_dtype),
        green_occ.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    ci2_green = 2.0 * (greenp @ ci2g_c.T) @ green - (greenp @ ci2g_e.T) @ green

    ci2g = 2.0 * ci2g_c - ci2g_e
    gci2g = jnp.einsum("qu,qu->", ci2g, green_occ, optimize="optimal")

    overlap = 1.0 + 2.0 * ci1g + gci2g
    gamma = g0 - (ci1_green + ci2_green) / overlap

    return jnp.stack([gamma, gamma], axis=0)


def make_cisd_meas_ops(
    sys: System,
    memory_mode: str = "high",
    mixed_precision: bool = True,
    testing: bool = False,
) -> MeasOps:
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"CISD MeasOps currently supports only restricted walkers, got: {sys.walker_kind}"
        )

    cfg = CisdMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )

    return MeasOps(
        overlap=cisd_overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg),
        kernels={k_force_bias: force_bias_kernel_rw_rh, k_energy: energy_kernel_rw_rh},
        observables={o_rdm1: rdm1_kernel_rw},
    )
