from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps
from ..ham.hubbard_nn import HamHubbardNN
from .cpmc import init_prop_state
from .hubbard_nn_cpmc_ops import (
    HubbardNNCpmcCtx,
    HubbardNNCpmcOps,
    _build_prop_ctx,
    make_hubbard_nn_cpmc_ops,
)
from .types import PropOps, PropState, QmcParams


def cpmc_step(
    state: PropState,
    *,
    params: QmcParams,
    trial_data: Any,
    meas_ops: MeasOps,
    cpmc_ops: HubbardNNCpmcOps,
    prop_ctx: HubbardNNCpmcCtx,
) -> PropState:
    """
    One CPMC step with discrete spin HS fields, implemented without fast updates.
    Requires only overlap for a single walker.
    Walkers are assumed to be unrestricted and stored as (w_up, w_dn), each (nw,n,ne).
    """
    nw = wk.n_walkers(state.walkers)

    w_floor = float(getattr(params, "weight_floor", 1.0e-8))
    w_cap = float(getattr(params, "weight_cap", 100.0))
    damping = float(getattr(params, "pop_control_damping", 0.1))

    # one body half step
    walkers = wk.vmap_chunked(
        cpmc_ops.apply_one_body_half, params.n_chunks, in_axes=(0, None)
    )(state.walkers, prop_ctx)
    overlaps = wk.vmap_chunked(
        meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers, trial_data)
    overlaps = jnp.real(overlaps)

    # constrained-path weight update via real overlap ratio
    ratio = jnp.real(overlaps / state.overlaps)
    ratio = jnp.where(ratio < w_floor, 0.0, ratio)
    node_encounters_step = jnp.sum(ratio <= 0.0)
    weights = state.weights * ratio
    weights = jnp.where(weights > w_cap, 0.0, weights)

    # two body
    # onsite interactions
    n_sites = cpmc_ops.n_sites()
    hs_onsite = prop_ctx.hs_constant_onsite  # (2,2)
    key, subkey = jax.random.split(state.rng_key)
    uniform_rns = jax.random.uniform(subkey, (nw, n_sites))

    # scan over sites
    def scanned_fun(carry, x):
        walkers, overlaps, weights, node_encounters = carry
        w_up, w_dn = walkers

        # propose field 0 update at site x
        w0_up = w_up.at[:, x, :].mul(hs[0, 0])
        w0_dn = w_dn.at[:, x, :].mul(hs[0, 1])
        ov0 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
            (w0_up, w0_dn), trial_data
        )
        ov0 = jnp.real(ov0)
        r0 = 0.5 * jnp.real(ov0 / overlaps)
        r0 = jnp.where(r0 < w_floor, 0.0, r0)
        node_encounters += jnp.sum(r0 <= 0.0)

        # propose field 1 update at site x
        w1_up = w_up.at[:, x, :].mul(hs[1, 0])
        w1_dn = w_dn.at[:, x, :].mul(hs[1, 1])
        ov1 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
            (w1_up, w1_dn), trial_data
        )
        ov1 = jnp.real(ov1)
        r1 = 0.5 * jnp.real(ov1 / overlaps)
        r1 = jnp.where(r1 < w_floor, 0.0, r1)
        node_encounters += jnp.sum(r1 <= 0.0)

        # normalize probabilities
        norm = r0 + r1 + 1.0e-13
        p0 = r0 / norm

        rns = uniform_rns[:, x]
        choose0 = rns < p0  # (nw,)

        # update walkers
        c_up = jnp.where(choose0, hs[0, 0], hs[1, 0])
        c_dn = jnp.where(choose0, hs[0, 1], hs[1, 1])

        w_up = w_up.at[:, x, :].mul(c_up[:, None])
        w_dn = w_dn.at[:, x, :].mul(c_dn[:, None])

        overlaps = jnp.where(choose0, ov0, ov1)
        weights = weights * norm

        return ((w_up, w_dn), overlaps, weights, node_encounters), x

    (walkers, overlaps, weights, node_encounters_step), _ = lax.scan(
        scanned_fun,
        (walkers, overlaps, weights, node_encounters_step),
        jnp.arange(n_sites),
    )

    # nearest-neighbor interactions
    n_bonds = cpmc_ops.n_bonds()
    bonds = cpmc_ops.bonds()
    hs_nn = prop_ctx.hs_constant_nn  # (2,2)
    key, subkey = jax.random.split(state.rng_key)
    uniform_rns = jax.random.uniform(subkey, (nw, 4, n_bonds))

    # scan over bonds
    def scanned_fun_nn(carry, x):
        walkers, overlaps, weights, node_encounters = carry
        w_up, w_dn = walkers
        site_i = bonds[x, 0]
        site_j = bonds[x, 1]

        # ---------------------------------------------------------------------
        # up up
        # field 0
        w0_up = (
            w_up.
            .at[:, site_i, :]
            .mul(hs_constant_nn[0, 0])
        )
        w0_up = (
            w0_up
            .at[:, site_j, :]
            .mul(hs_constant_nn[0, 1])
        )
        ov0 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
            (w0_up, w_dn), trial_data
        )
        ov0 = jnp.real(ov0)
        r0 = 0.5 * jnp.real(ov0 / overlaps)
        r0 = jnp.where(r0 < w_floor, 0.0, r0)
        node_encounters += jnp.sum(r0 <= 0.0)

        # field 1
        w1_up = (
            w_up.
            .at[:, site_i, :]
            .mul(hs_constant_nn[1, 0])
        )
        w1_up = (
            w1_up
            .at[:, site_j, :]
            .mul(hs_constant_nn[1, 1])
        )
        ov1 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
            (w1_up, w_dn), trial_data
        )
        ov1 = jnp.real(ov1)
        r1 = 0.5 * jnp.real(ov1 / overlaps)
        r1 = jnp.where(r0 < w_floor, 0.0, r1)
        node_encounters += jnp.sum(r1 <= 0.0)

        # normalize probabilities
        norm = r0 + r1 + 1.0e-13
        p0 = r0 / norm

        rns = uniform_rns[:, 0, x]
        choose0 = rns < p0  # (nw,)

        # update walkers
        w_up = jnp.where(choose0, w0_up, w1_up)
        overlaps = jnp.where(choose0, ov0, ov1)
        weights = weights * norm

        # ---------------------------------------------------------------------
        # up dn
        # field 0
        w0_up = (
            w_up
            .at[:, site_i, :]
            .mul(hs_constant_nn[0, 0])
        )
        w0_dn = (
            w_dn
            .at[:, site_j, :]
            .mul(hs_constant_nn[0, 1])
        )
        ov0 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
            (w0_up, w0_dn), trial_data
        )
        ov0 = jnp.real(ov0)
        r0 = 0.5 * jnp.real(ov0 / overlaps)
        r0 = jnp.where(r0 < w_floor, 0.0, r0)
        node_encounters += jnp.sum(r0 <= 0.0)

        # field 1
        w1_up = (
            w_up.
            .at[:, site_i, :]
            .mul(hs_constant_nn[1, 0])
        )
        w1_dn = (
            w_dn
            .at[:, site_j, :]
            .mul(hs_constant_nn[1, 1])
        )
        ov1 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
            (w1_up, w1_dn), trial_data
        )
        ov1 = jnp.real(ov1)
        r1 = 0.5 * jnp.real(ov1 / overlaps)
        r1 = jnp.where(r0 < w_floor, 0.0, r1)
        node_encounters += jnp.sum(r1 <= 0.0)

        # normalize probabilities
        norm = r0 + r1 + 1.0e-13
        p0 = r0 / norm

        rns = uniform_rns[:, 1, x]
        choose0 = rns < p0  # (nw,)

        # update walkers
        w_up = jnp.where(choose0, w0_up, w1_up)
        w_dn = jnp.where(choose0, w0_dn, w1_dn)
        overlaps = jnp.where(choose0, ov0, ov1)
        weights = weights * norm

        # ---------------------------------------------------------------------
        # dn up
        # field 0
        w0_dn = (
            w_dn
            .at[:, site_i, :]
            .mul(hs_constant_nn[0, 0])
        )
        w0_up = (
            w_up
            .at[:, site_j, :]
            .mul(hs_constant_nn[0, 1])
        )
        ov0 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
            (w0_up, w0_dn), trial_data
        )
        ov0 = jnp.real(ov0)
        r0 = 0.5 * jnp.real(ov0 / overlaps)
        r0 = jnp.where(r0 < w_floor, 0.0, r0)
        node_encounters += jnp.sum(r0 <= 0.0)

        # field 1
        w1_dn = (
            w_dn
            .at[:, site_i, :]
            .mul(hs_constant_nn[1, 0])
        )
        w1_up = (
            w_up
            .at[:, site_j, :]
            .mul(hs_constant_nn[1, 1])
        )
        ov1 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
            (w1_up, w1_dn), trial_data
        )
        ov1 = jnp.real(ov1)
        r1 = 0.5 * jnp.real(ov1 / overlaps)
        r1 = jnp.where(r0 < w_floor, 0.0, r1)
        node_encounters += jnp.sum(r1 <= 0.0)

        # normalize probabilities
        norm = r0 + r1 + 1.0e-13
        p0 = r0 / norm

        rns = uniform_rns[:, 2, x]
        choose0 = rns < p0  # (nw,)

        # update walkers
        w_up = jnp.where(choose0, w0_up, w1_up)
        w_dn = jnp.where(choose0, w0_dn, w1_dn)
        overlaps = jnp.where(choose0, ov0, ov1)
        weights = weights * norm

        # ---------------------------------------------------------------------
        # dn dn
        # field 0
        w0_dn = (
            w_dn.
            .at[:, site_i, :]
            .mul(hs_constant_nn[0, 0])
        )
        w0_dn = (
            w0_dn
            .at[:, site_j, :]
            .mul(hs_constant_nn[0, 1])
        )
        ov0 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
            (w_up, w0_dn), trial_data
        )
        ov0 = jnp.real(ov0)
        r0 = 0.5 * jnp.real(ov0 / overlaps)
        r0 = jnp.where(r0 < w_floor, 0.0, r0)
        node_encounters += jnp.sum(r0 <= 0.0)

        # field 1
        w1_dn = (
            w_dn
            .at[:, site_i, :]
            .mul(hs_constant_nn[1, 0])
        )
        w1_dn = (
            w1_dn
            .at[:, site_j, :]
            .mul(hs_constant_nn[1, 1])
        )
        ov1 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
            (w_up, w1_dn), trial_data
        )
        ov1 = jnp.real(ov1)
        r1 = 0.5 * jnp.real(ov1 / overlaps)
        r1 = jnp.where(r0 < w_floor, 0.0, r1)
        node_encounters += jnp.sum(r1 <= 0.0)

        # normalize probabilities
        norm = r0 + r1 + 1.0e-13
        p0 = r0 / norm

        rns = uniform_rns[:, 3, x]
        choose0 = rns < p0  # (nw,)

        # update walkers
        w_dn = jnp.where(choose0, w0_dn, w1_dn)
        overlaps = jnp.where(choose0, ov0, ov1)
        weights = weights * norm

        return ((w_up, w_dn), overlaps, weights, node_encounters), x

    (walkers, overlaps, weights, node_encounters_step), _ = lax.scan(
        scanned_fun_nn,
        (walkers, overlaps, weights, node_encounters_step),
        jnp.arange(n_bonds),
    )

    # one body half step again
    walkers = cpmc_ops.apply_one_body_half(walkers, prop_ctx)
    overlaps_new = wk.vmap_chunked(
        meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers, trial_data)
    overlaps_new = jnp.real(overlaps_new)

    ratio = jnp.real(overlaps_new / overlaps)
    ratio = jnp.where(ratio < w_floor, 0.0, ratio)
    node_encounters_step += jnp.sum(ratio <= 0.0)
    weights = weights * ratio
    weights = jnp.where(weights > w_cap, 0.0, weights)

    # population control factor and shift update
    weights = weights * jnp.exp(prop_ctx.dt * state.pop_control_ene_shift)
    weights = jnp.where(weights > w_cap, 0.0, weights)

    avg_w = jnp.clip(jnp.mean(weights), min=1.0e-300)
    pop_shift_new = state.e_estimate - damping * (jnp.log(avg_w) / prop_ctx.dt)

    node_encounters_new = state.node_encounters + node_encounters_step

    return PropState(
        walkers=walkers,
        weights=weights,
        overlaps=overlaps_new,
        rng_key=key,
        pop_control_ene_shift=pop_shift_new,
        e_estimate=state.e_estimate,
        node_encounters=node_encounters_new,
    )


def make_prop_ops(ham_data: HamHubbardNN, walker_kind: str) -> PropOps:
    cpmc_ops = make_hubbard_nn_cpmc_ops(ham_data, walker_kind)

    def step(
        state: PropState,
        *,
        params: QmcParams,
        ham_data: Any,
        trial_data: Any,
        trial_ops: TrialOps,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ctx: HubbardNNCpmcCtx,
    ) -> PropState:
        return cpmc_step(
            state,
            params=params,
            trial_data=trial_data,
            meas_ops=meas_ops,
            cpmc_ops=cpmc_ops,
            prop_ctx=prop_ctx,
        )

    def build_prop_ctx(
        ham_data: HamHubbardNN,
        trial_data: Any,
        params: QmcParams,
    ) -> HubbardNNCpmcCtx:
        return _build_prop_ctx(ham_data, params.dt)

    return PropOps(
        init_prop_state=init_prop_state,
        build_prop_ctx=build_prop_ctx,
        step=step,
    )
