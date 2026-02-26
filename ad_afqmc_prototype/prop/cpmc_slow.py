from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps
from ..ham.hubbard import HamHubbard
from .cpmc import init_prop_state
from .hubbard_cpmc_ops import (
    HubbardCpmcCtx,
    HubbardCpmcOps,
    _build_prop_ctx,
    make_hubbard_cpmc_ops,
)
from .types import PropOps, PropState, QmcParams


def cpmc_step(
    state: PropState,
    *,
    params: QmcParams,
    trial_data: Any,
    meas_ops: MeasOps,
    cpmc_ops: HubbardCpmcOps,
    prop_ctx: HubbardCpmcCtx,
) -> PropState:
    """
    One CPMC step with discrete spin HS fields, implemented without fast updates.
    Requires only overlap for a single walker.
    Walkers are assumed to be unrestricted and stored as (w_up, w_dn), each (nw,n,ne).
    """
    key, subkey = jax.random.split(state.rng_key)
    nw = wk.n_walkers(state.walkers)
    n_sites = cpmc_ops.n_sites()

    uniform_rns = jax.random.uniform(subkey, (nw, n_sites))

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

    # constrained-path weight update via real overlap ratio
    ratio = jnp.real(overlaps / state.overlaps)
    ratio = jnp.where(ratio <= w_floor, 0.0, ratio)
    node_encounters_step = jnp.sum(ratio <= 0.0)
    weights = state.weights * ratio
    weights = jnp.where(weights > w_cap, 0.0, weights)

    # two body: scan over sites
    hs = prop_ctx.hs_constant  # (2, 2)

    def scanned_fun(carry, x):
        walkers, overlaps, weights, node_encounters = carry
        w_up, w_dn = walkers

        # propose field 0 update at site x
        w0_up = w_up.at[:, x, :].mul(hs[0, 0])
        w0_dn = w_dn.at[:, x, :].mul(hs[0, 1])
        ov0 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
            (w0_up, w0_dn), trial_data
        )
        r0 = ov0 / overlaps
        r0 = jnp.where(r0 <= w_floor, 0.0, r0)
        node_encounters += jnp.sum(r0 <= 0.0)

        # propose field 1 update at site x
        w1_up = w_up.at[:, x, :].mul(hs[1, 0])
        w1_dn = w_dn.at[:, x, :].mul(hs[1, 1])
        ov1 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
            (w1_up, w1_dn), trial_data
        )
        r1 = ov1 / overlaps
        r1 = jnp.where(r1 <= w_floor, 0.0, r1)
        node_encounters += jnp.sum(r1 <= 0.0)

        # normalize probabilities
        p0 = 0.5 * r0.real
        p1 = 0.5 * r1.real
        norm = p0 + p1 + 1.0e-13
        p0 = p0 / norm

        choose0 = uniform_rns[:, x] < p0  # (nw,)

        # update walkers
        c_up = jnp.where(choose0, hs[0, 0], hs[1, 0])
        c_dn = jnp.where(choose0, hs[0, 1], hs[1, 1])
        w_up = w_up.at[:, x, :].mul(c_up[:, None])
        w_dn = w_dn.at[:, x, :].mul(c_dn[:, None])

        # update overlap and weights
        overlaps = jnp.where(choose0, ov0, ov1)
        weights = weights * norm

        return ((w_up, w_dn), overlaps, weights, node_encounters), x

    (walkers, overlaps, weights, node_encounters_step), _ = lax.scan(
        scanned_fun,
        (walkers, overlaps, weights, node_encounters_step),
        jnp.arange(n_sites),
    )

    # one body half step again
    walkers = cpmc_ops.apply_one_body_half(walkers, prop_ctx)
    overlaps_new = wk.vmap_chunked(
        meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers, trial_data)
    ratio = jnp.real(overlaps_new / overlaps)
    ratio = jnp.where(ratio <= w_floor, 0.0, ratio)
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


def make_prop_ops(ham_data: HamHubbard, walker_kind: str) -> PropOps:
    cpmc_ops = make_hubbard_cpmc_ops(ham_data, walker_kind)

    def step(
        state: PropState,
        *,
        params: QmcParams,
        ham_data: Any,
        trial_data: Any,
        trial_ops: TrialOps,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ctx: HubbardCpmcCtx,
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
        ham_data: HamHubbard,
        trial_data: Any,
        params: QmcParams,
    ) -> HubbardCpmcCtx:
        return _build_prop_ctx(ham_data, params.dt)

    return PropOps(
        init_prop_state=init_prop_state,
        build_prop_ctx=build_prop_ctx,
        step=step,
    )
