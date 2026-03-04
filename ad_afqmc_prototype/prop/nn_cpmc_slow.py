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
    Requires:
      - trial_ops.overlap
    Walkers: 
        - unrestricted (w_up, w_dn), each (nw, n_sites, n_elec_spin).
        - generalized, (nw, 2*n_sites, n_elec).
    """
    walkers = state.walkers
    nw = wk.n_walkers(walkers)
    walker_kind = prop_ctx.walker_kind
    node_encounters_step = jnp.asarray(0)

    w_floor = float(getattr(params, "weight_floor", 1.0e-8))
    w_cap = float(getattr(params, "weight_cap", 100.0))
    damping = float(getattr(params, "pop_control_damping", 0.1))

    # one body half step
    walkers = wk.vmap_chunked(
        cpmc_ops.apply_one_body_half, params.n_chunks, in_axes=(0, None)
    )(walkers, prop_ctx)
    overlaps = wk.vmap_chunked(
        meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers, trial_data)

    # constrained-path weight update via real overlap ratio
    ratio = jnp.real(overlaps / state.overlaps)
    ratio = jnp.where(ratio <= w_floor, 0.0, ratio)
    node_encounters_step = jnp.sum(ratio <= 0.0)
    weights = state.weights * ratio
    weights = jnp.where(weights > w_cap, 0.0, weights)

    # two body
    key, k1, k2 = jax.random.split(state.rng_key, 3)

    # onsite interactions
    n_sites = cpmc_ops.n_sites()
    hs_onsite = prop_ctx.hs_constant_onsite  # (2,2)
    uniform_rns_onsite = jax.random.uniform(k1, (nw, n_sites)) # uniform HS fields

    # scan over sites
    def scanned_fun(carry, x):
        walkers, overlaps, weights, node_encounters = carry
        if walker_kind == "unrestricted":
            # propose field 0 update at site x
            w0_up = walkers[0].at[:, x, :].mul(hs_onsite[0, 0])
            w0_dn = walkers[1].at[:, x, :].mul(hs_onsite[0, 1])
            w0 = (w0_up, w0_dn)

            # propose field 1 update at site x
            w1_up = walkers[0].at[:, x, :].mul(hs_onsite[1, 0])
            w1_dn = walkers[1].at[:, x, :].mul(hs_onsite[1, 1])
            w1 = (w1_up, w1_dn)

        elif walker_kind == "generalized":
            # propose field 0 update at site x
            w0 = walkers.at[:, x, :].mul(hs_onsite[0, 0])
            w0 = w0.at[:, x+n_sites, :].mul(hs_onsite[0, 1])

            # propose field 1 update at site x
            w1 = walkers.at[:, x, :].mul(hs_onsite[1, 0])
            w1 = w1.at[:, x+n_sites, :].mul(hs_onsite[1, 1])

        ov0 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
            w0, trial_data
        )
        r0 = ov0 / overlaps
        r0 = jnp.where(r0 <= w_floor, 0.0, r0)
        node_encounters += jnp.sum(r0 <= 0.0)

        ov1 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
            w1, trial_data
        )
        r1 = ov1 / overlaps
        r1 = jnp.where(r1 <= w_floor, 0.0, r1)
        node_encounters += jnp.sum(r1 <= 0.0)

        # normalize probabilities
        p0 = 0.5 * r0.real
        p1 = 0.5 * r1.real
        norm = p0 + p1 + 1.0e-13
        p0 = p0 / norm

        # random choice
        choose0 = uniform_rns_onsite[:, x] < p0  # (nw,)

        # update walkers
        if walker_kind == "unrestricted":
            w_up = jnp.where(choose0.reshape(-1, 1, 1), w0[0], w1[0])
            w_dn = jnp.where(choose0.reshape(-1, 1, 1), w0[1], w1[1])
            walkers = (w_up, w_dn)
        
        elif walker_kind == "generalized":
            walkers = jnp.where(choose0.reshape(-1, 1, 1), w0, w1)

        # update overlap and weights
        overlaps = jnp.where(choose0, ov0, ov1)
        weights = weights * norm

        return (walkers, overlaps, weights, node_encounters), x

    (walkers, overlaps, weights, node_encounters_step), _ = lax.scan(
        scanned_fun,
        (walkers, overlaps, weights, node_encounters_step),
        jnp.arange(n_sites),
    )

    # nearest-neighbor interactions
    n_bonds = cpmc_ops.n_bonds()
    bonds = cpmc_ops.bonds()
    hs_nn = prop_ctx.hs_constant_nn  # (2,2)
    uniform_rns_nn = jax.random.uniform(k2, (nw, 4, n_bonds))

    # scan over bonds
    def scanned_fun_nn(carry, x):
        walkers, overlaps, weights, node_encounters = carry
        site_i = bonds[x, 0]
        site_j = bonds[x, 1]
        
        if walker_kind == "unrestricted":
            w_up, w_dn = walkers

            # ---------------------------------------------------------------------
            # up up
            # field 0
            w0_up = (
                w_up
                .at[:, site_i, :]
                .mul(hs_nn[0, 0])
            )
            w0_up = (
                w0_up
                .at[:, site_j, :]
                .mul(hs_nn[0, 1])
            )
            ov0 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
                (w0_up, w_dn), trial_data
            )
            r0 = 0.5 * jnp.real(ov0 / overlaps)
            r0 = jnp.where(r0 < w_floor, 0.0, r0)
            node_encounters += jnp.sum(r0 <= 0.0)

            # field 1
            w1_up = (
                w_up
                .at[:, site_i, :]
                .mul(hs_nn[1, 0])
            )
            w1_up = (
                w1_up
                .at[:, site_j, :]
                .mul(hs_nn[1, 1])
            )
            ov1 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
                (w1_up, w_dn), trial_data
            )
            r1 = 0.5 * jnp.real(ov1 / overlaps)
            r1 = jnp.where(r0 < w_floor, 0.0, r1)
            node_encounters += jnp.sum(r1 <= 0.0)

            # normalize probabilities
            norm = r0 + r1 + 1.0e-13
            p0 = r0 / norm

            # update walkers
            choose0 = uniform_rns_nn[:, 0, x] < p0 # (nw,)
            w_up = jnp.where(choose0.reshape(-1, 1, 1), w0_up, w1_up)
            overlaps = jnp.where(choose0, ov0, ov1)
            weights = weights * norm

            # ---------------------------------------------------------------------
            # up dn
            # field 0
            w0_up = (
                w_up
                .at[:, site_i, :]
                .mul(hs_nn[0, 0])
            )
            w0_dn = (
                w_dn
                .at[:, site_j, :]
                .mul(hs_nn[0, 1])
            )
            ov0 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
                (w0_up, w0_dn), trial_data
            )
            r0 = 0.5 * jnp.real(ov0 / overlaps)
            r0 = jnp.where(r0 < w_floor, 0.0, r0)
            node_encounters += jnp.sum(r0 <= 0.0)

            # field 1
            w1_up = (
                w_up
                .at[:, site_i, :]
                .mul(hs_nn[1, 0])
            )
            w1_dn = (
                w_dn
                .at[:, site_j, :]
                .mul(hs_nn[1, 1])
            )
            ov1 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
                (w1_up, w1_dn), trial_data
            )
            r1 = 0.5 * jnp.real(ov1 / overlaps)
            r1 = jnp.where(r0 < w_floor, 0.0, r1)
            node_encounters += jnp.sum(r1 <= 0.0)

            # normalize probabilities
            norm = r0 + r1 + 1.0e-13
            p0 = r0 / norm

            # update walkers
            choose0 = uniform_rns_nn[:, 1, x] < p0 # (nw,)
            w_up = jnp.where(choose0.reshape(-1, 1, 1), w0_up, w1_up)
            w_dn = jnp.where(choose0.reshape(-1, 1, 1), w0_dn, w1_dn)
            overlaps = jnp.where(choose0, ov0, ov1)
            weights = weights * norm

            # ---------------------------------------------------------------------
            # dn up
            # field 0
            w0_dn = (
                w_dn
                .at[:, site_i, :]
                .mul(hs_nn[0, 0])
            )
            w0_up = (
                w_up
                .at[:, site_j, :]
                .mul(hs_nn[0, 1])
            )
            ov0 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
                (w0_up, w0_dn), trial_data
            )
            r0 = 0.5 * jnp.real(ov0 / overlaps)
            r0 = jnp.where(r0 < w_floor, 0.0, r0)
            node_encounters += jnp.sum(r0 <= 0.0)

            # field 1
            w1_dn = (
                w_dn
                .at[:, site_i, :]
                .mul(hs_nn[1, 0])
            )
            w1_up = (
                w_up
                .at[:, site_j, :]
                .mul(hs_nn[1, 1])
            )
            ov1 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
                (w1_up, w1_dn), trial_data
            )
            r1 = 0.5 * jnp.real(ov1 / overlaps)
            r1 = jnp.where(r0 < w_floor, 0.0, r1)
            node_encounters += jnp.sum(r1 <= 0.0)

            # normalize probabilities
            norm = r0 + r1 + 1.0e-13
            p0 = r0 / norm

            # update walkers
            choose0 = uniform_rns_nn[:, 2, x] < p0 # (nw,)
            w_up = jnp.where(choose0.reshape(-1, 1, 1), w0_up, w1_up)
            w_dn = jnp.where(choose0.reshape(-1, 1, 1), w0_dn, w1_dn)
            overlaps = jnp.where(choose0, ov0, ov1)
            weights = weights * norm

            # ---------------------------------------------------------------------
            # dn dn
            # field 0
            w0_dn = (
                w_dn
                .at[:, site_i, :]
                .mul(hs_nn[0, 0])
            )
            w0_dn = (
                w0_dn
                .at[:, site_j, :]
                .mul(hs_nn[0, 1])
            )
            ov0 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
                (w_up, w0_dn), trial_data
            )
            r0 = 0.5 * jnp.real(ov0 / overlaps)
            r0 = jnp.where(r0 < w_floor, 0.0, r0)
            node_encounters += jnp.sum(r0 <= 0.0)

            # field 1
            w1_dn = (
                w_dn
                .at[:, site_i, :]
                .mul(hs_nn[1, 0])
            )
            w1_dn = (
                w1_dn
                .at[:, site_j, :]
                .mul(hs_nn[1, 1])
            )
            ov1 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
                (w_up, w1_dn), trial_data
            )
            r1 = 0.5 * jnp.real(ov1 / overlaps)
            r1 = jnp.where(r0 < w_floor, 0.0, r1)
            node_encounters += jnp.sum(r1 <= 0.0)

            # normalize probabilities
            norm = r0 + r1 + 1.0e-13
            p0 = r0 / norm

            # update walkers
            choose0 = uniform_rns_nn[:, 3, x] < p0 # (nw,)
            w_dn = jnp.where(choose0.reshape(-1, 1, 1), w0_dn, w1_dn)
            overlaps = jnp.where(choose0, ov0, ov1)
            weights = weights * norm

            walkers = (w_up, w_dn)

        elif walker_kind == "generalized":
            # ---------------------------------------------------------------------
            # up up
            # field 0
            w0 = (
                walkers
                .at[:, site_i, :]
                .mul(hs_nn[0, 0])
            )
            w0 = (
                w0
                .at[:, site_j, :]
                .mul(hs_nn[0, 1])
            )
            ov0 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
                w0, trial_data
            )
            r0 = 0.5 * jnp.real(ov0 / overlaps)
            r0 = jnp.where(r0 < w_floor, 0.0, r0)
            node_encounters += jnp.sum(r0 <= 0.0)

            # field 1
            w1 = (
                walkers
                .at[:, site_i, :]
                .mul(hs_nn[1, 0])
            )
            w1 = (
                w1
                .at[:, site_j, :]
                .mul(hs_nn[1, 1])
            )
            ov1 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
                w1, trial_data
            )
            r1 = 0.5 * jnp.real(ov1 / overlaps)
            r1 = jnp.where(r0 < w_floor, 0.0, r1)
            node_encounters += jnp.sum(r1 <= 0.0)

            # normalize probabilities
            norm = r0 + r1 + 1.0e-13
            p0 = r0 / norm

            # update walkers
            choose0 = uniform_rns_nn[:, 0, x] < p0 # (nw,)
            walkers = jnp.where(choose0.reshape(-1, 1, 1), w0, w1)
            overlaps = jnp.where(choose0, ov0, ov1)
            weights = weights * norm

            # ---------------------------------------------------------------------
            # up dn
            # field 0
            w0 = (
                walkers
                .at[:, site_i, :]
                .mul(hs_nn[0, 0])
            )
            w0 = (
                w0
                .at[:, site_j+n_sites, :]
                .mul(hs_nn[0, 1])
            )
            ov0 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
                w0, trial_data
            )
            r0 = 0.5 * jnp.real(ov0 / overlaps)
            r0 = jnp.where(r0 < w_floor, 0.0, r0)
            node_encounters += jnp.sum(r0 <= 0.0)

            # field 1
            w1 = (
                walkers
                .at[:, site_i, :]
                .mul(hs_nn[1, 0])
            )
            w1 = (
                w1
                .at[:, site_j+n_sites, :]
                .mul(hs_nn[1, 1])
            )
            ov1 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
                w1, trial_data
            )
            r1 = 0.5 * jnp.real(ov1 / overlaps)
            r1 = jnp.where(r0 < w_floor, 0.0, r1)
            node_encounters += jnp.sum(r1 <= 0.0)

            # normalize probabilities
            norm = r0 + r1 + 1.0e-13
            p0 = r0 / norm

            # update walkers
            choose0 = uniform_rns_nn[:, 1, x] < p0 # (nw,)
            walkers = jnp.where(choose0.reshape(-1, 1, 1), w0, w1)
            overlaps = jnp.where(choose0, ov0, ov1)
            weights = weights * norm

            # ---------------------------------------------------------------------
            # dn up
            # field 0
            w0 = (
                walkers
                .at[:, site_i+n_sites, :]
                .mul(hs_nn[0, 0])
            )
            w0 = (
                w0
                .at[:, site_j, :]
                .mul(hs_nn[0, 1])
            )
            ov0 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
                w0, trial_data
            )
            r0 = 0.5 * jnp.real(ov0 / overlaps)
            r0 = jnp.where(r0 < w_floor, 0.0, r0)
            node_encounters += jnp.sum(r0 <= 0.0)

            # field 1
            w1 = (
                walkers
                .at[:, site_i+n_sites, :]
                .mul(hs_nn[1, 0])
            )
            w1 = (
                w1
                .at[:, site_j, :]
                .mul(hs_nn[1, 1])
            )
            ov1 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
                w1, trial_data
            )
            r1 = 0.5 * jnp.real(ov1 / overlaps)
            r1 = jnp.where(r0 < w_floor, 0.0, r1)
            node_encounters += jnp.sum(r1 <= 0.0)

            # normalize probabilities
            norm = r0 + r1 + 1.0e-13
            p0 = r0 / norm

            # update walkers
            choose0 = uniform_rns_nn[:, 2, x] < p0 # (nw,)
            walkers = jnp.where(choose0.reshape(-1, 1, 1), w0, w1)
            overlaps = jnp.where(choose0, ov0, ov1)
            weights = weights * norm

            # ---------------------------------------------------------------------
            # dn dn
            # field 0
            w0 = (
                walkers
                .at[:, site_i+n_sites, :]
                .mul(hs_nn[0, 0])
            )
            w0 = (
                w0
                .at[:, site_j+n_sites, :]
                .mul(hs_nn[0, 1])
            )
            ov0 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
                w0, trial_data
            )
            r0 = 0.5 * jnp.real(ov0 / overlaps)
            r0 = jnp.where(r0 < w_floor, 0.0, r0)
            node_encounters += jnp.sum(r0 <= 0.0)

            # field 1
            w1 = (
                walkers
                .at[:, site_i+n_sites, :]
                .mul(hs_nn[1, 0])
            )
            w1 = (
                w1
                .at[:, site_j+n_sites, :]
                .mul(hs_nn[1, 1])
            )
            ov1 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
                w1, trial_data
            )
            r1 = 0.5 * jnp.real(ov1 / overlaps)
            r1 = jnp.where(r0 < w_floor, 0.0, r1)
            node_encounters += jnp.sum(r1 <= 0.0)

            # normalize probabilities
            norm = r0 + r1 + 1.0e-13
            p0 = r0 / norm

            # update walkers
            choose0 = uniform_rns_nn[:, 3, x] < p0 # (nw,)
            walkers = jnp.where(choose0.reshape(-1, 1, 1), w0, w1)
            overlaps = jnp.where(choose0, ov0, ov1)
            weights = weights * norm

        return (walkers, overlaps, weights, node_encounters), x

    (walkers, overlaps, weights, node_encounters_step), _ = lax.scan(
        scanned_fun_nn,
        (walkers, overlaps, weights, node_encounters_step),
        jnp.arange(n_bonds),
    )

    # one body half step again
    walkers = wk.vmap_chunked(
        cpmc_ops.apply_one_body_half, params.n_chunks, in_axes=(0, None)
    )(walkers, prop_ctx)
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
