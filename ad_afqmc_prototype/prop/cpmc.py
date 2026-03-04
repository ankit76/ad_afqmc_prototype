from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps, k_energy, require_cpmc_trial_ops
from ..core.system import System
from ..ham.hubbard import HamHubbard
from ..sharding import shard_prop_state
from ..walkers import init_walkers
from .hubbard_cpmc_ops import (
    HubbardCpmcCtx,
    HubbardCpmcOps,
    _build_prop_ctx,
    make_hubbard_cpmc_ops,
)
from .types import PropOps, PropState, QmcParams


def init_prop_state(
    *,
    sys: System,
    ham_data: HamHubbard,
    trial_ops: TrialOps,
    trial_data: Any,
    meas_ops: MeasOps,
    params: QmcParams,
    initial_walkers: Any | None = None,
    initial_e_estimate: jax.Array | None = None,
    rdm1: jax.Array | None = None,
    mesh: Mesh | None = None,
) -> PropState:
    """
    Initialize CPMC propagation state.
    """
    n_walkers = params.n_walkers
    seed = params.seed
    key = jax.random.PRNGKey(int(seed))
    weights = jnp.ones((n_walkers,))

    if initial_walkers is None:
        if rdm1 is None:
            rdm1 = trial_ops.get_rdm1(trial_data)
        initial_walkers = init_walkers(sys=sys, rdm1=rdm1, n_walkers=n_walkers)

    initial_walkers = jax.tree_util.tree_map(lambda x: jnp.real(x), initial_walkers)

    overlaps = wk.vmap_chunked(
        trial_ops.overlap,
        n_chunks=params.n_chunks,
        in_axes=(0, None),
    )(initial_walkers, trial_data)

    e_est = None
    if initial_e_estimate is not None:
        e_est = jnp.asarray(initial_e_estimate)
    else:
        meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
        e_kernel = meas_ops.require_kernel(k_energy)
        e_samples = jnp.real(
            wk.vmap_chunked(e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None))(
                initial_walkers, ham_data, meas_ctx, trial_data
            )
        )
        e_est = jnp.mean(e_samples)
    pop_shift = e_est

    node_encounters = jnp.asarray(0)

    state = PropState(
        walkers=initial_walkers,
        weights=weights,
        overlaps=overlaps,
        rng_key=key,
        pop_control_ene_shift=pop_shift,
        e_estimate=e_est,
        node_encounters=node_encounters,
    )
    return shard_prop_state(state, mesh)


def cpmc_step(
    state: PropState,
    *,
    params: QmcParams,
    trial_ops: TrialOps,
    trial_data: Any,
    meas_ops: MeasOps,
    cpmc_ops: HubbardCpmcOps,
    prop_ctx: HubbardCpmcCtx,
) -> PropState:
    """
    One CPMC step with discrete spin HS fields + fast updates.
    Requires:
      - trial_ops.calc_green
      - trial_ops.calc_overlap_ratio
      - trial_ops.update_green
    Walkers: 
        - unrestricted (w_up, w_dn), each (nw, n_sites, n_elec_spin).
        - generalized, (nw, 2*n_sites, n_elec).
    """
    green_ops = require_cpmc_trial_ops(trial_ops)
    walkers = state.walkers
    nw = wk.n_walkers(walkers)
    walker_kind = prop_ctx.walker_kind
    node_encounters_step = jnp.asarray(0)

    w_floor = float(getattr(params, "weight_floor", 1.0e-8))
    w_cap = float(getattr(params, "weight_cap", 100.0))
    damping = float(getattr(params, "pop_control_damping", 0.1))


    # one body half step (1)
    walkers = wk.vmap_chunked(
        cpmc_ops.apply_one_body_half, params.n_chunks, in_axes=(0, None)
    )(walkers, prop_ctx)
    overlaps = wk.vmap_chunked(
        meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers, trial_data)
    ratio = jnp.real(overlaps / state.overlaps)
    ratio = jnp.where(ratio <= w_floor, 0.0, ratio) # constraint
    node_encounters_step = node_encounters_step + jnp.sum(ratio <= 0.0)
    weights = state.weights * ratio
    weights = jnp.where(weights > w_cap, 0.0, weights)

    # compute greens
    greens = wk.vmap_chunked(
        green_ops.calc_green, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers, trial_data)
    
    # two body: scan over sites
    n_sites = cpmc_ops.n_sites()
    hs = prop_ctx.hs_constant  # (2,2)
    key, subkey = jax.random.split(state.rng_key)
    uniform_rns = jax.random.uniform(subkey, (nw, n_sites)) # uniform HS fields

    def scanned_fun(carry, x):
        walkers, overlaps, weights, greens, node_encounters = carry
        upd_indices = jnp.array([[0, x], [1, x]], dtype=jnp.int32) # (2,2)

        # field 0 ratio
        upd0 = hs[0] - 1.0
        r0 = wk.vmap_chunked(
            green_ops.calc_overlap_ratio,
            n_chunks=params.n_chunks,
            in_axes=(0, None, None),
        )(greens, upd_indices, upd0)
        r0 = jnp.where(r0 <= w_floor, 0.0, r0) # constraint
        node_encounters = node_encounters + jnp.sum(r0 <= 0.0)

        # field 1 ratio
        upd1 = hs[1] - 1.0
        r1 = wk.vmap_chunked(
            green_ops.calc_overlap_ratio,
            n_chunks=params.n_chunks,
            in_axes=(0, None, None),
        )(greens, upd_indices, upd1)
        r1 = jnp.where(r1 <= w_floor, 0.0, r1) # constraint
        node_encounters = node_encounters + jnp.sum(r1 <= 0.0)
        
        # probabilities
        p0 = 0.5 * r0.real
        p1 = 0.5 * r1.real
        norm = p0 + p1 + 1.0e-13
        p0 = p0 / norm

        # random choice
        choose0 = uniform_rns[:, x] < p0  # (nw,)

        # apply chosen HS constants to walker row x
        c_up = jnp.where(choose0, hs[0, 0], hs[1, 0]) 
        c_dn = jnp.where(choose0, hs[0, 1], hs[1, 1])

        if walker_kind == "unrestricted":
            w_up = walkers[0].at[:, x, :].mul(c_up.reshape(-1, 1))
            w_dn = walkers[1].at[:, x, :].mul(c_dn.reshape(-1, 1))
            walkers = (w_up, w_dn)

        elif walker_kind == "generalized":
            walkers = walkers.at[:, x, :].mul(c_up.reshape(-1, 1))
            walkers = walkers.at[:, x+n_sites, :].mul(c_dn.reshape(-1, 1))

        # update overlap and weights
        r = jnp.where(choose0, r0, r1)
        overlaps = overlaps * r
        weights = weights * norm
        
        # fast greens update
        upd_constants = jnp.stack([c_up - 1.0, c_dn - 1.0], axis=1)  # (nw, 2)
        greens = wk.vmap_chunked(
            green_ops.update_green, n_chunks=params.n_chunks, in_axes=(0, None, 0)
        )(greens, upd_indices, upd_constants)

        return (walkers, overlaps, weights, greens, node_encounters), None

    (walkers, overlaps, weights, greens, node_encounters_step), _ = lax.scan(
        scanned_fun,
        (walkers, overlaps, weights, greens, node_encounters_step),
        jnp.arange(n_sites, dtype=jnp.int32),
    )

    # one body half step (2)
    walkers = wk.vmap_chunked(
        cpmc_ops.apply_one_body_half, params.n_chunks, in_axes=(0, None)
    )(walkers, prop_ctx)
    overlaps_new = wk.vmap_chunked(
        meas_ops.overlap, params.n_chunks, in_axes=(0, None)
    )(walkers, trial_data)
    ratio = jnp.real(overlaps_new / overlaps)
    ratio = jnp.where(ratio <= w_floor, 0.0, ratio) # constraint
    node_encounters_step = node_encounters_step + jnp.sum(ratio <= 0.0)
    weights = weights * ratio
    weights = jnp.where(weights > w_cap, 0.0, weights)

    # population control
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


def make_prop_ops(
    ham_data: HamHubbard,
    walker_kind: str,
    trial_ops: TrialOps,
) -> PropOps:
    """
    Build PropOps for CPMC with fast updates.
    """
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
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            cpmc_ops=cpmc_ops,
            prop_ctx=prop_ctx,
        )

    def build_prop_ctx(
        ham_data: HamHubbard,
        trial_data: Any,
        params: QmcParams,
    ) -> HubbardCpmcCtx:
        return _build_prop_ctx(ham_data, params.dt, walker_kind)

    return PropOps(
        init_prop_state=init_prop_state,
        build_prop_ctx=build_prop_ctx,
        step=step,
    )
