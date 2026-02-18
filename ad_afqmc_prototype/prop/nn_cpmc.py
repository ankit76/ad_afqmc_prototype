from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps, k_energy, require_cpmc_trial_ops
from ..core.system import System
from ..ham.hubbard_nn import HamHubbardNN
from ..sharding import shard_prop_state
from ..walkers import init_walkers
from .hubbard_nn_cpmc_ops import (
    HubbardNNCpmcCtx,
    HubbardNNCpmcOps,
    _build_prop_ctx,
    make_hubbard_nn_cpmc_ops,
)
from .types import PropOps, PropState, QmcParams


def init_prop_state(
    *,
    sys: System,
    ham_data: HamHubbardNN,
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
    Initialize NN-CPMC propagation state.
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
    overlaps = jnp.real(overlaps)

    e_est = None
    if initial_e_estimate is not None:
        e_est = jnp.asarray(initial_e_estimate)
    else:
        meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
        e_kernel = meas_ops.require_kernel(k_energy)
        e_samples = jnp.real(
            wk.vmap_chunked(
                e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None)
            )(initial_walkers, ham_data, meas_ctx, trial_data)
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
    cpmc_ops: HubbardNNCpmcOps,
    prop_ctx: HubbardNNCpmcCtx,
) -> PropState:
    """
    One CPMC step with discrete spin HS fields + fast updates.
    Requires:
      - trial_ops.calc_green
      - trial_ops.calc_overlap_ratio
      - trial_ops.update_green
    Walkers: unrestricted (w_up, w_dn), each (nw, n_sites, n_elec_spin).
    """
    def update_walkers(
        carry, 
        rns, 
        spins, 
        sites, 
        hs,
        w_floor,
        tol=1.0e-8
    ):
        walkers, overlaps, weights, greens, node = carry
        w_up, w_dn = walkers
        update_indices = jnp.column_stack((spins, sites))

        # field 0 ratio
        u0_0 = hs[0, 0] - 1.0
        u1_0 = hs[0, 1] - 1.0
        upd0 = jnp.array([u0_0, u1_0], dtype=jnp.result_type(u0_0, u1_0))
        r0 = wk.vmap_chunked(
            green_ops.calc_overlap_ratio,
            n_chunks=params.n_chunks,
            in_axes=(0, None, None),
        )(greens, update_indices, upd0)
        r0 = jnp.real(r0)
        r0 = jnp.where(r0 <= w_floor, 0.0, r0)
        node = node + jnp.sum(r0 <= 0.0)

        # field 1 ratio
        u0_1 = hs[1, 0] - 1.0
        u1_1 = hs[1, 1] - 1.0
        upd1 = jnp.array([u0_1, u1_1], dtype=jnp.result_type(u0_1, u1_1))
        r1 = wk.vmap_chunked(
            green_ops.calc_overlap_ratio,
            n_chunks=params.n_chunks,
            in_axes=(0, None, None),
        )(greens, update_indices, upd1)
        r1 = jnp.real(r1)
        r1 = jnp.where(r1 <= w_floor, 0.0, r1)
        node = node + jnp.sum(r1 <= 0.0)

        # probabilities
        p0 = 0.5 * r0
        p1 = 0.5 * r1
        norm = p0 + p1 + 1.0e-13
        p0 = p0 / norm

        choose0 = rns < p0  # (nw,)

        # apply chosen HS constants to walker
        constants = jnp.where( # (nw, 2) for lambda_{\pm}
            choose0.reshape(-1, 1),
            hs[0], # field 0
            hs[1], # field 1
        )

        spin_i, spin_j = spins
        site_i, site_j = sites
        
        if spin_i == 0:
            w_up = (
                w_up
                .at[:, site_i, :]
                .mul(constants[:, 0].reshape(-1, 1)) # lambda_{+}
            )

        else:
            w_dn = (
                w_dn
                .at[:, site_i, :]
                .mul(constants[:, 0].reshape(-1, 1)) # lambda_{+}
            )

        if spin_j == 0:
            w_up = (
                w_up
                .at[:, site_j, :]
                .mul(constants[:, 1].reshape(-1, 1)) # lambda_{-}
            )

        else:
            w_dn = (
                w_dn
                .at[:, site_j, :]
                .mul(constants[:, 1].reshape(-1, 1)) # lambda_{-}
            )

        walkers = (w_up, w_dn)

        # update overlap and weights
        r_sel = jnp.where(choose0, r0, r1)
        overlaps = overlaps * r_sel
        weights = weights * norm

        # fast greens update
        greens = wk.vmap_chunked(
            green_ops.update_green,
            n_chunks=params.n_chunks,
            in_axes=(0, None, 0)
        )(greens, update_indices, constants - 1)

        return (walkers, overlaps, weights, greens, node)
        
    green_ops = require_cpmc_trial_ops(trial_ops)
    walkers = state.walkers
    nw = wk.n_walkers(walkers)

    w_floor = float(getattr(params, "weight_floor", 1.0e-8))
    w_cap = float(getattr(params, "weight_cap", 100.0))
    damping = float(getattr(params, "pop_control_damping", 0.1))

    node_step = jnp.asarray(0)

    # one body half step
    walkers = wk.vmap_chunked(
        cpmc_ops.apply_one_body_half, params.n_chunks, in_axes=(0, None)
    )(walkers, prop_ctx)

    overlaps_1 = wk.vmap_chunked(
        meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers, trial_data)
    overlaps_1 = jnp.real(overlaps_1)

    # constraint
    ratio_1 = jnp.real(overlaps_1 / state.overlaps)
    ratio_1 = jnp.where(ratio_1 <= w_floor, 0.0, ratio_1)
    node_step = node_step + jnp.sum(ratio_1 <= 0.0)

    weights = state.weights * ratio_1
    weights = jnp.where(weights > w_cap, 0.0, weights)

    # compute greens
    greens = wk.vmap_chunked(
        green_ops.calc_green, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers, trial_data)

    overlaps = overlaps_1

    # two body 
    # onsite interactions
    n_sites = cpmc_ops.n_sites()
    hs_onsite = prop_ctx.hs_constant_onsite  # (2,2)
    key, subkey = jax.random.split(state.rng_key)
    uniform_rns = jax.random.uniform(subkey, (nw, n_sites))
    
    # scan over sites
    def scanned_fun(carry, x):
        carry = update_walkers(
            carry, 
            uniform_rns[:, x], 
            (0, 1),
            (x, x),
            hs_onsite,
            w_floor,
        )
        return carry, x

    (walkers, overlaps, weights, greens, node_step), _ = lax.scan(
        scanned_fun,
        (walkers, overlaps, weights, greens, node_step),
        jnp.arange(n_sites, dtype=jnp.int32),
    )

    # nearest-neighbor interactions
    bonds = cpmc_ops.bonds()
    n_bonds = cpmc_ops.n_bonds()
    hs_nn = prop_ctx.hs_constant_nn  # (2,2)
    key, subkey = jax.random.split(state.rng_key)
    uniform_rns = jax.random.uniform(subkey, (nw, 4, n_bonds))

    # scan over nearest-neighbour bonds
    def scanned_fun_nn(carry, x):
        site_i = bonds[x, 0]
        site_j = bonds[x, 1]

        # up, up
        carry = update_walkers(
            carry,
            uniform_rns[:, 0, x],
            (0, 0),
            (site_i, site_j),
            hs_nn,
            w_floor,
        )

        # up, dn
        carry = update_walkers(
            carry,
            uniform_rns[:, 1, x],
            (0, 1),
            (site_i, site_j),
            hs_nn,
            w_floor,
        )

        # dn, up
        carry = update_walkers(
            carry,
            uniform_rns[:, 2, x],
            (1, 0),
            (site_i, site_j),
            hs_nn,
            w_floor,
        )

        # dn, dn
        carry = update_walkers(
            carry,
            uniform_rns[:, 3, x],
            (1, 1),
            (site_i, site_j),
            hs_nn,
            w_floor,
        )

        return carry, x

    (walkers, overlaps, weights, greens, node_step), _ = lax.scan(
        scanned_fun_nn,
        (walkers, overlaps, weights, greens, node_step),
        jnp.arange(n_bonds, dtype=jnp.int32),
    )

    # one body half step (2)
    walkers = wk.vmap_chunked(
        cpmc_ops.apply_one_body_half, params.n_chunks, in_axes=(0, None)
    )(walkers, prop_ctx)

    overlaps_2 = wk.vmap_chunked(meas_ops.overlap, params.n_chunks, in_axes=(0, None))(
        walkers, trial_data
    )
    overlaps_2 = jnp.real(overlaps_2)

    ratio_2 = jnp.real(overlaps_2 / overlaps)
    ratio_2 = jnp.where(ratio_2 <= w_floor, 0.0, ratio_2)
    node_step = node_step + jnp.sum(ratio_2 <= 0.0)

    weights = weights * ratio_2
    weights = jnp.where(weights > w_cap, 0.0, weights)

    overlaps_new = overlaps_2

    # population control
    weights = weights * jnp.exp(prop_ctx.dt * state.pop_control_ene_shift)
    weights = jnp.where(weights > w_cap, 0.0, weights)

    avg_w = jnp.clip(jnp.mean(weights), min=1.0e-300)
    pop_shift_new = state.e_estimate - damping * (jnp.log(avg_w) / prop_ctx.dt)

    node_new = state.node_encounters + node_step

    return PropState(
        walkers=walkers,
        weights=weights,
        overlaps=overlaps_new,
        rng_key=key,
        pop_control_ene_shift=pop_shift_new,
        e_estimate=state.e_estimate,
        node_encounters=node_new,
    )


def make_prop_ops(
    ham_data: HamHubbardNN,
    walker_kind: str,
    trial_ops: TrialOps,
) -> PropOps:
    """
    Build PropOps for NN-CPMC with fast updates.
    """
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
            trial_ops=trial_ops,
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
