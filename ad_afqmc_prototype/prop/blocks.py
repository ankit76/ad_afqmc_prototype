from __future__ import annotations

from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax import lax

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps, k_energy
from ..core.system import System
from .types import PropOps, PropState, QmcParams


class BlockFn(Protocol):
    def __call__(
        self,
        state: PropState,
        *,
        sys: System,
        params: QmcParams,
        ham_data: Any,
        trial_data: Any,
        trial_ops: TrialOps,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ops: PropOps,
        prop_ctx: Any,
    ) -> tuple[PropState, BlockObs]: ...


class BlockObs(NamedTuple):
    scalars: dict[str, jax.Array]


def block(
    state: PropState,
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    trial_data: Any,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    meas_ctx: Any,
    prop_ops: PropOps,
    prop_ctx: Any,
) -> tuple[PropState, BlockObs]:
    """
    propagation + measurement
    """
    step = lambda st: prop_ops.step(
        st,
        params=params,
        ham_data=ham_data,
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ctx=prop_ctx,
        meas_ctx=meas_ctx,
    )

    def _scan_step(carry: PropState, _x: Any):
        carry = step(carry)
        return carry, None

    state, _ = lax.scan(_scan_step, state, xs=None, length=params.n_prop_steps)

    walkers_new = wk.orthonormalize(state.walkers, sys.walker_kind)
    overlaps_new = wk.vmap_chunked(
        meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers_new, trial_data)
    state = state._replace(walkers=walkers_new, overlaps=overlaps_new)

    e_kernel = meas_ops.require_kernel(k_energy)
    e_samples = wk.vmap_chunked(
        e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None)
    )(state.walkers, ham_data, meas_ctx, trial_data)
    e_samples = jnp.real(e_samples)

    thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
    e_ref = state.e_estimate
    e_samples = jnp.where(jnp.abs(e_samples - e_ref) > thresh, e_ref, e_samples)

    weights = state.weights
    w_sum = jnp.sum(weights)
    w_sum_safe = jnp.where(w_sum == 0, 1.0, w_sum)
    e_block = jnp.sum(weights * e_samples) / w_sum_safe
    e_block = jnp.where(w_sum == 0, e_ref, e_block)

    # nchol
    nchol = ham_data.chol.shape[0] // 5
    ene_1 = wk.vmap_chunked(
        e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None, None)
    )(state.walkers, ham_data, meas_ctx, trial_data, nchol)
    ene_1_block = jnp.real(jnp.sum(state.weights * (e_samples - ene_1)) / w_sum_safe)
    nchol = 2 * ham_data.chol.shape[0] // 5
    ene_2 = wk.vmap_chunked(
        e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None, None)
    )(state.walkers, ham_data, meas_ctx, trial_data, nchol)
    ene_2_block = jnp.real(jnp.sum(state.weights * (e_samples - ene_2)) / w_sum_safe)
    nchol = 3 * ham_data.chol.shape[0] // 5
    ene_3 = wk.vmap_chunked(
        e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None, None)
    )(state.walkers, ham_data, meas_ctx, trial_data, nchol)
    ene_3_block = jnp.real(jnp.sum(state.weights * (e_samples - ene_3)) / w_sum_safe)
    nchol = 4 * ham_data.chol.shape[0] // 5
    ene_4 = wk.vmap_chunked(
        e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None, None)
    )(state.walkers, ham_data, meas_ctx, trial_data, nchol)
    ene_4_block = jnp.real(jnp.sum(state.weights * (e_samples - ene_4)) / w_sum_safe)
    ene_chol = jnp.array([ene_1_block, ene_2_block, ene_3_block, ene_4_block])

    alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(e_block))
    state = state._replace(
        e_estimate=(1.0 - alpha) * state.e_estimate + alpha * e_block
    )

    key, subkey = jax.random.split(state.rng_key)
    zeta = jax.random.uniform(subkey)
    w_sr, weights_sr = wk.stochastic_reconfiguration(
        state.walkers, state.weights, zeta, sys.walker_kind
    )
    overlaps_sr = wk.vmap_chunked(
        meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(w_sr, trial_data)
    state = state._replace(
        walkers=w_sr,
        weights=weights_sr,
        overlaps=overlaps_sr,
        rng_key=key,
    )

    obs = BlockObs(scalars={"energy": e_block, "weight": w_sum, "ene_chol": ene_chol})
    return state, obs


def block_1(
    state: PropState,
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    trial_data: Any,
    trial_data_e: Any,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    meas_ctx: Any,
    meas_ops_e: MeasOps,
    meas_ctx_e: Any,
    prop_ops: PropOps,
    prop_ctx: Any,
) -> tuple[PropState, BlockObs]:
    """
    propagation + measurement
    """
    step = lambda st: prop_ops.step(
        st,
        params=params,
        ham_data=ham_data,
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ctx=prop_ctx,
        meas_ctx=meas_ctx,
    )

    def _scan_step(carry: PropState, _x: Any):
        carry = step(carry)
        return carry, None

    state, _ = lax.scan(_scan_step, state, xs=None, length=params.n_prop_steps)

    walkers_new = wk.orthonormalize(state.walkers, sys.walker_kind)
    overlaps_new = wk.vmap_chunked(
        meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers_new, trial_data)
    state = state._replace(walkers=walkers_new, overlaps=overlaps_new)

    e_kernel = meas_ops_e.require_kernel(k_energy)
    e_samples = wk.vmap_chunked(
        e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None)
    )(state.walkers, ham_data, meas_ctx_e, trial_data_e)
    e_samples = jnp.real(e_samples)

    thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
    e_ref = state.e_estimate
    e_samples = jnp.where(jnp.abs(e_samples - e_ref) > thresh, e_ref, e_samples)
    overlap_e = wk.vmap_chunked(
        meas_ops_e.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(state.walkers, trial_data_e)

    weights = state.weights
    w_sum = jnp.sum(weights * overlap_e / state.overlaps)
    w_sum_safe = w_sum
    # w_sum_safe = jnp.where(w_sum == 0, 1.0, w_sum)
    e_block = jnp.sum(weights * e_samples * overlap_e / state.overlaps) / w_sum_safe
    e_block_r = jnp.real(e_block)
    # e_block = jnp.where(w_sum == 0, e_ref, e_block)

    alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(e_block_r))
    state = state._replace(
        e_estimate=(1.0 - alpha) * state.e_estimate + alpha * e_block_r
    )

    key, subkey = jax.random.split(state.rng_key)
    zeta = jax.random.uniform(subkey)
    w_sr, weights_sr = wk.stochastic_reconfiguration(
        state.walkers, state.weights, zeta, sys.walker_kind
    )
    overlaps_sr = wk.vmap_chunked(
        meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(w_sr, trial_data)
    state = state._replace(
        walkers=w_sr,
        weights=weights_sr,
        overlaps=overlaps_sr,
        rng_key=key,
    )

    obs = BlockObs(scalars={"energy": e_block.real, "weight": w_sum.real})
    return state, obs
