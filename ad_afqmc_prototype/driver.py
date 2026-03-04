from __future__ import annotations

import time
from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .core.ops import MeasOps, TrialOps
from .core.system import System
from .prop.blocks import BlockFn
from .prop.types import PropOps, PropState, QmcParams
from .stat_utils import blocking_analysis_ratio, jackknife_ratios, rebin_observable, reject_outliers
from .walkers import stochastic_reconfiguration

print = partial(print, flush=True)


class QmcResult(NamedTuple):
    mean_energy: jax.Array
    stderr_energy: jax.Array
    block_energies: jax.Array
    block_weights: jax.Array
    block_observables: dict[str, jax.Array]
    observable_means: dict[str, jax.Array]
    observable_stderrs: dict[str, jax.Array]


def _weighted_block_mean(values: jax.Array, weights: jax.Array) -> jax.Array:
    w_sum = jnp.sum(weights)
    w_shape = (weights.shape[0],) + (1,) * max(values.ndim - 1, 0)
    num = jnp.sum(weights.reshape(w_shape) * values, axis=0)
    zero = jnp.zeros_like(num)
    return jnp.where(w_sum == 0, zero, num / w_sum)


def make_run_blocks(
    *,
    block_fn: BlockFn,
    sys: System,
    params: QmcParams,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    prop_ops: PropOps,
    observable_names: tuple[str, ...] = (),
) -> Callable:
    """
    Build a jitted run_blocks.
    We keep ham_data, trial_data, meas_ctx, prop_ctx as arguments to
    improve compilation, as these objects can be large.
    """

    @partial(jax.jit, static_argnames=("n_blocks",))
    def run_blocks(
        state0,
        *,
        ham_data,
        trial_data,
        meas_ctx,
        prop_ctx,
        n_blocks: int,
    ):
        def one_block(state, _):
            state, obs = block_fn(
                state,
                sys=sys,
                params=params,
                ham_data=ham_data,
                trial_data=trial_data,
                trial_ops=trial_ops,
                meas_ops=meas_ops,
                meas_ctx=meas_ctx,
                prop_ops=prop_ops,
                prop_ctx=prop_ctx,
                observable_names=observable_names,
            )
            obs_tuple = tuple(obs.observables[name] for name in observable_names)
            return state, (obs.scalars["energy"], obs.scalars["weight"], obs_tuple)

        stateN, (e, w, obs) = lax.scan(one_block, state0, xs=None, length=n_blocks)
        return stateN, e, w, obs

    return run_blocks


def run_qmc(
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    trial_data: Any,
    meas_ops: MeasOps,
    trial_ops: TrialOps,
    prop_ops: PropOps,
    block_fn: BlockFn,
    state: PropState | None = None,
    meas_ctx: Any | None = None,
    target_error: float | None = None,
    mesh: Mesh | None = None,
    observable_names: tuple[str, ...] = (),
) -> QmcResult:
    """
    equilibration blocks then sampling blocks.

    Returns:
      QmcResult with energy statistics plus block-level observable estimates.
    """
    for name in observable_names:
        meas_ops.require_observable(name)

    # build ctx
    prop_ctx = prop_ops.build_prop_ctx(ham_data, trial_ops.get_rdm1(trial_data), params)
    if meas_ctx is None:
        meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
    if state is None:
        state = prop_ops.init_prop_state(
            sys=sys,
            ham_data=ham_data,
            trial_ops=trial_ops,
            trial_data=trial_data,
            meas_ops=meas_ops,
            params=params,
            mesh=mesh,
        )

    if mesh is None or mesh.size == 1:
        block_fn_sr = block_fn
    else:
        data_sh = NamedSharding(mesh, P("data"))
        sr_sharded = partial(stochastic_reconfiguration, data_sharding=data_sh)
        block_fn_sr = partial(block_fn, sr_fn=sr_sharded)

    run_blocks = make_run_blocks(
        block_fn=block_fn_sr,
        sys=sys,
        params=params,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
        observable_names=observable_names,
    )

    t0 = time.perf_counter()
    t_mark = t0

    print_every = params.n_eql_blocks // 5 if params.n_eql_blocks >= 5 else 0
    block_e_eq = []
    block_w_eq = []
    block_obs_eq = {name: [] for name in observable_names}
    block_e_eq.append(state.e_estimate)
    block_w_eq.append(jnp.sum(state.weights))
    print("\nEquilibration:\n")
    if print_every:
        print(
            f"{'':4s}"
            f"{'block':>9s}  "
            f"{'E_blk':>14s}  "
            f"{'W':>12s}   "
            f"{'nodes':>10s}  "
            f"{'t[s]':>8s}"
        )
    print(
        f"[eql {0:4d}/{params.n_eql_blocks}]  "
        f"{float(state.e_estimate):14.10f}  "
        f"{float(jnp.sum(state.weights)):12.6e}  "
        f"{int(state.node_encounters):10d}  "
        f"{0.0:8.1f}"
    )
    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_eql_blocks, chunk):
        n = min(chunk, params.n_eql_blocks - start)
        state, e_chunk, w_chunk, obs_chunk = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=n,
        )
        block_e_eq.extend(e_chunk.tolist())
        block_w_eq.extend(w_chunk.tolist())
        for i, name in enumerate(observable_names):
            block_obs_eq[name].append(obs_chunk[i])
        w_chunk_avg = jnp.mean(w_chunk)
        e_chunk_avg = jnp.mean(e_chunk * w_chunk) / w_chunk_avg
        elapsed = time.perf_counter() - t0
        print(
            f"[eql {start + n:4d}/{params.n_eql_blocks}]  "
            f"{float(e_chunk_avg):14.10f}  "
            f"{float(w_chunk_avg):12.6e}  "
            f"{int(state.node_encounters):10d}  "
            f"{elapsed:8.1f}"
        )
    block_e_eq = jnp.asarray(block_e_eq)
    block_w_eq = jnp.asarray(block_w_eq)
    block_obs_eq = {
        name: (jnp.concatenate(block_obs_eq[name], axis=0) if len(block_obs_eq[name]) > 0 else None)
        for name in observable_names
    }

    # sampling
    print("\nSampling:\n")
    if target_error is None:
        target_error = 0.0
    print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 0
    block_e_s = []
    block_w_s = []
    block_obs_s = {name: [] for name in observable_names}
    if print_every:
        print(
            f"{'':4s}{'block':>9s}  {'E_avg':>14s}  {'E_err':>10s}  {'E_block':>14s}  "
            f"{'W':>12s}    {'nodes':>10s}  {'dt[s/bl]':>10s}  {'t[s]':>7s}"
        )

    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_blocks, chunk):
        n = min(chunk, params.n_blocks - start)
        state, e_chunk, w_chunk, obs_chunk = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=n,
        )
        block_e_s.extend(e_chunk.tolist())
        block_w_s.extend(w_chunk.tolist())
        for i, name in enumerate(observable_names):
            block_obs_s[name].append(obs_chunk[i])
        w_chunk_avg = jnp.mean(w_chunk)
        e_chunk_avg = jnp.mean(e_chunk * w_chunk) / w_chunk_avg
        elapsed = time.perf_counter() - t0
        dt_per_block = (time.perf_counter() - t_mark) / float(n)
        t_mark = time.perf_counter()
        stats = blocking_analysis_ratio(
            jnp.asarray(block_e_s), jnp.asarray(block_w_s), print_q=False
        )
        mu = stats["mu"]
        se = stats["se_star"]
        nodes = int(state.node_encounters)
        print(
            f"[blk {start + n:4d}/{params.n_blocks}]  "
            f"{mu:14.10f}  "
            f"{(f'{se:10.3e}' if se is not None else ' ' * 10)}  "
            f"{float(e_chunk_avg):16.10f}  "
            f"{float(w_chunk_avg):12.6e}  "
            f"{nodes:10d}  "
            f"{dt_per_block:9.3f}  "
            f"{elapsed:8.1f}"
        )
        if se is not None and se <= target_error and target_error > 0.0:
            print(f"\nTarget error {target_error:.3e} reached at block {start + n}.")
            break
    block_e_s = jnp.asarray(block_e_s)
    block_w_s = jnp.asarray(block_w_s)
    block_obs_s = {
        name: (jnp.concatenate(block_obs_s[name], axis=0) if len(block_obs_s[name]) > 0 else None)
        for name in observable_names
    }

    data_clean, keep_mask = reject_outliers(jnp.column_stack((block_e_s, block_w_s)), obs=0)
    print(f"\nRejected {block_e_s.shape[0] - data_clean.shape[0]} outlier blocks.")
    block_e_s = jnp.asarray(data_clean[:, 0])
    block_w_s = jnp.asarray(data_clean[:, 1])
    keep_mask = jnp.asarray(keep_mask)
    block_obs_s = {
        name: (arr[keep_mask] if arr is not None else None) for name, arr in block_obs_s.items()
    }
    print("\nFinal blocking analysis:")
    stats = blocking_analysis_ratio(block_e_s, block_w_s, print_q=True)
    mean, err = stats["mu"], stats["se_star"]

    block_e_all = jnp.concatenate([block_e_eq, block_e_s])
    block_w_all = jnp.concatenate([block_w_eq, block_w_s])
    block_obs_all: dict[str, jax.Array] = {}
    for name in observable_names:
        arr_eq = block_obs_eq[name]
        arr_s = block_obs_s[name]
        if arr_eq is None and arr_s is None:
            block_obs_all[name] = jnp.zeros((0,))
            continue
        if arr_eq is None:
            arr_eq = arr_s[:0]
        if arr_s is None:
            arr_s = arr_eq[:0]
        block_obs_all[name] = jnp.concatenate([arr_eq, arr_s], axis=0)

    obs_means: dict[str, jax.Array] = {}
    obs_stderrs: dict[str, jax.Array] = {}
    b_star = stats.get("B_star")
    for name in observable_names:
        arr = block_obs_s[name]
        if arr is None:
            obs_means[name] = jnp.zeros((0,))
            obs_stderrs[name] = jnp.zeros((0,))
            continue
        obs_means[name] = _weighted_block_mean(arr, block_w_s)
        if b_star is not None and b_star >= 1:
            import numpy as np

            num, denom = rebin_observable(np.asarray(arr), np.asarray(block_w_s), b_star)
            if num.shape[0] >= 2:
                _, se = jackknife_ratios(num, denom)
                obs_stderrs[name] = jnp.asarray(se)
            else:
                obs_stderrs[name] = jnp.full(arr.shape[1:], jnp.nan)
        else:
            obs_stderrs[name] = jnp.full(arr.shape[1:], jnp.nan)

    return QmcResult(
        mean_energy=mean,
        stderr_energy=err,
        block_energies=block_e_all,
        block_weights=block_w_all,
        block_observables=block_obs_all,
        observable_means=obs_means,
        observable_stderrs=obs_stderrs,
    )


def run_qmc_energy(
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    trial_data: Any,
    meas_ops: MeasOps,
    trial_ops: TrialOps,
    prop_ops: PropOps,
    block_fn: BlockFn,
    state: PropState | None = None,
    meas_ctx: Any | None = None,
    target_error: float | None = None,
    mesh: Mesh | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    out = run_qmc(
        sys=sys,
        params=params,
        ham_data=ham_data,
        trial_data=trial_data,
        meas_ops=meas_ops,
        trial_ops=trial_ops,
        prop_ops=prop_ops,
        block_fn=block_fn,
        state=state,
        meas_ctx=meas_ctx,
        target_error=target_error,
        mesh=mesh,
        observable_names=(),
    )
    return out.mean_energy, out.stderr_energy, out.block_energies, out.block_weights
