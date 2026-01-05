from __future__ import annotations

import time
from functools import partial
from pprint import pprint
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax

from .core.ops import MeasOps, TrialOps
from .core.system import System
from .prop.blocks import BlockFn, BlockObs
from .prop.types import PropOps, PropState, QmcParams
from .stat_utils import blocking_analysis_ratio, reject_outliers

print = partial(print, flush=True)


def make_run_blocks(
    *,
    block_fn: BlockFn,
    sys: System,
    params: QmcParams,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    prop_ops: PropOps,
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
            )
            return state, (
                obs.scalars["energy"],
                obs.scalars["weight"],
                obs.scalars["ene_chol"],
            )

        stateN, (e, w, ene_chol) = lax.scan(one_block, state0, xs=None, length=n_blocks)
        return stateN, e, w, ene_chol

    return run_blocks


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
    rdm1: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    equilibration blocks then sampling blocks.

    Returns:
      (mean_energy, stderr, block_energies, block_weights)
    """
    print("Starting QMC driver...")
    print(f"Parameters:")
    pprint(params)
    print("")
    # build ctx
    if rdm1 is None:
        rdm1 = trial_ops.get_rdm1(trial_data)
    prop_ctx = prop_ops.build_prop_ctx(ham_data, rdm1, params)
    meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
    if state is None:
        state = prop_ops.init_prop_state(
            sys=sys,
            ham_data=ham_data,
            trial_ops=trial_ops,
            trial_data=trial_data,
            meas_ops=meas_ops,
            params=params,
            rdm1=rdm1,
        )

    run_blocks = make_run_blocks(
        block_fn=block_fn,
        sys=sys,
        params=params,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
    )

    t0 = time.perf_counter()
    t_mark = t0

    print_every = params.n_eql_blocks // 5 if params.n_eql_blocks >= 5 else 0
    block_e_eq = []
    block_w_eq = []
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
    chunk = print_every
    for start in range(0, params.n_eql_blocks, chunk):
        n = min(chunk, params.n_eql_blocks - start)
        state, e_chunk, w_chunk, _ = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=n,
        )
        block_e_eq.extend(e_chunk.tolist())
        block_w_eq.extend(w_chunk.tolist())
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

    # sampling
    print("\nSampling:\n")
    print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 0
    block_e_s = []
    block_w_s = []
    block_echol_s = []
    if print_every:
        print(
            f"{'':4s}{'block':>9s}  {'E_avg':>14s}  {'E_err':>10s}  {'E_block':>14s}  "
            f"{'W':>12s}    {'nodes':>10s}  {'dt[s/bl]':>10s}  {'t[s]':>7s}"
        )

    chunk = print_every
    for start in range(0, params.n_blocks, chunk):
        n = min(chunk, params.n_blocks - start)
        state, e_chunk, w_chunk, ene_chol_chunk = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=n,
        )
        block_e_s.extend(e_chunk.tolist())
        block_w_s.extend(w_chunk.tolist())
        block_echol_s.extend(ene_chol_chunk.tolist())
        w_chunk_avg = jnp.mean(w_chunk)
        e_chunk_avg = jnp.mean(e_chunk * w_chunk) / w_chunk_avg
        elapsed = time.perf_counter() - t0
        dt_per_block = (time.perf_counter() - t_mark) / float(n)
        t_mark = time.perf_counter()
        stats = blocking_analysis_ratio(
            jnp.asarray(block_e_s), jnp.asarray(block_w_s), print_q=False
        )
        mu = float(stats["mu"])
        se = float(stats["se_star"])
        nodes = int(state.node_encounters)
        print(
            f"[blk {start + n:4d}/{params.n_blocks}]  "
            f"{mu:14.10f}  "
            f"{se:10.3e}  "
            f"{float(e_chunk_avg):14.10f}  "
            f"{float(w_chunk_avg):12.6e}  "
            f"{nodes:10d}  "
            f"{dt_per_block:9.3f}  "
            f"{elapsed:8.1f}"
        )
    block_e_s = jnp.asarray(block_e_s)
    block_w_s = jnp.asarray(block_w_s)

    data_clean, _ = reject_outliers(jnp.column_stack((block_e_s, block_w_s)), obs=0)
    print(f"\nRejected {block_e_s.shape[0] - data_clean.shape[0]} outlier blocks.")
    block_e_s = jnp.asarray(data_clean[:, 0])
    block_w_s = jnp.asarray(data_clean[:, 1])
    print("\nFinal blocking analysis:")
    stats = blocking_analysis_ratio(block_e_s, block_w_s, print_q=True)
    mean, err = stats["mu"], stats["se_star"]

    block_e_all = jnp.concatenate([block_e_eq, block_e_s])
    block_w_all = jnp.concatenate([block_w_eq, block_w_s])

    n_chol_pts = len(block_echol_s[0])
    ene_chol_avg = []
    ene_chol_serr = []
    for i in range(n_chol_pts):
        chol_data = jnp.asarray([blk[i] for blk in block_echol_s])
        print(f"\nCholesky energy contribution {i+1}/{n_chol_pts}:")
        stats_i = blocking_analysis_ratio(chol_data, block_w_s, print_q=True)
        ene_chol_avg.append(stats_i["mu"])
        ene_chol_serr.append(stats_i["se_star"])
    print("")
    print("Cholesky energy contributions (avg +/- err):")
    for i in range(n_chol_pts):
        print(
            f"  Cholesky {i+1:3d}: {ene_chol_avg[i]:14.10f} +/- {ene_chol_serr[i]:10.3e}"
        )

    return mean, err, block_e_all, block_w_all


def make_run_blocks_1(
    *,
    block_fn: Callable,
    sys: System,
    params: QmcParams,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    meas_ops_e: MeasOps,
    prop_ops: PropOps,
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
        trial_data_e,
        meas_ctx,
        meas_ctx_e,
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
                trial_data_e=trial_data_e,
                trial_ops=trial_ops,
                meas_ops=meas_ops,
                meas_ops_e=meas_ops_e,
                meas_ctx=meas_ctx,
                meas_ctx_e=meas_ctx_e,
                prop_ops=prop_ops,
                prop_ctx=prop_ctx,
            )
            return state, (obs.scalars["energy"], obs.scalars["weight"])

        stateN, (e, w) = lax.scan(one_block, state0, xs=None, length=n_blocks)
        return stateN, e, w

    return run_blocks


def run_qmc_energy_1(
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    trial_data: Any,
    trial_data_e: Any,
    meas_ops: MeasOps,
    meas_ops_e: MeasOps,
    trial_ops: TrialOps,
    prop_ops: PropOps,
    block_fn: Callable,
    state: PropState | None = None,
    rdm1: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    equilibration blocks then sampling blocks.

    Returns:
      (mean_energy, stderr, block_energies, block_weights)
    """
    print("Starting QMC driver...")
    print(f"Parameters:")
    pprint(params)
    print("")
    # build ctx
    if rdm1 is None:
        rdm1 = trial_ops.get_rdm1(trial_data)
    prop_ctx = prop_ops.build_prop_ctx(ham_data, rdm1, params)
    meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
    meas_ctx_e = meas_ops_e.build_meas_ctx(ham_data, trial_data_e)
    if state is None:
        state = prop_ops.init_prop_state(
            sys=sys,
            ham_data=ham_data,
            trial_ops=trial_ops,
            trial_data=trial_data,
            meas_ops=meas_ops,
            params=params,
            rdm1=rdm1,
        )

    run_blocks = make_run_blocks_1(
        block_fn=block_fn,
        sys=sys,
        params=params,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        meas_ops_e=meas_ops_e,
        prop_ops=prop_ops,
    )

    t0 = time.perf_counter()
    t_mark = t0

    print_every = params.n_eql_blocks // 5 if params.n_eql_blocks >= 5 else 0
    block_e_eq = []
    block_w_eq = []
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
    chunk = print_every
    for start in range(0, params.n_eql_blocks, chunk):
        n = min(chunk, params.n_eql_blocks - start)
        state, e_chunk, w_chunk = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            trial_data_e=trial_data_e,
            meas_ctx=meas_ctx,
            meas_ctx_e=meas_ctx_e,
            prop_ctx=prop_ctx,
            n_blocks=n,
        )
        block_e_eq.extend(e_chunk.tolist())
        block_w_eq.extend(w_chunk.tolist())
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

    # sampling
    print("\nSampling:\n")
    print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 0
    block_e_s = []
    block_w_s = []
    if print_every:
        print(
            f"{'':4s}{'block':>9s}  {'E_avg':>14s}  {'E_err':>10s}  {'E_block':>14s}  "
            f"{'W':>12s}    {'nodes':>10s}  {'dt[s/bl]':>10s}  {'t[s]':>7s}"
        )

    chunk = print_every
    for start in range(0, params.n_blocks, chunk):
        n = min(chunk, params.n_blocks - start)
        state, e_chunk, w_chunk = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            trial_data_e=trial_data_e,
            meas_ctx=meas_ctx,
            meas_ctx_e=meas_ctx_e,
            prop_ctx=prop_ctx,
            n_blocks=n,
        )
        block_e_s.extend(e_chunk.tolist())
        block_w_s.extend(w_chunk.tolist())
        w_chunk_avg = jnp.mean(w_chunk)
        e_chunk_avg = jnp.mean(e_chunk * w_chunk) / w_chunk_avg
        elapsed = time.perf_counter() - t0
        dt_per_block = (time.perf_counter() - t_mark) / float(n)
        t_mark = time.perf_counter()
        stats = blocking_analysis_ratio(
            jnp.asarray(block_e_s), jnp.asarray(block_w_s), print_q=False
        )
        mu = float(stats["mu"])
        se = float(stats["se_star"])
        nodes = int(state.node_encounters)
        print(
            f"[blk {start + n:4d}/{params.n_blocks}]  "
            f"{mu:14.10f}  "
            f"{se:10.3e}  "
            f"{float(e_chunk_avg):14.10f}  "
            f"{float(w_chunk_avg):12.6e}  "
            f"{nodes:10d}  "
            f"{dt_per_block:9.3f}  "
            f"{elapsed:8.1f}"
        )
    block_e_s = jnp.asarray(block_e_s)
    block_w_s = jnp.asarray(block_w_s)

    data_clean, _ = reject_outliers(
        jnp.column_stack((block_e_s, block_w_s)), obs=0, m=1e6
    )
    print(f"\nRejected {block_e_s.shape[0] - data_clean.shape[0]} outlier blocks.")
    block_e_s = jnp.asarray(data_clean[:, 0])
    block_w_s = jnp.asarray(data_clean[:, 1])
    print("\nFinal blocking analysis:")
    stats = blocking_analysis_ratio(block_e_s, block_w_s, print_q=True)
    mean, err = stats["mu"], stats["se_star"]

    block_e_all = jnp.concatenate([block_e_eq, block_e_s])
    block_w_all = jnp.concatenate([block_w_eq, block_w_s])

    return mean, err, block_e_all, block_w_all
