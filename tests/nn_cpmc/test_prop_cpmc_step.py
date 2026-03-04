from ad_afqmc_prototype import config
config.configure_once()

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.ham.hubbard_nn import HamHubbardNN
from ad_afqmc_prototype.prop.nn_cpmc import cpmc_step
from ad_afqmc_prototype.prop.nn_cpmc_slow import cpmc_step as cpmc_step_slow
from ad_afqmc_prototype.prop.hubbard_nn_cpmc_ops import (
    _build_prop_ctx, 
    make_hubbard_nn_cpmc_ops
)
from ad_afqmc_prototype.trial.uhf import make_uhf_trial_ops
from ad_afqmc_prototype.meas.uhf import make_uhf_meas_ops_hubbard_nn
from ad_afqmc_prototype.trial.ghf import make_ghf_trial_ops
from ad_afqmc_prototype.meas.ghf import make_ghf_meas_ops_hubbard_nn
from ad_afqmc_prototype.prop.types import PropState, QmcParams
from ad_afqmc_prototype.testing import (
    make_random_ham_hubbard_nn,
    make_random_uhf_trial,
    make_random_ghf_trial,
    make_walkers,
)

# ---------------------
# Unrestricted walkers
# ---------------------

def test_unrestricted_step_matches_nn_cpmc_slow():
    key = jax.random.PRNGKey(42)
    norb, n_bonds, nup, ndn, nw = 5, 3, 2, 1, 6
    params = QmcParams(dt=0.1, n_chunks=1)
    sys = System(norb=norb, nelec=(nup, ndn), walker_kind="unrestricted")
    ham = make_random_ham_hubbard_nn(key, norb, n_bonds)
    trial_data = make_random_uhf_trial(key, norb, nup, ndn)
    trial_ops = make_uhf_trial_ops(sys)
    meas_ops = make_uhf_meas_ops_hubbard_nn(sys)
    walkers = make_walkers(key, sys, nw)

    state = PropState(
        walkers=walkers,
        weights=jnp.ones((nw,)),
        overlaps=jnp.ones((nw,), dtype=jnp.complex64),
        rng_key=jax.random.PRNGKey(0),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )
    
    cpmc_ops = make_hubbard_nn_cpmc_ops(ham, sys.walker_kind)
    prop_ctx = _build_prop_ctx(ham, params.dt, sys.walker_kind)

    out = cpmc_step(
        state,
        params=params,
        trial_ops=trial_ops,
        trial_data=trial_data,
        meas_ops=meas_ops,
        cpmc_ops=cpmc_ops,
        prop_ctx=prop_ctx,
    )

    out_slow = cpmc_step_slow(
        state,
        params=params,
        trial_data=trial_data,
        meas_ops=meas_ops,
        cpmc_ops=cpmc_ops,
        prop_ctx=prop_ctx,
    )
    
    np.testing.assert_allclose(out.weights, out_slow.weights)
    np.testing.assert_allclose(out.walkers[0], out_slow.walkers[0]) 
    np.testing.assert_allclose(out.walkers[1], out_slow.walkers[1]) 
    np.testing.assert_allclose(out.overlaps, out_slow.overlaps)  
    np.testing.assert_allclose(out.node_encounters, out_slow.node_encounters)  
    np.testing.assert_allclose(out.pop_control_ene_shift, out_slow.pop_control_ene_shift)  
    assert jnp.all(out.rng_key == out_slow.rng_key)

def test_unrestricted_step_is_chunk_invariant():
    key = jax.random.PRNGKey(42)
    norb, n_bonds, nup, ndn, nw = 5, 3, 2, 1, 6
    sys = System(norb=norb, nelec=(nup, ndn), walker_kind="unrestricted")
    ham = make_random_ham_hubbard_nn(key, norb, n_bonds)
    trial_data = make_random_uhf_trial(key, norb, nup, ndn)
    trial_ops = make_uhf_trial_ops(sys)
    meas_ops = make_uhf_meas_ops_hubbard_nn(sys)
    walkers = make_walkers(key, sys, nw)

    params1 = QmcParams(dt=0.1, n_chunks=1)
    params2 = QmcParams(dt=0.1, n_chunks=3)

    state = PropState(
        walkers=walkers,
        weights=jnp.ones((nw,)),
        overlaps=jnp.ones((nw,), dtype=jnp.complex64),
        rng_key=jax.random.PRNGKey(0),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )
    
    cpmc_ops = make_hubbard_nn_cpmc_ops(ham, sys.walker_kind)
    prop_ctx = _build_prop_ctx(ham, params1.dt, sys.walker_kind)

    out1 = cpmc_step(
        state,
        params=params1,
        trial_ops=trial_ops,
        trial_data=trial_data,
        meas_ops=meas_ops,
        cpmc_ops=cpmc_ops,
        prop_ctx=prop_ctx,
    )

    out2 = cpmc_step(
        state,
        params=params2,
        trial_ops=trial_ops,
        trial_data=trial_data,
        meas_ops=meas_ops,
        cpmc_ops=cpmc_ops,
        prop_ctx=prop_ctx,
    )

    np.testing.assert_allclose(out1.weights, out2.weights)
    np.testing.assert_allclose(out1.walkers[0], out2.walkers[0]) 
    np.testing.assert_allclose(out1.walkers[1], out2.walkers[1]) 
    np.testing.assert_allclose(out1.overlaps, out2.overlaps)  
    np.testing.assert_allclose(out1.node_encounters, out2.node_encounters)  
    np.testing.assert_allclose(out1.pop_control_ene_shift, out2.pop_control_ene_shift)  
    assert jnp.all(out1.rng_key == out2.rng_key)

# ---------------------
# Generalized walkers
# ---------------------

def test_generalized_step_matches_nn_cpmc_slow():
    key = jax.random.PRNGKey(42)
    norb, n_bonds, nup, ndn, nw = 5, 3, 2, 1, 6
    params = QmcParams(dt=0.1, n_chunks=1)
    sys = System(norb=norb, nelec=(nup, ndn), walker_kind="generalized")
    ham = make_random_ham_hubbard_nn(key, norb, n_bonds)
    trial_data = make_random_ghf_trial(key, norb, nup, ndn)
    trial_ops = make_ghf_trial_ops(sys)
    meas_ops = make_ghf_meas_ops_hubbard_nn(sys)
    walkers = make_walkers(key, sys, nw)

    state = PropState(
        walkers=walkers,
        weights=jnp.ones((nw,)),
        overlaps=jnp.ones((nw,), dtype=jnp.complex64),
        rng_key=jax.random.PRNGKey(0),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )
    
    cpmc_ops = make_hubbard_nn_cpmc_ops(ham, sys.walker_kind)
    prop_ctx = _build_prop_ctx(ham, params.dt, sys.walker_kind)

    out = cpmc_step(
        state,
        params=params,
        trial_ops=trial_ops,
        trial_data=trial_data,
        meas_ops=meas_ops,
        cpmc_ops=cpmc_ops,
        prop_ctx=prop_ctx,
    )

    out_slow = cpmc_step_slow(
        state,
        params=params,
        trial_data=trial_data,
        meas_ops=meas_ops,
        cpmc_ops=cpmc_ops,
        prop_ctx=prop_ctx,
    )
    
    np.testing.assert_allclose(out.weights, out_slow.weights)
    np.testing.assert_allclose(out.walkers, out_slow.walkers) 
    np.testing.assert_allclose(out.overlaps, out_slow.overlaps)  
    np.testing.assert_allclose(out.node_encounters, out_slow.node_encounters)  
    np.testing.assert_allclose(out.pop_control_ene_shift, out_slow.pop_control_ene_shift)  
    assert jnp.all(out.rng_key == out_slow.rng_key)

def test_generalized_step_is_chunk_invariant():
    key = jax.random.PRNGKey(42)
    norb, n_bonds, nup, ndn, nw = 5, 3, 2, 1, 6
    sys = System(norb=norb, nelec=(nup, ndn), walker_kind="generalized")
    ham = make_random_ham_hubbard_nn(key, norb, n_bonds)
    trial_data = make_random_ghf_trial(key, norb, nup, ndn)
    trial_ops = make_ghf_trial_ops(sys)
    meas_ops = make_ghf_meas_ops_hubbard_nn(sys)
    walkers = make_walkers(key, sys, nw)

    params1 = QmcParams(dt=0.1, n_chunks=1)
    params2 = QmcParams(dt=0.1, n_chunks=3)

    state = PropState(
        walkers=walkers,
        weights=jnp.ones((nw,)),
        overlaps=jnp.ones((nw,), dtype=jnp.complex64),
        rng_key=jax.random.PRNGKey(0),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )
    
    cpmc_ops = make_hubbard_nn_cpmc_ops(ham, sys.walker_kind)
    prop_ctx = _build_prop_ctx(ham, params1.dt, sys.walker_kind)

    out1 = cpmc_step(
        state,
        params=params1,
        trial_ops=trial_ops,
        trial_data=trial_data,
        meas_ops=meas_ops,
        cpmc_ops=cpmc_ops,
        prop_ctx=prop_ctx,
    )

    out2 = cpmc_step(
        state,
        params=params2,
        trial_ops=trial_ops,
        trial_data=trial_data,
        meas_ops=meas_ops,
        cpmc_ops=cpmc_ops,
        prop_ctx=prop_ctx,
    )

    np.testing.assert_allclose(out1.weights, out2.weights)
    np.testing.assert_allclose(out1.walkers, out2.walkers) 
    np.testing.assert_allclose(out1.overlaps, out2.overlaps)  
    np.testing.assert_allclose(out1.node_encounters, out2.node_encounters)  
    np.testing.assert_allclose(out1.pop_control_ene_shift, out2.pop_control_ene_shift)  
    assert jnp.all(out1.rng_key == out2.rng_key)



if __name__ == "__main__":
    pytest.main([__file__])
