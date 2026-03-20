from ad_afqmc_prototype import config

config.configure_once()

import pytest
from typing import cast

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from pyscf import gto, scf, ao2mo

from ad_afqmc_prototype.lattices import TriangularGrid
from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.ham.hubbard_nn import HamHubbardNN
from ad_afqmc_prototype.trial.uhf import UhfTrial, make_uhf_trial_ops
from ad_afqmc_prototype.trial.ghf import GhfTrial, make_ghf_trial_ops
from ad_afqmc_prototype.meas.uhf import (
    make_uhf_meas_ops_hubbard_nn,
    energy_kernel_uw_hubbard_nn,
    energy_kernel_gw_hubbard_nn,
)
from ad_afqmc_prototype.prop.nn_cpmc import make_prop_ops
from ad_afqmc_prototype.prop.types import QmcParams
from ad_afqmc_prototype.prop.blocks import block
from ad_afqmc_prototype.driver import run_qmc_energy
from ad_afqmc_prototype.testing import (
    make_walkers,
    make_random_uhf_trial,
    make_common_hubbard_nn,
    run_calc,
)
from testing import make_hubbard_nn_integrals

dtype = jnp.float64 # Must be real for CPMC.

def test_energy_equal_when_wg_eq_wu():
    norb, n_bonds, nup, ndn = 6, 3, 2, 1
    walker_kind = "unrestricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
    ) = make_common_hubbard_nn(
        key,
        walker_kind,
        norb,
        n_bonds,
        (nup, ndn),
        make_trial_fn=make_random_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
            dtype=dtype,
        ),
    )

    for i in range(4):
        wi = make_walkers(jax.random.fold_in(k_w, i), sys, dtype=dtype)
        wi = cast(tuple, wi)
        eu = energy_kernel_uw_hubbard_nn(wi, ham, None, trial)
        wa, wb = wi
        wi = jnp.zeros((2 * norb, nup + ndn), dtype=wa.dtype)
        wi = lax.dynamic_update_slice(wi, wa, (0, 0))
        wi = lax.dynamic_update_slice(wi, wb, (norb, nup))
        eg = energy_kernel_gw_hubbard_nn(wi, ham, None, trial)

        assert jnp.allclose(eu, eg, atol=1e-12), (eu, eg)


def mf():
    nx, ny = 4, 4
    bc = 'xc'
    nup, ndn = 2, 2
    
    # lattice
    lattice = TriangularGrid(nx, ny, boundary=bc)
    adj = lattice.create_adjacency_matrix()
    bonds = lattice.get_neighboring_bonds(adj)
    n_sites = lattice.n_sites
    nocc = nup + ndn

    # integrals
    u = 12.
    v = 4.
    integrals = make_hubbard_nn_integrals(lattice, u, v)

    # make dummy molecule
    mol = gto.Mole()
    mol.nelectron = nocc
    mol.incore_anyway = True
    mol.spin = abs(nup - ndn)
    mol.build()

    # uhf
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: integrals["h1"]
    mf.get_ovlp = lambda *args: np.eye(n_sites)
    mf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
    dm0 = lattice.get_neel_guess().real
    dm0 = [dm0[:n_sites, :n_sites], dm0[n_sites:, n_sites:]]
    mf.kernel(dm0)
    return mf, integrals

mf_input = mf()  # type: ignore

@pytest.mark.parametrize(
    "mf_input, walker_kind, e_ref, err_ref",
    [
        (mf_input, "unrestricted", 0., None),
    ],
)
def test_calc_hubbard_nn(mf_input, params, walker_kind, e_ref, err_ref):
    mf, integrals = mf_input
    n_elec = mf.mol.nelec
    u = integrals["u"]
    v = integrals["v"]
    h1 = integrals["h1"]
    bonds = integrals["bonds"]
    n_sites = h1.shape[0]
    
    sys = System(
        norb=n_sites,
        nelec=n_elec,
        walker_kind=walker_kind,
    )
    ham_data = HamHubbardNN(h1=jnp.array(h1), u=u, v=v, bonds=jnp.array(bonds))

    # uhf trial
    uhf_trial_data = UhfTrial(
        mo_coeff_a=jnp.array(mf.mo_coeff[0][:, :n_elec[0]]),
        mo_coeff_b=jnp.array(mf.mo_coeff[1][:, :n_elec[1]]),
    )
    uhf_trial_ops = make_uhf_trial_ops(sys)
    uhf_meas_ops = make_uhf_meas_ops_hubbard_nn(sys)
    uhf_prop_ops = make_prop_ops(ham_data, sys.walker_kind, uhf_trial_ops)

    e_uhf, err_uhf, e_all_uhf, w_all_uhf = run_calc(
        sys=sys, 
        meas_ops=uhf_meas_ops,
        ham_data=ham_data,
        trial_ops=uhf_trial_ops,
        trial_data=uhf_trial_data,
        params=params,
        block_fn=block,
        prop_ops=uhf_prop_ops,
    )
    
    np.testing.assert_allclose(e_uhf, e_ref)

    if (err_uhf is not None) and (err_ref is not None):
        np.testing.assert_allclose(err_uhf, err_ref)


@pytest.fixture(scope="module")
def params():
    return QmcParams(
        dt=0.005,
        n_eql_blocks=5,
        n_blocks=5,
        seed=42,
        n_walkers=5,
        weight_floor=1e-8
    )


if __name__ == "__main__":
    pytest.main([__file__])

    #test_energy_equal_when_wg_eq_wu()
