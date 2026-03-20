from ad_afqmc_prototype import config

config.configure_once()

import pytest
from typing import cast

import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp

from pyscf import gto, scf, ao2mo

from ad_afqmc_prototype.lattices import TriangularGrid
from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.ham.hubbard_nn import HamHubbardNN
from ad_afqmc_prototype.trial.uhf import UhfTrial, make_uhf_trial_ops
from ad_afqmc_prototype.trial.ghf import GhfTrial, make_ghf_trial_ops
from ad_afqmc_prototype.meas.uhf import make_uhf_meas_ops_hubbard_nn
from ad_afqmc_prototype.meas.ghf import make_ghf_meas_ops_hubbard_nn
from ad_afqmc_prototype.prop.nn_cpmc import make_prop_ops
from ad_afqmc_prototype.prop.types import QmcParams
from ad_afqmc_prototype.prop.blocks import block
from ad_afqmc_prototype.driver import run_qmc_energy
from ad_afqmc_prototype.testing import run_calc
from testing import make_hubbard_nn_integrals

dtype = jnp.float64 # Must be real for CPMC.

def mf():
    nx, ny = 4, 4
    bc = 'xc'
    nup, ndn = 2, 1
    
    # lattice
    lattice = TriangularGrid(nx, ny, boundary=bc)
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
    "mf_input, walker_kind",
    [
        (mf_input, "unrestricted"),
        (mf_input, "generalized"),
    ],
)
def test_calc_hubbard(mf_input, params, walker_kind):
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

    print(f'\ne_uhf = {e_uhf}')
    print(f'err_uhf = {err_uhf}')

    # ghf trial from uhf
    ghf_trial_data = GhfTrial(
        mo_coeff=jsp.linalg.block_diag(
            mf.mo_coeff[0][:, :n_elec[0]],
            mf.mo_coeff[1][:, :n_elec[1]]
        )
    )
    ghf_trial_ops = make_ghf_trial_ops(sys)
    ghf_meas_ops = make_ghf_meas_ops_hubbard_nn(sys)
    ghf_prop_ops = make_prop_ops(ham_data, sys.walker_kind, ghf_trial_ops)

    e_ghf, err_ghf, e_all_ghf, w_all_ghf = run_calc(
        sys=sys, 
        meas_ops=ghf_meas_ops,
        ham_data=ham_data,
        trial_ops=ghf_trial_ops,
        trial_data=ghf_trial_data,
        params=params,
        block_fn=block,
        prop_ops=ghf_prop_ops,
    )
    
    print(f'\ne_ghf = {e_ghf}')
    print(f'err_ghf = {err_ghf}')

    # cpmc with ghf trial from uhf should be identical to uhf trial
    np.testing.assert_allclose(e_uhf, e_ghf)

    if (err_uhf is not None) and (err_ghf is not None):
        np.testing.assert_allclose(err_uhf, err_ghf)

    np.testing.assert_allclose(e_all_uhf, e_all_ghf)
    np.testing.assert_allclose(w_all_uhf, w_all_ghf)


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
