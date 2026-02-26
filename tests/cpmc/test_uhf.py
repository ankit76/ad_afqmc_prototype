from ad_afqmc_prototype import config

config.configure_once()

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from pyscf import gto, scf

from ad_afqmc_prototype import testing
from ad_afqmc_prototype.afqmc import AFQMC
from ad_afqmc_prototype.core.ops import k_energy
from ad_afqmc_prototype.meas.uhf import (
    energy_kernel_uw_hubbard,
    energy_kernel_gw_hubbard,
    make_uhf_meas_ops_hubbard,
)
from ad_afqmc_prototype.prop.types import QmcParams
from ad_afqmc_prototype.trial.uhf import UhfTrial, make_uhf_trial_ops


def _make_uhf_trial(key, n_sites, nup, ndn, dtype=jnp.complex128) -> UhfTrial:
    ka, kb = jax.random.split(key)
    ca = testing.rand_orthonormal_cols(ka, n_sites, nup, dtype=dtype)
    cb = testing.rand_orthonormal_cols(kb, n_sites, ndn, dtype=dtype)
    return UhfTrial(mo_coeff_a=ca, mo_coeff_b=cb)


def test_energy_equal_when_wg_eq_wu():
    n_sites = 6
    nup, ndn = 2, 1
    walker_kind = "unrestricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_hubbard(
        key,
        walker_kind,
        n_sites,
        (nup, ndn),
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            n_sites=n_sites,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        build_meas_ctx_fn=None,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        wi = cast(tuple, wi)
        eu = energy_kernel_uw_hubbard(wi, ham, ctx, trial)
        wa, wb = wi
        wi = jnp.zeros((2 * n_sites, nup + ndn), dtype=wa.dtype)
        wi = lax.dynamic_update_slice(wi, wa, (0, 0))
        wi = lax.dynamic_update_slice(wi, wb, (n_sites, nup))
        eg = energy_kernel_gw_hubbard(wi, ham, ctx, trial)

        assert jnp.allclose(eu, eg, atol=1e-12), (eu, eg)

#def mf():
#    mol = gto.M(
#        atom="""
#        O        0.0000000000      0.0000000000      0.0000000000
#        H        0.9562300000      0.0000000000      0.0000000000
#        H       -0.2353791634      0.9268076728      0.0000000000
#        """,
#        basis="sto-6g",
#    )
#    mf = scf.UHF(mol).newton()
#    mf.kernel()
#    return mf
#
#
#def mf2():
#    mol = gto.M(
#        atom="""
#        N        0.0000000000      0.0000000000      0.0000000000
#        H        1.0225900000      0.0000000000      0.0000000000
#        H       -0.2281193615      0.9968208791      0.0000000000
#        """,
#        basis="sto-6g",
#        spin=1,
#    )
#    mf = scf.UHF(mol).newton()
#    mf.kernel()
#    return mf
#
#
#mf = mf()  # type: ignore
#mf2 = mf2()  # type: ignore
#
#
#@pytest.mark.parametrize(
#    "mf, walker_kind, e_ref, err_ref",
#    [
#        (mf, "restricted", -75.75594187783527, 0.01213383697785241),
#        (mf2, "unrestricted", -55.43066756011652, 0.00761980459817991),
#        (mf2, "generalized", -55.43066756011653, 0.007619804598170696),
#    ],
#)
#def test_calc_rhf_hamiltonian(mf, params, walker_kind, e_ref, err_ref):
#    myafqmc = AFQMC(mf)
#    myafqmc.params = params
#    myafqmc.walker_kind = walker_kind
#    myafqmc.mixed_precision = False
#    myafqmc.chol_cut = 1e-6
#    mean, err = myafqmc.kernel()
#    assert jnp.isclose(mean, e_ref), (mean, e_ref, mean - e_ref)
#    assert jnp.isclose(err, err_ref), (err, err_ref, err - err_ref)
#
#
#@pytest.fixture(scope="module")
#def params():
#    return QmcParams(
#        n_eql_blocks=4,
#        n_blocks=20,
#        seed=1234,
#        n_walkers=5,
#    )

if __name__ == "__main__":
    pytest.main([__file__])
