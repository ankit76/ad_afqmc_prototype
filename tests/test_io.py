from ad_afqmc_prototype import config

config.configure_once()

import pytest
import jax.numpy as jnp
from pyscf import gto, scf
from ad_afqmc_prototype.afqmc import AFQMC, from_staged
from ad_afqmc_prototype.prop.types import QmcParams


def mf():
    mol = gto.M(
        atom="""
        N        0.0000000000      0.0000000000      0.0000000000
        H        1.0225900000      0.0000000000      0.0000000000
        H       -0.2281193615      0.9968208791      0.0000000000
        """,
        basis="sto-6g",
        spin=1,
    )
    mf = scf.UHF(mol).newton()
    mf.kernel()
    return mf


mf = mf()  # type: ignore


@pytest.mark.parametrize(
    "mf, walker_kind, e_ref, err_ref",
    [
        (mf, "unrestricted", -55.43066756011652, 0.00761980459817991),
    ],
)
def test_calc_rhf_hamiltonian(mf, tmp_path, params, walker_kind, e_ref, err_ref):
    h5_file = str(tmp_path / "nh2.h5")
    myafqmc = AFQMC(mf)
    myafqmc.chol_cut = 1e-6
    myafqmc.save_staged(h5_file)

    af = from_staged(h5_file)
    af.params = params
    af.mixed_precision = False
    af.walker_kind = walker_kind
    mean, err = af.kernel()
    assert jnp.isclose(mean, e_ref), (mean, e_ref, mean - e_ref)
    assert jnp.isclose(err, err_ref), (err, err_ref, err - err_ref)


@pytest.fixture(scope="module")
def params():
    return QmcParams(
        n_eql_blocks=4,
        n_blocks=20,
        seed=1234,
        n_walkers=5,
    )


if __name__ == "__main__":
    pytest.main([__file__])
