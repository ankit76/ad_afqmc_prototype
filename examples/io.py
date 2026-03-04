from pyscf import gto, scf
from ad_afqmc_prototype.afqmc import AFQMC, from_staged

mol = gto.M(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis="6-31g",
    verbose=3,
)
mf = scf.RHF(mol)
mf.kernel()

af = AFQMC(mf)
af.chol_cut = 1e-6  # Cholesky decomposition threshold
af.norb_frozen = 1  # freeze O 1s core
af.n_walkers = 20  # number of walkers
af.n_eql_blocks = 10  # number of equilibration blocks
af.n_blocks = 200  # number of sampling blocks
mean1, err1 = af.kernel()  # Not required, juts to show it leads to the exact same result
af.save_staged("h2o.h5")  # Staged in h2o_af.h5

af2 = from_staged("h2o.h5")  # New instance from h2o_af.h5
# af2.norb_frozen = 1  # Cannot be changed as it has been staged
# af2.chol_cut = 1e-6  # Cannot be changed as it has been staged

# Parameters are NOT staged
af2.n_walkers = 20
af2.n_eql_blocks = 10
af2.n_blocks = 200
mean2, err2 = af2.kernel()

assert abs(mean1 - mean2) < 1e-12
assert abs(err1 - err2) < 1e-12
