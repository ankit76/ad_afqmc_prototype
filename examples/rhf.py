from pyscf import gto, scf

from ad_afqmc_prototype.afqmc import AFQMC

mol = gto.M(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis="6-31g",
    verbose=3,
)
mf = scf.RHF(mol)
mf.kernel()

myafqmc = AFQMC(mf)
myafqmc.norb_frozen = 1  # freeze O 1s core
myafqmc.n_walkers = 200  # number of walkers
myafqmc.n_eql_blocks = 10  # number of equilibration blocks
myafqmc.n_blocks = 200  # number of sampling blocks
mean, err = myafqmc.kernel()
