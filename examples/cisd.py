from pyscf import cc, gto, scf

from ad_afqmc_prototype.afqmc import AFQMC

mol = gto.M(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis="6-31g",
    verbose=3,
)
mf = scf.RHF(mol)
mf.kernel()

mycc = cc.CCSD(mf)
mycc.frozen = 1  # freeze O 1s core
mycc.kernel()
et = mycc.ccsd_t()  # for comparison
print(f"CCSD(T) total energy: {mycc.e_tot + et}")

afqmc = AFQMC(mycc)
afqmc.n_walkers = 100  # number of walkers
afqmc.n_eql_blocks = 10  # number of equilibration blocks
afqmc.n_blocks = 100  # number of sampling blocks
mean, err = afqmc.kernel()
