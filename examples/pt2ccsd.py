"""
Example: a manual setup of AFQMC/pt2CCSD energy for 8 non-interacting H2 dimers
================================================================================
"""

from pyscf import cc, gto, scf
from ad_afqmc_prototype.afqmc import AfqmcPt2Ccsd

a = 2  # intra-dimer bond length (Bohr)
d = 100  # centre-to-centre distance between dimers (Bohr)
na = 2  # atoms per monomer (H2)
nc = 8  # number of monomers
elmt = "H"
unit = "b"  # length unit: Bohr
basis = "sto6g"

atoms = ""
for n in range(nc * na):
    shift = ((n - n % na) // na) * (d - a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis=basis, unit=unit, verbose=4)

mf = scf.RHF(mol)
mf.kernel()

mycc = cc.CCSD(mf)
mycc.kernel()

af = AfqmcPt2Ccsd(mycc)
af.kernel()
