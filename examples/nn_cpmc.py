import numpy as np
import scipy as sp
import jax.numpy as jnp
from jax import scipy as jsp

from pyscf import gto, scf, fci, ao2mo
from ad_afqmc_prototype import lattices
from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.ham.hubbard_nn import HamHubbardNN
from ad_afqmc_prototype.trial import ghf as ghf_trial
from ad_afqmc_prototype.trial import multi_ghf as multi_ghf_trial
from ad_afqmc_prototype.meas import ghf as ghf_meas
from ad_afqmc_prototype.meas import multi_ghf as multi_ghf_meas
from ad_afqmc_prototype.prop.types import QmcParams
from ad_afqmc_prototype.prop.nn_cpmc import make_prop_ops
from ad_afqmc_prototype.driver import run_qmc_energy
from ad_afqmc_prototype.prop.blocks import block

# -----------------------------------------------------------------------------
# create lattice
n_elec = (2, 2)
u = 4.0
v = 1.0

#lattice = lattices.TriangularGrid(2, 2, boundary="pbc")
#adj = lattice.create_adjacency_matrix()
#bonds = lattice.get_neighboring_bonds(adj)
#n_sites = lattice.n_sites

n_sites = 4
lattice = lattices.OneDimensionalChain(n_sites)
adj = lattice.create_adjacency_matrix()
bonds = lattice.get_all_neighboring_bonds(adj)

# -----------------------------------------------------------------------------
# custom integrals
h1 = -1.0 * adj
h2 = np.zeros((n_sites, n_sites, n_sites, n_sites))
for i in range(n_sites): h2[i, i, i, i] = u
for bond in bonds:
    i, j = bond
    h2[i, i, j, j] = v
    h2[j, j, i, i] = v

integrals = {
    "h0": 0.0,
    "h1": h1,
    "h2": ao2mo.restore(8, h2, n_sites)
}
ene_h1, evec_h1 = np.linalg.eigh(h1)

# -----------------------------------------------------------------------------
# pyscf
# make dummy molecule
mol = gto.Mole()
mol.nelectron = sum(n_elec)
mol.incore_anyway = True
mol.spin = abs(n_elec[0] - n_elec[1])
mol.nelec = n_elec
mol.nao = n_sites
mol.build()

mf = scf.GHF(mol)
mf.get_hcore = lambda *args: sp.linalg.block_diag(integrals["h1"], integrals["h1"])
mf.get_ovlp = lambda *args: np.eye(2*n_sites)
mf._eri = ao2mo.restore(8, integrals["h2"], n_sites)

dm_init = np.random.random((2*n_sites, 2*n_sites))
dm_init += dm_init.T
dm_init /= 2.0

mf.kernel(dm_init)
mo1 = mf.stability()
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()

if sum(n_elec) <= 12:
    from pyscf import fci

    mol.verbose = 5
    ci = fci.FCI(mol)
    ci.nroots = 2
    ci.max_memory = 10000
    ci.max_cycle = 300
    e, ci_coeffs = ci.kernel(
        h1e=integrals["h1"], eri=integrals["h2"], norb=n_sites, nelec=n_elec, verbose=5
    )
    print(
        f"fci energy: {e},\nspin: {[ci.spin_square(ci_coeff, n_sites, n_elec)[1] for ci_coeff in ci_coeffs]}"
    )

# -----------------------------------------------------------------------------
# data
sys = System(
    norb=lattice.n_sites,
    nelec=n_elec,
    walker_kind="unrestricted",
)
ham_data = HamHubbardNN(h1=jnp.array(h1), u=u, v=v, bonds=jnp.array(bonds))

mf_coeffs = mf.mo_coeff[:, : sum(n_elec)]
ghf_trial_data = ghf_trial.GhfTrial(mf_coeffs)
multi_ghf_trial_data = multi_ghf_trial.MultiGhfTrial(
    ci_coeffs=jnp.array([1.0]),
    mo_coeffs=jnp.array([mf_coeffs]),
    green_complex_dtype=jnp.complex64,
    green_real_dtype=jnp.float32,
)

# -----------------------------------------------------------------------------
# trial and measurement operations
ghf_trial_ops = ghf_trial.make_ghf_trial_ops(sys=sys)
ghf_meas_ops = ghf_meas.make_ghf_meas_ops_hubbard_nn(sys=sys)
multi_ghf_trial_ops = multi_ghf_trial.make_multi_ghf_trial_ops(sys=sys)
multi_ghf_meas_ops = multi_ghf_meas.make_multi_ghf_meas_ops_hubbard_nn(sys=sys)

# -----------------------------------------------------------------------------
# propagation operations
params = QmcParams(
    n_walkers=200, 
    n_eql_blocks=100, 
    n_blocks=500, 
    seed=42
)
prop_ops = make_prop_ops(ham_data, sys.walker_kind, trial_ops=ghf_trial_ops)
#prop_ops = make_prop_ops(ham_data, sys.walker_kind, trial_ops=multi_ghf_trial_ops)

# driver
mean, err, block_e_all, block_w_all = run_qmc_energy(
    sys=sys,
    params=params,
    ham_data=ham_data,
    trial_data=multi_ghf_trial_data,
    trial_ops=multi_ghf_trial_ops,
    meas_ops=multi_ghf_meas_ops,
    prop_ops=prop_ops,
    block_fn=block,
)
