import numpy as np
from pyscf import gto, scf, ao2mo

def make_hubbard_integrals(lattice, u):
    n_sites = lattice.n_sites
    h1 = -1.0 * lattice.create_adjacency_matrix()
    h2 = np.zeros((n_sites, n_sites, n_sites, n_sites))
    for i in range(n_sites): h2[i, i, i, i] = u
    h2 = ao2mo.restore(8, h2, n_sites)
    return {"h1": h1, "h2": h2, "u": u}
