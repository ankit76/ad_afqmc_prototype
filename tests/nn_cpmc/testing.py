import numpy as np
from pyscf import gto, scf, ao2mo

def make_hubbard_nn_integrals(lattice, u, v):
    n_sites = lattice.n_sites
    adj = lattice.create_adjacency_matrix()
    bonds = lattice.get_neighboring_bonds(adj)
    h1 = -1.0 * adj
    h2 = np.zeros((n_sites, n_sites, n_sites, n_sites))
    for i in range(n_sites): h2[i, i, i, i] = u
    for bond in bonds:
        i, j = bond
        h2[i, i, j, j] = v
        h2[j, j, i, i] = v
    h2 = ao2mo.restore(8, h2, n_sites)
    return {"h1": h1, "h2": h2, "u": u, "v": v, "bonds": bonds}
