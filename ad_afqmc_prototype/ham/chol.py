from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
from jax import tree_util

HamBasis = Literal["restricted", "generalized"]


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HamChol:
    """
    cholesky hamiltonian.

    basis="restricted":
      h1:   (norb, norb)
      chol: (n_fields, norb, norb)

    basis="generalized":
      h1:   (nso, nso)   where nso = 2*norb
      chol: (n_fields, nso, nso)
    """

    h0: jax.Array
    h1: jax.Array
    chol: jax.Array
    basis: HamBasis = "restricted"

    def __post_init__(self):
        if self.basis not in ("restricted", "generalized"):
            raise ValueError(f"unknown basis: {self.basis}")

    def tree_flatten(self):
        children = (self.h0, self.h1, self.chol)
        aux = self.basis
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        h0, h1, chol = children
        basis = aux
        return cls(h0=h0, h1=h1, chol=chol, basis=basis)


def n_fields(ham: HamChol) -> int:
    return int(ham.chol.shape[0])


def slice_ham_level(ham: HamChol, *, norb_keep: int | None, nchol_keep: int | None) -> HamChol:
    """
    Build a HamChol view for measurement in MLMC:
      - slice orbitals as a prefix [:norb_keep]
      - slice chol as a prefix [:nchol_keep]
    """
    h0 = ham.h0
    h1 = ham.h1
    chol = ham.chol

    if norb_keep is not None:
        h1 = h1[:norb_keep, :norb_keep]
        chol = chol[:, :norb_keep, :norb_keep]

    if nchol_keep is not None:
        chol = chol[:nchol_keep]

    return HamChol(h0=h0, h1=h1, chol=chol, basis=ham.basis)
