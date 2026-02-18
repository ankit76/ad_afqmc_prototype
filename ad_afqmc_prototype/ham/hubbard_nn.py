from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import tree_util


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HamHubbardNN:
    """
    Nearest-neigbor Hubbard Hamiltonian data.

    h1: one body term  ((norb, norb))
    u: on-site interaction
    v: nearest-neighbor interaction
    bonds: nearest-neighbor bonds ((nbond, 2))
    """

    h1: jax.Array
    u: float
    v: float
    bonds: jax.Array

    def tree_flatten(self):
        return (self.h1, self.u, self.v, self.bonds), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        h1, u, v, bonds = children
        return cls(h1=h1, u=u, v=v, bonds=bonds)
