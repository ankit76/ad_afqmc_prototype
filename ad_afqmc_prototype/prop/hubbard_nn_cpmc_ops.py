from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import tree_util

from ..ham.hubbard_nn import HamHubbardNN


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HubbardNNCpmcCtx:
    """
    Propagation context for (slow) NN-CPMC.

    exp_h1_half: exp(-dt/2 * h1).
    hs_constant_onsite encodes the discrete HS factors including the overall constant exp(-dt*U/2).
    hs_constant_nn encodes the discrete HS factors including the overall constant exp(-dt*V/2).

      hs_constant* has shape (2, 2):
        hs_constant*[0] -> field 0 factors (up, dn)
        hs_constant*[1] -> field 1 factors (up, dn)
    """

    dt: jax.Array
    exp_h1_half: jax.Array  # (n,n)
    hs_constant_onsite: jax.Array  # (2,2)
    hs_constant_nn: jax.Array  # (2,2)

    def tree_flatten(self):
        return (
            self.dt, 
            self.exp_h1_half, 
            self.hs_constant_onsite, 
            self.hs_constant_nn,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        dt, exp_h1_half, hs_constant_onsite, hs_constant_nn = children
        return cls(
            dt=dt, 
            exp_h1_half=exp_h1_half, 
            hs_constant_onsite=hs_constant_onsite,
            hs_constant_nn=hs_constant_nn,
        )


class HubbardNNCpmcOps(NamedTuple):
    """
    NN-CPMC propagation ops.
    """

    n_sites: Callable[[], int]
    n_bonds: Callable[[], int]
    bonds: Callable[[], jax.Array]
    apply_one_body_half: Callable[[Any, HubbardNNCpmcCtx], Any]


def _build_exp_h1_half(h1: jax.Array, dt: jax.Array) -> jax.Array:
    return jsp.linalg.expm(-0.5 * dt * h1)


def _build_hs_constant(u: jax.Array, v: jax.Array, dt: jax.Array) -> Tuple[jax.Array, jax.Array]:
    # onsite interactions
    gamma_onsite = jnp.arccosh(jnp.exp(0.5 * dt * u))
    const_onsite = jnp.exp(-0.5 * dt * u)
    hs_onsite = jnp.array(
        [[jnp.exp( gamma_onsite), jnp.exp(-gamma_onsite)], 
         [jnp.exp(-gamma_onsite), jnp.exp( gamma_onsite)]], 
        dtype=dt.dtype,
    )

    # nearest-neighbor interactions
    gamma_nn = jnp.arccosh(jnp.exp(0.5 * dt * v))
    const_nn = jnp.exp(-0.5 * dt * v)
    hs_nn = jnp.array(
        [[jnp.exp( gamma_nn), jnp.exp(-gamma_nn)], 
         [jnp.exp(-gamma_nn), jnp.exp( gamma_nn)]], 
        dtype=dt.dtype,
    )

    return (const_onsite * hs_onsite, const_nn * hs_nn)  # (2,2)


def _apply_one_body_half_unrestricted(
    walker: tuple[jax.Array, jax.Array], prop_ctx: HubbardNNCpmcCtx
) -> tuple[jax.Array, jax.Array]:
    """
    Apply one body half step to a batch of unrestricted Hubbard walkers.

    walkers is expected to be a tuple/list (w_up, w_dn), each with shape (nw, n, ne_sigma).
    """
    w_up, w_dn = walker
    w_up = prop_ctx.exp_h1_half @ w_up
    w_dn = prop_ctx.exp_h1_half @ w_dn
    return (w_up, w_dn)


def _build_prop_ctx(ham_data: HamHubbardNN, dt: float) -> HubbardNNCpmcCtx:
    dt_a = jnp.asarray(dt)
    u_a = jnp.asarray(ham_data.u)
    v_a = jnp.asarray(ham_data.v)
    exp_h1_half = _build_exp_h1_half(ham_data.h1, dt_a)  # (n,n)
    hs_constant_onsite, hs_constant_nn = _build_hs_constant(u_a, v_a, dt_a)  # (2,2)
    return HubbardNNCpmcCtx(
        dt=dt_a, 
        exp_h1_half=exp_h1_half, 
        hs_constant_onsite=hs_constant_onsite,
        hs_constant_nn=hs_constant_nn,
    )


def make_hubbard_nn_cpmc_ops(ham_data: HamHubbardNN, walker_kind: str) -> HubbardNNCpmcOps:
    assert (
        walker_kind.lower() == "unrestricted"
    ), "only unrestricted walkers supported for hubbard_cpmc_ops"

    def n_sites() -> int:
        return int(ham_data.h1.shape[-1])

    def n_bonds() -> int:
        return int(ham_data.bonds.shape[0])

    def bonds() -> jax.Array:
        return ham_data.bonds

    return HubbardNNCpmcOps(
        n_sites=n_sites,
        n_bonds=n_bonds,
        bonds=bonds,
        apply_one_body_half=_apply_one_body_half_unrestricted,
    )
