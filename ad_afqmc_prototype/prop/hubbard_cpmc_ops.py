from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import tree_util

from ..ham.hubbard import HamHubbard


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HubbardCpmcCtx:
    """
    Propagation context for (slow) CPMC.

    exp_h1_half: exp(-dt/2 * h1).
    hs_constant encodes the discrete HS factors including the overall constant exp(-dt*U/2).

      hs_constant has shape (2, 2):
        hs_constant[0] -> field 0 factors (up, dn)
        hs_constant[1] -> field 1 factors (up, dn)
    """

    dt: jax.Array
    exp_h1_half: jax.Array  # (n,n)
    hs_constant: jax.Array  # (2,2)

    def tree_flatten(self):
        return (self.dt, self.exp_h1_half, self.hs_constant), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        dt, exp_h1_half, hs_constant = children
        return cls(dt=dt, exp_h1_half=exp_h1_half, hs_constant=hs_constant)


class HubbardCpmcOps(NamedTuple):
    """
    CPMC propagation ops.
    """

    n_sites: Callable[[], int]
    apply_one_body_half: Callable[[Any, HubbardCpmcCtx], Any]


def _build_exp_h1_half(h1: jax.Array, dt: jax.Array) -> jax.Array:
    return jsp.linalg.expm(-0.5 * dt * h1)


def _build_hs_constant(u: jax.Array, dt: jax.Array) -> jax.Array:
    gamma = jnp.arccosh(jnp.exp(0.5 * dt * u))
    const = jnp.exp(-0.5 * dt * u)
    hs = jnp.array(
        [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]],
        dtype=dt.dtype,
    )
    return const * hs  # (2,2)


def _apply_one_body_half_unrestricted(
    walker: tuple[jax.Array, jax.Array], prop_ctx: HubbardCpmcCtx
) -> tuple[jax.Array, jax.Array]:
    """
    Apply one body half step to a batch of unrestricted Hubbard walkers.

    walkers is expected to be a tuple/list (wu, wd), each with shape (nw, n, ne_sigma).
    """
    wu, wd = walker
    wu = prop_ctx.exp_h1_half @ wu
    wd = prop_ctx.exp_h1_half @ wd
    return (wu, wd)


def _apply_one_body_half_generalized(
    walker: jax.Array, prop_ctx: HubbardCpmcCtx
) -> tuple[jax.Array, jax.Array]:
    """
    Apply one body half step to a batch of unrestricted Hubbard walkers.

    walkers is expected to be a tuple/list (wu, wd), each with shape (nw, n, ne_sigma).
    """
    norb = walker.shape[0] // 2
    top = prop_ctx.exp_h1_half @ walker[:norb, :]
    bot = prop_ctx.exp_h1_half @ walker[norb:, :]
    return jnp.vstack([top, bot])


def _build_prop_ctx(ham_data: HamHubbard, dt: float) -> HubbardCpmcCtx:
    dt_a = jnp.asarray(dt)
    u_a = jnp.asarray(ham_data.u)
    exp_h1_half = _build_exp_h1_half(ham_data.h1, dt_a)  # (n,n)
    hs_constant = _build_hs_constant(u_a, dt_a)  # (2,2)
    return HubbardCpmcCtx(dt=dt_a, exp_h1_half=exp_h1_half, hs_constant=hs_constant)


def make_hubbard_cpmc_ops(ham_data: HamHubbard, walker_kind: str) -> HubbardCpmcOps:
    assert type(walker_kind) == str

    walker_kind = walker_kind.lower()

    if walker_kind not in ("unrestricted", "generalized"):
        raise ValueError(f"unknown walker_kind: {walker_kind}")
    
    def n_sites() -> int:
        return int(ham_data.h1.shape[-1])

    if walker_kind == "unrestricted":
        return HubbardCpmcOps(
            n_sites=n_sites,
            apply_one_body_half=_apply_one_body_half_unrestricted,
        )

    elif walker_kind == "generalized":
        return HubbardCpmcOps(
            n_sites=n_sites,
            apply_one_body_half=_apply_one_body_half_generalized,
        )
