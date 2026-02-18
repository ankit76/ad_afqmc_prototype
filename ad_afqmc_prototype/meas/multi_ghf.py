from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ..core.ops import MeasOps, k_energy
from ..core.system import System
from ..ham.hubbard import HamHubbard
from ..ham.hubbard_nn import HamHubbardNN
from ..trial.multi_ghf import (
    MultiGhfTrial,
    calc_green_g,
    calc_green_u,
    overlap_g,
    overlap_u,
)
from .ghf import _energy_from_full_green, _energy_from_full_green_nn

# ---------------------
# hubbard
# ---------------------


def energy_kernel_hubbard_u(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamHubbard,
    meas_ctx: Any,
    trial_data: MultiGhfTrial,
) -> jax.Array:
    """
    Multi-det local energy
    """
    greens = calc_green_u(walker, trial_data)
    G_states = greens["G"]  # (nd,2n,2n)
    w_states = greens["w"]  # (nd,)

    norb = trial_data.norb
    E_k = jax.vmap(lambda G: _energy_from_full_green(G, ham_data, norb), in_axes=0)(
        G_states
    )

    num = jnp.sum(w_states * E_k)
    den = jnp.sum(w_states)

    E = jnp.where(jnp.abs(den) < 1.0e-16, 0.0 + 0.0j, num / den)
    return jnp.real(E)


def energy_kernel_hubbard_g(
    walker: jax.Array,
    ham_data: HamHubbard,
    meas_ctx: Any,
    trial_data: MultiGhfTrial,
) -> jax.Array:
    greens = calc_green_g(walker, trial_data)
    G_states = greens["G"]
    w_states = greens["w"]

    norb = trial_data.norb
    E_k = jax.vmap(lambda G: _energy_from_full_green(G, ham_data, norb), in_axes=0)(
        G_states
    )

    num = jnp.sum(w_states * E_k)
    den = jnp.sum(w_states)

    E = jnp.where(jnp.abs(den) < 1.0e-16, 0.0 + 0.0j, num / den)
    return jnp.real(E)


def make_multi_ghf_meas_ops_hubbard(sys: System) -> MeasOps:
    # meant to be used with CPMC, which requires real overlaps
    wk = sys.walker_kind.lower()

    if wk == "unrestricted":

        def real_overlap_u(
            walker: tuple[jax.Array, jax.Array], trial_data: MultiGhfTrial
        ) -> jax.Array:
            ov = overlap_u(walker, trial_data)
            return jnp.real(ov)

        return MeasOps(
            overlap=real_overlap_u,
            kernels={k_energy: energy_kernel_hubbard_u},
        )

    if wk == "generalized":

        def real_overlap_g(walker: jax.Array, trial_data: MultiGhfTrial) -> jax.Array:
            ov = overlap_g(walker, trial_data)
            return jnp.real(ov)

        return MeasOps(
            overlap=real_overlap_g,
            kernels={k_energy: energy_kernel_hubbard_g},
        )

    raise ValueError(
        f"Hubbard multi-GHF meas only implemented for unrestricted/generalized; got walker_kind={sys.walker_kind}"
    )


# ---------------------
# hubbard_nn
# ---------------------


def energy_kernel_hubbard_nn_u(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamHubbardNN,
    meas_ctx: Any,
    trial_data: MultiGhfTrial,
) -> jax.Array:
    """
    Multi-det local energy
    """
    greens = calc_green_u(walker, trial_data)
    G_states = greens["G"]  # (nd,2n,2n)
    w_states = greens["w"]  # (nd,)

    norb = trial_data.norb
    E_k = jax.vmap(lambda G: _energy_from_full_green_nn(G, ham_data, norb), in_axes=0)(
        G_states
    )

    num = jnp.sum(w_states * E_k)
    den = jnp.sum(w_states)

    E = jnp.where(jnp.abs(den) < 1.0e-16, 0.0 + 0.0j, num / den)
    return jnp.real(E)


def energy_kernel_hubbard_nn_g(
    walker: jax.Array,
    ham_data: HamHubbardNN,
    meas_ctx: Any,
    trial_data: MultiGhfTrial,
) -> jax.Array:
    greens = calc_green_g(walker, trial_data)
    G_states = greens["G"]
    w_states = greens["w"]

    norb = trial_data.norb
    E_k = jax.vmap(lambda G: _energy_from_full_green_nn(G, ham_data, norb), in_axes=0)(
        G_states
    )

    num = jnp.sum(w_states * E_k)
    den = jnp.sum(w_states)

    E = jnp.where(jnp.abs(den) < 1.0e-16, 0.0 + 0.0j, num / den)
    return jnp.real(E)


def make_multi_ghf_meas_ops_hubbard_nn(sys: System) -> MeasOps:
    # meant to be used with CPMC, which requires real overlaps
    wk = sys.walker_kind.lower()

    if wk == "unrestricted":

        def real_overlap_u(
            walker: tuple[jax.Array, jax.Array], trial_data: MultiGhfTrial
        ) -> jax.Array:
            ov = overlap_u(walker, trial_data)
            return jnp.real(ov)

        return MeasOps(
            overlap=real_overlap_u,
            kernels={k_energy: energy_kernel_hubbard_nn_u},
        )

    if wk == "generalized":

        def real_overlap_g(walker: jax.Array, trial_data: MultiGhfTrial) -> jax.Array:
            ov = overlap_g(walker, trial_data)
            return jnp.real(ov)

        return MeasOps(
            overlap=real_overlap_g,
            kernels={k_energy: energy_kernel_hubbard_nn_g},
        )

    raise ValueError(
        f"Nearest-neighbor Hubbard multi-GHF meas only implemented for unrestricted/generalized; got walker_kind={sys.walker_kind}"
    )
