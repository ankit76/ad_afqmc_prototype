from ad_afqmc_prototype import config

config.configure_once()


import jax
import jax.numpy as jnp
import pytest

from ad_afqmc_prototype import testing
from ad_afqmc_prototype.core.ops import k_energy, k_force_bias
from ad_afqmc_prototype.meas.cis import make_cis_meas_ops
from ad_afqmc_prototype.trial.cis import CisTrial, make_cis_trial_ops


def _make_cis_trial(
    key,
    norb: int,
    nocc: int,
    *,
    dtype=jnp.float64,
    scale_ci1: float = 0.05,
) -> CisTrial:
    """
    Random CIS coefficients in the MO basis where the reference occupies [0..nocc-1].

    We keep coefficients modest in magnitude to reduce catastrophic cancellation
    when comparing against overlap-based finite differences.
    """
    nvir = norb - nocc
    k1, k2 = jax.random.split(key)

    ci1 = scale_ci1 * jax.random.normal(k1, (nocc, nvir), dtype=dtype)

    return CisTrial(ci1=ci1)


@pytest.mark.parametrize("norb,nocc,n_chol", [(10, 4, 12)])
def test_auto_force_bias_matches_manual_cis(norb, nocc, n_chol):
    walker_kind = "restricted"
    key = jax.random.PRNGKey(123)
    k_ham, k_trial, k_w = jax.random.split(key, 3)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nocc, nocc),
        n_chol,
        make_trial_fn=_make_cis_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nocc=nocc,
        ),
        make_trial_ops_fn=make_cis_trial_ops,
        make_meas_ops_fn=make_cis_meas_ops,
    )

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = testing.make_restricted_walker_near_ref(
            jax.random.fold_in(k_w, i), norb, nocc, mix=0.25
        )

        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)

        # CISD overlap is more structured than RHF; auto (finite-diff / overlap-derivative)
        # can need a slightly looser tolerance.
        assert jnp.allclose(v_a, v_m, rtol=2e-5, atol=2e-6), (v_a, v_m)


@pytest.mark.parametrize("norb,nocc,n_chol", [(10, 4, 12)])
def test_auto_energy_matches_manual_cis(norb, nocc, n_chol):
    walker_kind = "restricted"
    key = jax.random.PRNGKey(456)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nocc, nocc),
        n_chol,
        make_trial_fn=_make_cis_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nocc=nocc,
        ),
        make_trial_ops_fn=make_cis_trial_ops,
        make_meas_ops_fn=make_cis_meas_ops,
    )

    if not meas_manual.has_kernel(k_energy):
        pytest.skip("manual CIS meas does not provide k_energy")

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(4):
        wi = testing.make_restricted_walker_near_ref(
            jax.random.fold_in(k_w, i), norb, nocc, mix=0.25
        )

        em = e_manual(wi, ham, ctx_manual, trial)
        ea = e_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(ea, em, rtol=2e-5, atol=2e-6), (ea, em)


if __name__ == "__main__":
    pytest.main([__file__])
