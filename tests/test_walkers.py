from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ad_afqmc_prototype import walkers as wk


def _rand_complex(key, shape, dtype=jnp.complex64):
    k1, k2 = jax.random.split(key)
    re = jax.random.normal(k1, shape)
    im = jax.random.normal(k2, shape)
    return (re + 1j * im).astype(dtype)


def test_n_walkers_array_and_tuple():
    key = jax.random.PRNGKey(0)
    w = _rand_complex(key, (8, 6, 3))
    assert wk.n_walkers(w) == 8

    key1, key2 = jax.random.split(key)
    wu = _rand_complex(key1, (10, 6, 3))
    wd = _rand_complex(key2, (10, 6, 2))
    assert wk.n_walkers((wu, wd)) == 10


@pytest.mark.parametrize("n_chunks", [1, 3, 4])
def test_apply_chunked_matches_vmap_array(n_chunks: int):
    key = jax.random.PRNGKey(1)
    nwalk, norb, nocc = 8, 6, 3

    w = _rand_complex(key, (nwalk, norb, nocc))

    def fn_one(walker, alpha):
        return alpha * jnp.sum(jnp.abs(walker) ** 2)

    alpha = 0.3
    out_chunked = wk.vmap_chunked(fn_one, n_chunks, in_axes=(0, None))(w, alpha)
    out_vmap = jax.vmap(lambda x: fn_one(x, alpha))(w)

    assert out_chunked.shape == out_vmap.shape
    assert jnp.allclose(out_chunked, out_vmap, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n_chunks", [1, 3, 4])
def test_apply_chunked_matches_vmap_unrestricted(n_chunks: int):
    key = jax.random.PRNGKey(2)
    nwalk, norb, nocc_u, nocc_d = 8, 6, 3, 2

    k1, k2 = jax.random.split(key)
    wu = _rand_complex(k1, (nwalk, norb, nocc_u))
    wd = _rand_complex(k2, (nwalk, norb, nocc_d))

    def fn_one(w_i, beta):
        wu_i, wd_i = w_i
        return beta * (jnp.sum(jnp.abs(wu_i) ** 2) + 2.0 * jnp.sum(jnp.abs(wd_i) ** 2))

    beta = -0.7
    out_chunked = wk.vmap_chunked(fn_one, n_chunks, in_axes=(0, None))((wu, wd), beta)
    out_vmap = jax.vmap(lambda a, b: fn_one((a, b), beta))(wu, wd)

    assert out_chunked.shape == out_vmap.shape
    assert jnp.allclose(out_chunked, out_vmap, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n_chunks", [1, 3, 4])
def test_apply_chunked_prop_matches_vmap_array(n_chunks: int):
    key = jax.random.PRNGKey(3)
    nwalk, norb, nocc, nfields = 8, 6, 3, 5

    k1, k2, k3 = jax.random.split(key, 3)
    w = _rand_complex(k1, (nwalk, norb, nocc))
    fields = jax.random.normal(k2, (nwalk, nfields))
    mat = _rand_complex(k3, (norb, norb))

    def prop_one(walker, field_i, mat):
        return walker + (0.1 * field_i[0]) * (mat @ walker)

    out_chunked = wk.vmap_chunked(prop_one, n_chunks, in_axes=(0, 0, None))(w, fields, mat)
    out_vmap = jax.vmap(lambda wi, fi: prop_one(wi, fi, mat))(w, fields)

    assert out_chunked.shape == out_vmap.shape
    assert jnp.allclose(out_chunked, out_vmap, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n_chunks", [1, 3, 4])
def test_apply_chunked_prop_matches_vmap_unrestricted(n_chunks: int):
    key = jax.random.PRNGKey(4)
    nwalk, norb, nocc_u, nocc_d, nfields = 8, 6, 3, 2, 4

    k1, k2, k3, k4 = jax.random.split(key, 4)
    wu = _rand_complex(k1, (nwalk, norb, nocc_u))
    wd = _rand_complex(k2, (nwalk, norb, nocc_d))
    fields = jax.random.normal(k3, (nwalk, nfields))
    mat = _rand_complex(k4, (norb, norb))

    def prop_one(w_i, field_i, mat):
        wu_i, wd_i = w_i
        s = 0.05 * field_i[1]
        return (wu_i + s * (mat @ wu_i), wd_i - s * (mat @ wd_i))

    out_chunked = wk.vmap_chunked(prop_one, n_chunks, in_axes=(0, 0, None))((wu, wd), fields, mat)
    out_vmap_u, out_vmap_d = jax.vmap(lambda a, b, f: prop_one((a, b), f, mat), in_axes=(0, 0, 0))(
        wu, wd, fields
    )

    assert isinstance(out_chunked, tuple) and len(out_chunked) == 2
    assert out_chunked[0].shape == out_vmap_u.shape
    assert out_chunked[1].shape == out_vmap_d.shape
    assert jnp.allclose(out_chunked[0], out_vmap_u, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(out_chunked[1], out_vmap_d, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
