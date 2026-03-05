import jax
import jax.numpy as jnp

from ad_afqmc_prototype import driver
from ad_afqmc_prototype.core.ops import TrialOps
from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.ham.chol import HamBasis, HamChol
from ad_afqmc_prototype.ham.hubbard import HamHubbard
from ad_afqmc_prototype.ham.hubbard_nn import HamHubbardNN
from ad_afqmc_prototype.trial.uhf import UhfTrial
from ad_afqmc_prototype.trial.ghf import GhfTrial
from ad_afqmc_prototype.meas.auto import make_auto_meas_ops
from ad_afqmc_prototype.prop.afqmc import make_prop_ops
from ad_afqmc_prototype.staging import StagedMfOrCc, _stage_ham_input


def rand_orthonormal_cols(key, nrow, ncol, dtype=jnp.complex128):
    """
    Random (nrow, ncol) matrix with orthonormal columns via QR.
    """
    k1, k2 = jax.random.split(key)

    if dtype in (jnp.complex128, jnp.complex64):
        a = jax.random.normal(k1, (nrow, ncol), dtype=jnp.float64) + 1.0j * jax.random.normal(
            k2, (nrow, ncol), dtype=jnp.float64
        )
    elif dtype in (jnp.float64, jnp.float32):
        a = jax.random.normal(k1, (nrow, ncol), dtype=jnp.float64)
    else:
        raise TypeError(f"Received unsupported type {dtype}.")

    q, _ = jnp.linalg.qr(a, mode="reduced")
    return q.astype(dtype)


def make_random_uhf_trial(key, norb, nup, ndn, dtype=jnp.complex128) -> UhfTrial:
    ka, kb = jax.random.split(key)
    ca = rand_orthonormal_cols(ka, norb, nup, dtype=dtype)
    cb = rand_orthonormal_cols(kb, norb, ndn, dtype=dtype)
    return UhfTrial(mo_coeff_a=ca, mo_coeff_b=cb)


def make_random_ghf_trial(key, norb, nup, ndn, dtype=jnp.complex128) -> GhfTrial:
    ne = nup + ndn
    c = rand_orthonormal_cols(key, 2 * norb, ne, dtype=dtype)
    return GhfTrial(mo_coeff=c)


def make_random_ham_chol(
    key, norb, n_chol, basis: HamBasis = "restricted", dtype=jnp.float64
) -> HamChol:
    """
    Build a small HamChol with:
      - symmetric real h1
      - symmetric real chol[g]
    """
    assert basis in ["restricted", "generalized"]

    if basis == "generalized":
        norb = 2 * norb

    k1, k2, k3 = jax.random.split(key, 3)

    a = jax.random.normal(k1, (norb, norb), dtype=dtype)
    h1 = 0.5 * (a + a.T)

    b = jax.random.normal(k2, (n_chol, norb, norb), dtype=dtype)
    chol = 0.5 * (b + jnp.swapaxes(b, 1, 2))

    h0 = jax.random.normal(k3, (), dtype=dtype)

    return HamChol(basis=basis, h0=h0, h1=h1, chol=chol)


def make_random_ham_hubbard(
    key, norb, dtype=jnp.float64
) -> HamHubbard:
    """
    Build a small HamHubbard with:
      - symmetric real h1
    """
    k1, k2 = jax.random.split(key, 2)
    a = jax.random.normal(k1, (norb, norb), dtype=dtype)
    h1 = 0.5 * (a + a.T)
    u = 100.0 * jax.random.uniform(k2, (), dtype=dtype)
    return HamHubbard(h1=h1, u=u)


def make_random_ham_hubbard_nn(
    key, norb, n_bonds, dtype=jnp.float64
) -> HamHubbardNN:
    """
    Build a small HamHubbardNN with:
      - symmetric real h1
    """
    def random_unique_bonds(key, norb: int, n_bonds: int):
        # number of possible undirected bonds
        n_pairs = norb * (norb - 1) // 2
        n_bonds = min(n_bonds, n_pairs)

        # sample unique indices in [0, n_pairs)
        idx = jax.random.choice(key, n_pairs, shape=(n_bonds,), replace=False)

        # map each index -> (i, j) with i < j
        # build lookup table once (OK for moderate norb)
        i, j = jnp.triu_indices(norb, k=1)   # each shape (n_pairs,)
        bonds = jnp.stack([i[idx], j[idx]], axis=1)  # (n_bondss, 2), i<j
        return bonds.astype(jnp.int32)

    k1, k2, k3, k4 = jax.random.split(key, 4)
    a = jax.random.normal(k1, (norb, norb), dtype=dtype)
    h1 = 0.5 * (a + a.T)
    u = 100.0 * jax.random.uniform(k2, (), dtype=dtype)
    v = u * jax.random.uniform(k3, (), dtype=dtype)
    bonds = random_unique_bonds(k4, norb=norb, n_bonds=n_bonds)
    return HamHubbardNN(h1=h1, u=u, v=v, bonds=bonds)


def make_walkers(key, sys: System, nw: int=1, dtype=jnp.complex128):
    """
    Build `nw` random walkers that can be either
    - restricted (nw, norb, nocc)
    - unrestricted ((nw, norb, na), (nw, norb, nb))
    - generalized (nw, 2*norb, na+nb)
    """
    norb, na, nb = sys.norb, sys.nup, sys.ndn
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        w = jnp.stack([
            rand_orthonormal_cols(
                jax.random.fold_in(key, i), norb, na, dtype=dtype
            ) for i in range(nw)
        ])
        if nw == 1: return w[0]
        return w

    elif wk == "unrestricted":
        k1, k2 = jax.random.split(key)
        wu = jnp.stack([
            rand_orthonormal_cols(
                jax.random.fold_in(k1, i), norb, na, dtype=dtype
            ) for i in range(nw)
        ])
        wd = jnp.stack([
            rand_orthonormal_cols(
                jax.random.fold_in(k2, i), norb, nb, dtype=dtype
            ) for i in range(nw)
        ])
        if nw == 1: return (wu[0], wd[0])
        return (wu, wd)

    elif wk == "generalized":
        w = jnp.stack([
            rand_orthonormal_cols(
                jax.random.fold_in(key, i), 2*norb, na+nb, dtype=dtype
            ) for i in range(nw)
        ])
        if nw == 1: return w[0]
        return w

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")


def make_restricted_walker_near_ref(
    key, norb: int, nocc: int, *, mix: float = 0.2, dtype=jnp.complex128
) -> jax.Array:
    """
    Make a restricted walker (norb, nocc) whose occupied block isn't near-singular.

    Start from the reference [I;0] and add a small random perturbation, then QR.
    This avoids tiny det(w[:nocc,:]) which can make overlap-based finite differences noisy.
    """
    k1, k2 = jax.random.split(key)
    w0 = jnp.zeros((norb, nocc), dtype=jnp.complex128)
    w0 = w0.at[:nocc, :].set(jnp.eye(nocc, dtype=jnp.complex128))
    noise = jax.random.normal(k1, (norb, nocc), dtype=jnp.float64) + 1.0j * jax.random.normal(
        k2, (norb, nocc), dtype=jnp.float64
    )
    w = w0 + mix * noise
    q, _ = jnp.linalg.qr(w, mode="reduced")
    return q.astype(dtype)


def make_dummy_trial_ops():
    def get_rdm1(trial_data):
        return trial_data["rdm1"]

    def overlap(walker, trial_data):
        return jnp.asarray(1.0 + 0.0j)

    return TrialOps(overlap=overlap, get_rdm1=get_rdm1)


def make_common_auto(
    key,
    walker_kind,
    norb: int,
    nelec: tuple[int, int],
    n_chol: int,
    *,
    make_trial_fn,
    make_trial_fn_kwargs=(),
    make_trial_ops_fn,
    make_meas_ops_fn,
    ham_basis: HamBasis = "restricted",
):
    sys = System(norb=norb, nelec=nelec, walker_kind=walker_kind)

    k_ham, k_trial = jax.random.split(key, 2)

    ham = make_random_ham_chol(k_ham, norb=norb, n_chol=n_chol, basis=ham_basis)
    trial = make_trial_fn(k_trial, **make_trial_fn_kwargs)

    t_ops = make_trial_ops_fn(sys)
    meas_manual = make_meas_ops_fn(sys)
    meas_auto = make_auto_meas_ops(sys, t_ops, eps=1.0e-4)

    ctx_manual = meas_manual.build_meas_ctx(ham, trial)
    ctx_auto = meas_auto.build_meas_ctx(ham, trial)

    return sys, ham, trial, meas_manual, ctx_manual, meas_auto, ctx_auto


def make_common_manual_only(
    key,
    walker_kind,
    norb: int,
    nelec: tuple[int, int],
    n_chol: int,
    *,
    make_trial_fn,
    make_trial_fn_kwargs=(),
    make_trial_ops_fn,
    build_meas_ctx_fn,
):
    sys = System(norb=norb, nelec=nelec, walker_kind=walker_kind)

    k_ham, k_trial = jax.random.split(key, 2)

    ham = make_random_ham_chol(k_ham, norb=norb, n_chol=n_chol)
    trial = make_trial_fn(k_trial, **make_trial_fn_kwargs)
    ctx = build_meas_ctx_fn(ham, trial)

    return sys, ham, trial, ctx


def make_common_hubbard(
    key,
    walker_kind,
    norb: int,
    nelec: tuple[int, int],
    *,
    make_trial_fn,
    make_trial_fn_kwargs=(),
):
    sys = System(norb=norb, nelec=nelec, walker_kind=walker_kind)

    k_ham, k_trial = jax.random.split(key, 2)

    ham = make_random_ham_hubbard(k_ham, norb=norb)
    trial = make_trial_fn(k_trial, **make_trial_fn_kwargs)

    return sys, ham, trial


def make_common_hubbard_nn(
    key,
    walker_kind,
    norb: int,
    n_bonds: int,
    nelec: tuple[int, int],
    *,
    make_trial_fn,
    make_trial_fn_kwargs=(),
):
    sys = System(norb=norb, nelec=nelec, walker_kind=walker_kind)

    k_ham, k_trial = jax.random.split(key, 2)

    ham = make_random_ham_hubbard_nn(k_ham, norb=norb, n_bonds=n_bonds)
    trial = make_trial_fn(k_trial, **make_trial_fn_kwargs)

    return sys, ham, trial


def run_calc(
    sys, 
    meas_ops, 
    ham_data, 
    trial_ops, 
    trial_data, 
    params, 
    block_fn, 
    prop_ops,
    state=None,
):
    mean, err, block_e_all, block_w_all = driver.run_qmc_energy(
        sys=sys,
        params=params,
        ham_data=ham_data,
        trial_ops=trial_ops,
        trial_data=trial_data,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
        block_fn=block_fn,
        state=state,
    )
    return mean, err, block_e_all, block_w_all


def make_common_pyscf(
    mf,
    make_meas_ops_fn,
    make_trial_ops_fn,
    walker_kind,
    ham_basis: HamBasis = "restricted",
):
    obj = StagedMfOrCc(mf, norb_frozen=0)
    ham_input = _stage_ham_input(obj, chol_cut=1e-6, verbose=False)
    h0 = jnp.asarray(ham_input.h0)
    h1 = jnp.asarray(ham_input.h1)
    chol = jnp.asarray(ham_input.chol)
    sys = System(
        norb=mf.mol.nao,
        nelec=mf.mol.nelec,
        walker_kind=walker_kind,
    )
    meas_ops = make_meas_ops_fn(sys)
    ham_data = HamChol(h0, h1, chol, basis=ham_basis)
    prop_ops = make_prop_ops(ham_data.basis, sys.walker_kind)
    trial_ops = make_trial_ops_fn(sys=sys)

    return sys, ham_data, trial_ops, prop_ops, meas_ops
