from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional, Union

import jax.numpy as jnp
import numpy as np

from . import driver
from .core.system import System, WalkerKind
from .ham.chol import HamChol
from .prop.afqmc import make_prop_ops
from .prop.blocks import block as default_block
from .prop.types import QmcParams
from .staging import StagedInputs, load, stage


def _filter_kwargs_for(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Filter kwargs to only those accepted by callable_obj's signature.
    """
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _make_params(
    *,
    params: Optional[QmcParams] = None,
    n_eql_blocks: Optional[int] = None,
    n_blocks: Optional[int] = None,
    seed: Optional[int] = None,
    dt: Optional[float] = None,
    n_walkers: Optional[int] = None,
    **params_kwargs: Any,
) -> QmcParams:
    base = params or QmcParams()

    if seed is None and params is None:
        seed = int(np.random.randint(0, int(1e9)))

    explicit: dict[str, Any] = {}
    if n_eql_blocks is not None:
        explicit["n_eql_blocks"] = int(n_eql_blocks)
    if n_blocks is not None:
        explicit["n_blocks"] = int(n_blocks)
    if dt is not None:
        explicit["dt"] = float(dt)
    if n_walkers is not None:
        explicit["n_walkers"] = int(n_walkers)
    if seed is not None:
        explicit["seed"] = int(seed)

    merged = dict(params_kwargs)
    merged.update(explicit)

    merged = _filter_kwargs_for(QmcParams, merged)

    return replace(base, **merged)


def _make_prop(
    ham_data: HamChol,
    walker_kind: str,
    *,
    mixed_precision: bool,
) -> Any:
    return make_prop_ops(
        ham_data.basis,
        walker_kind,
        mixed_precision=mixed_precision,
    )


def _make_trial_bundle(sys: System, staged: StagedInputs, mixed_precision: bool) -> tuple[Any, Any, Any]:
    """
    Return (trial_data, trial_ops, meas_ops)
    """
    tr = staged.trial
    data = tr.data

    kind = tr.kind.lower()

    if kind == "rhf":
        from .meas.rhf import make_rhf_meas_ops
        from .trial.rhf import RhfTrial, make_rhf_trial_ops

        mo = jnp.asarray(data["mo"])
        mo_occ = mo[:, : sys.nup]
        trial_data = RhfTrial(mo_occ)
        trial_ops = make_rhf_trial_ops(sys=sys)
        meas_ops = make_rhf_meas_ops(sys=sys)
        return trial_data, trial_ops, meas_ops

    if kind == "uhf":
        from .meas.uhf import make_uhf_meas_ops_chol
        from .trial.uhf import UhfTrial, make_uhf_trial_ops

        if "mo_a" in data and "mo_b" in data:
            mo_a = jnp.asarray(data["mo_a"])[:, : sys.nup]
            mo_b = jnp.asarray(data["mo_b"])[:, : sys.ndn]
        elif "mo" in data:
            mo_a = jnp.asarray(data["mo"])[:, : sys.nup]
            mo_b = jnp.asarray(data["mo"])[:, : sys.ndn]
        trial_data = UhfTrial(mo_a, mo_b)
        trial_ops = make_uhf_trial_ops(sys=sys)
        meas_ops = make_uhf_meas_ops_chol(sys=sys)
        return trial_data, trial_ops, meas_ops

    if kind == "ghf":
        from .meas.ghf import make_ghf_meas_ops_chol
        from .trial.ghf import GhfTrial, make_ghf_trial_ops

        mo = jnp.asarray(data["mo"])
        mo_occ = mo[:, :sys.ne]
        trial_data = GhfTrial(mo_occ)
        trial_ops = make_ghf_trial_ops(sys=sys)
        meas_ops = make_ghf_meas_ops_chol(sys=sys)
        return trial_data, trial_ops, meas_ops

    if kind == "cisd":
        from .meas.cisd import make_cisd_meas_ops
        from .trial.cisd import CisdTrial, make_cisd_trial_ops

        ci1 = jnp.asarray(data["ci1"])
        ci2 = jnp.asarray(data["ci2"])
        trial_data = CisdTrial(ci1, ci2)
        trial_ops = make_cisd_trial_ops(sys=sys)
        meas_ops = make_cisd_meas_ops(sys=sys, mixed_precision=mixed_precision)
        return trial_data, trial_ops, meas_ops

    if kind == "ucisd":
        from .meas.ucisd import make_ucisd_meas_ops
        from .trial.ucisd import UcisdTrial, make_ucisd_trial_ops

        trial_data = UcisdTrial(
            mo_coeff_a=jnp.asarray(data["mo_coeff_a"]),
            mo_coeff_b=jnp.asarray(data["mo_coeff_b"]),
            c1a=jnp.asarray(data["ci1a"]),
            c1b=jnp.asarray(data["ci1b"]),
            c2aa=jnp.asarray(data["ci2aa"]),
            c2ab=jnp.asarray(data["ci2ab"]),
            c2bb=jnp.asarray(data["ci2bb"]),
        )
        trial_ops = make_ucisd_trial_ops(sys=sys)
        meas_ops = make_ucisd_meas_ops(sys=sys, mixed_precision=mixed_precision)
        return trial_data, trial_ops, meas_ops

    if kind == "gcisd":
        from .meas.gcisd import make_gcisd_meas_ops
        from .trial.gcisd import GcisdTrial, make_gcisd_trial_ops

        ci1 = jnp.asarray(data["ci1"])
        ci2 = jnp.asarray(data["ci2"])
        trial_data = GcisdTrial(data["mo_coeff"], ci1, ci2)
        trial_ops = make_gcisd_trial_ops(sys=sys)
        meas_ops = make_gcisd_meas_ops(sys=sys, mixed_precision=mixed_precision)
        return trial_data, trial_ops, meas_ops

    raise ValueError(f"Unsupported TrialInput.kind: {tr.kind!r}")


@dataclass(frozen=True)
class Job:
    """
    A fully assembled AFQMC run bundle.
    """

    staged: StagedInputs
    sys: System
    params: QmcParams
    ham_data: Any
    trial_data: Any
    trial_ops: Any
    meas_ops: Any
    prop_ops: Any
    block_fn: Callable[..., Any]

    def kernel(self, **driver_kwargs: Any):
        """
        Run AFQMC energy driver.
        Extra kwargs are forwarded to driver.run_qmc_energy (e.g. state=..., meas_ctx=...).
        """
        return driver.run_qmc_energy(
            sys=self.sys,
            params=self.params,
            ham_data=self.ham_data,
            trial_ops=self.trial_ops,
            trial_data=self.trial_data,
            meas_ops=self.meas_ops,
            prop_ops=self.prop_ops,
            block_fn=self.block_fn,
            **driver_kwargs,
        )


def setup(
    obj_or_staged: Union[Any, StagedInputs, str, Path],
    *,
    # staging options (used only if we need to stage)
    norb_frozen: Optional[int] = None,
    chol_cut: float = 1e-5,
    cache: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    verbose: bool = False,
    # system/prop options
    walker_kind: Optional[WalkerKind] = None,
    mixed_precision: bool = True,
    # params options
    params: Optional[QmcParams] = None,
    # overrides for customized runs
    trial_data: Any = None,
    trial_ops: Any = None,
    meas_ops: Any = None,
    prop_ops: Any = None,
    block_fn: Optional[Callable[..., Any]] = None,
    # extra kwargs
    params_kwargs: Optional[dict[str, Any]] = None,
    prop_kwargs: Optional[dict[str, Any]] = None,
) -> Job:
    """
    Assemble a runnable AFQMC Job from either:
      - a pyscf mf/cc object,
      - StagedInputs,
      - or a path to a staged .h5 cache file.

    Basic usage:
        job = setup(mf)
        job.kernel()

    Advanced usage:
        staged = stage(cc, cache="afqmc.h5")
        job = setup(staged, walker_kind="restricted", mixed_precision=False, params=myparams)
        job.kernel()
    """
    staged: StagedInputs
    if isinstance(obj_or_staged, StagedInputs):
        staged = obj_or_staged
    else:
        p = (
            Path(obj_or_staged).expanduser().resolve()
            if isinstance(obj_or_staged, (str, Path))
            else None
        )
        if p is not None and p.exists():
            staged = load(p)
        else:
            staged = stage(
                obj_or_staged,
                norb_frozen=norb_frozen,
                chol_cut=chol_cut,
                cache=cache,
                overwrite=overwrite,
                verbose=verbose,
            )

    ham = staged.ham

    match walker_kind, ham.basis, ham.nelec[0] == ham.nelec[1]:
        case None, "restricted", True:
            walker_kind = "restricted"
        case None, "restricted", False:
            walker_kind = "unrestricted"
        case None, "generalized", _:
            walker_kind = "generalized"

    sys = System(norb=int(ham.norb), nelec=ham.nelec, walker_kind=walker_kind)

    ham_data = HamChol(
        jnp.asarray(ham.h0),
        jnp.asarray(ham.h1),
        jnp.asarray(ham.chol),
        basis=ham.basis
    )

    if params_kwargs is None:
        params_kwargs = {}
    qmc_params = _make_params(
        params=params,
        **params_kwargs,
    )

    if trial_data is None or trial_ops is None or meas_ops is None:
        td, to, mo = _make_trial_bundle(sys, staged, mixed_precision)
        trial_data = td if trial_data is None else trial_data
        trial_ops = to if trial_ops is None else trial_ops
        meas_ops = mo if meas_ops is None else meas_ops

    if prop_ops is None:
        if prop_kwargs is None:
            prop_kwargs = {}
        prop_ops = _make_prop(
            ham_data,
            sys.walker_kind,
            mixed_precision=mixed_precision,
            **prop_kwargs,
        )

    if block_fn is None:
        block_fn = default_block

    return Job(
        staged=staged,
        sys=sys,
        params=qmc_params,
        ham_data=ham_data,
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
        block_fn=block_fn,
    )
