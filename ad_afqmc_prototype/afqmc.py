from __future__ import annotations

from .config import configure_once

configure_once()

import dataclasses
from functools import partial
from pathlib import Path
from typing import Any, Callable, Union, cast

print = partial(print, flush=True)

from jax.sharding import Mesh

from .core.system import WalkerKind
from .prop.types import QmcParams, QmcParamsBase, QmcParamsFp
from .runtime_provenance import print_runtime_provenance
from .setup import Job
from .setup import setup as setup_job
from .setup_fp import JobFp
from .setup_fp import setup_fp as setup_job_fp
from .staging import StagedInputs, _is_cc_like
from .staging import dump as dump_staged
from .staging import load as load_staged
from .staging import stage as stage_inputs


def banner_afqmc() -> str:
    return r"""
 █████╗ ██████╗        █████╗ ███████╗ ██████╗ ███╗   ███╗ ██████╗
██╔══██╗██╔══██╗      ██╔══██╗██╔════╝██╔═══██╗████╗ ████║██╔════╝
███████║██║  ██║█████╗███████║█████╗  ██║   ██║██╔████╔██║██║
██╔══██║██║  ██║╚════╝██╔══██║██╔══╝  ██║▄▄ ██║██║╚██╔╝██║██║
██║  ██║██████╔╝      ██║  ██║██║     ╚██████╔╝██║ ╚═╝ ██║╚██████╗
╚═╝  ╚═╝╚═════╝       ╚═╝  ╚═╝╚═╝      ╚══▀▀═╝ ╚═╝     ╚═╝ ╚═════╝
     differentiable auxiliary-field quantum Monte Carlo
"""


class Afqmc:
    """
    AFQMC driver object.

    Parameters
    ----------
    mf_or_cc : Any
        Mean-field or coupled-cluster object from which to build Hamiltonian and trial wavefunction.
    norb_frozen : int, optional
        Number of lowest occupied orbitals removed from the AFQMC Hamiltonian. For CC objects with integer cc.frozen, norb_frozen must match this attribute. For restricted CCSD objects with list-valued cc.frozen, the trial space frozen occupied/virtual blocks are inferred from cc.frozen and norb_frozen controls the orbitals frozen in AFQMC.
    chol_cut : float, optional
        Cholesky decomposition cutoff, by default 1e-5
    cache : Union[str, Path], optional
        Path to cache file for staged inputs, by default None
    n_eql_blocks : int, optional
        Number of equilibration blocks if params is not provided, by default 20
    n_blocks : int, optional
        Number of production blocks if params is not provided, by default 200
    seed : int | None, optional
        Random seed if params is not provided, by default None
    dt : float | None, optional
        Time step if params is not provided, by default None
    n_walkers : int | None, optional
        Number of walkers if params is not provided, by default None
    n_chunk : int | None, optional
        Number of chunks if params is not provided, by default 1
    """

    params_cls = QmcParams
    job_cls = Job
    setup_fn = staticmethod(setup_job)

    def __init__(
        self,
        mf_or_cc: Any,
        *,
        norb_frozen: int | None = None,
        chol_cut: float = 1e-5,
        cache: Union[str, Path] | None = None,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int | None = None,
    ):
        self._obj = mf_or_cc
        self._cc: Any = None
        if _is_cc_like(mf_or_cc):
            self._cc = mf_or_cc
            self._scf = mf_or_cc._scf
            self.source_kind = "cc"
        else:
            self._scf = mf_or_cc
            self.source_kind = "mf"

        self.norb_frozen = norb_frozen
        self.chol_cut = float(chol_cut)
        self.cache = Path(cache).expanduser().resolve() if cache is not None else None
        self.overwrite_cache = False
        self.verbose = False

        self.walker_kind: WalkerKind | None = None  # resolved in kernel
        self.mixed_precision = True

        self.params: QmcParamsBase | None = None  # resolved in kernel
        defaults = self.params_cls()
        self.dt = defaults.dt if dt is None else dt
        self.n_walkers = defaults.n_walkers if n_walkers is None else n_walkers
        self.n_blocks = defaults.n_blocks if n_blocks is None else n_blocks
        self.seed = defaults.seed if seed is None else seed
        self.n_chunks = defaults.n_chunks if n_chunks is None else n_chunks
        if hasattr(defaults, "n_eql_blocks"):
            self.n_eql_blocks = defaults.n_eql_blocks if n_eql_blocks is None else n_eql_blocks

        self._staged: StagedInputs | None = None
        self._job: Job | None = None
        self._cache_key: tuple | None = None

        self.e_tot: Any = None
        self.e_err: Any = None
        self.block_energies: Any = None
        self.block_weights: Any = None

    @property
    def staged(self) -> StagedInputs | None:
        return self._staged

    @property
    def job(self) -> Job | None:
        return self._job

    def _dump_params(self, params: QmcParamsBase) -> None:
        fields = dataclasses.fields(params)
        width = len(max(fields, key=lambda f: len(f.name)).name)
        print(f" {type(params).__name__}:")
        for field in fields:
            print(f"  {field.name:<{width}} = {getattr(params, field.name)}")
        print("")

    def _resolve_meas_cfg(self, job: Job) -> object | None:
        from .meas.cisd import get_cisd_meas_cfg
        from .meas.rhf import get_rhf_meas_cfg

        for getter in (get_cisd_meas_cfg, get_rhf_meas_cfg):
            cfg = getter(job.meas_ops)
            if cfg is not None:
                return cfg
        return None

    def _dump_cfg(self, name: str, cfg: object) -> None:
        if not dataclasses.is_dataclass(cfg):
            print(f" {name:<15} = {cfg}")
            return

        print(f" {name:<15} = {type(cfg).__name__}")
        fields = dataclasses.fields(cfg)
        width = len(max(fields, key=lambda f: len(f.name)).name)
        for field in fields:
            value = getattr(cfg, field.name)
            if isinstance(value, type):
                value_str = value.__name__
            else:
                value_str = str(value)
            print(f"  {field.name:<{width}} = {value_str}")

    def dump_flags(self, job: Job) -> None:
        self._dump_flags_helper(job)

    def _dump_flags_helper(self, job: Job) -> None:
        meta = job.staged.meta
        src = meta["source_kind"]
        chol_cut = meta["chol_cut"]
        sys = job.sys
        nchol = job.ham_data.nchol
        params = job.params
        trial = job.staged.trial
        print("\n******** AFQMC ********")
        print(f" norb            = {sys.norb}")
        print(f" nelec_up        = {sys.nelec[0]}")
        print(f" nelec_dn        = {sys.nelec[1]}")
        print(f" nchol           = {nchol}")
        print(f" source_kind     = {src}")
        print(f" trial_kind      = {trial.kind}")
        print(f" chol_cut        = {chol_cut:g}")
        print(f" cache           = {str(self.cache) if self.cache else None}")
        print(f" walker_kind     = {sys.walker_kind}")
        print(f" mixed_precision = {self.mixed_precision}\n")
        meas_cfg = self._resolve_meas_cfg(job)
        if meas_cfg is not None:
            self._dump_cfg("meas_cfg", meas_cfg)
            print("")
        self._dump_params(params)

    def _key(self) -> tuple:
        """Key for determining whether staged/job caches are still valid."""
        cache_mtime = None
        if self.cache is not None and self.cache.exists():
            cache_mtime = self.cache.stat().st_mtime
        return (
            self.source_kind,
            self.norb_frozen,
            float(self.chol_cut),
            str(self.cache) if self.cache is not None else None,
            bool(self.overwrite_cache),
            cache_mtime,
        )

    def stage(self, *, force: bool = False) -> StagedInputs:
        """
        Compute or load HamInput/TrialInput.
        If cache is set and exists, loads unless overwrite_cache=True.
        """
        key = self._key()
        if self._staged is not None and self._cache_key == key and not force:
            return self._staged

        staged = stage_inputs(
            self._obj,
            norb_frozen=self.norb_frozen if self.norb_frozen is not None else None,
            chol_cut=self.chol_cut,
            cache=self.cache,
            overwrite=self.overwrite_cache if self.cache is not None else False,
            verbose=self.verbose,
        )
        self._staged = staged
        self._cache_key = key
        self._job = None
        return staged

    def save_staged(self, path: Union[str, Path]) -> None:
        """Write current staged inputs to a single file cache."""
        staged = self.stage()
        dump_staged(staged, path)

    # def load_staged(self, path: Union[str, Path]): -> StagedInputs:
    #    """Load staged inputs from a cache file and attach them to this object."""
    #    staged = load_staged(path)
    #    self._staged = staged
    #    self._cache_key = None
    #    self._job = None
    #    return staged

    def _validate_params(self, params: QmcParamsBase) -> QmcParamsBase:
        return params

    def _make_params(self) -> QmcParamsBase:
        """
        Create QmcParams if user didn't provide one.
        """
        params_cls = self.params_cls

        if self.params is not None and isinstance(self.params, params_cls):
            params = self.params
        elif self.params is not None and not isinstance(self.params, params_cls):
            raise TypeError(
                f"Expected type {params_cls.__name__} for self.params, but received '{type(self.params)}'"
            )
        else:
            kwargs: dict[str, Any] = {}
            for field in dataclasses.fields(params_cls):
                if hasattr(self, field.name):
                    val = getattr(self, field.name)
                    if val is not None:
                        kwargs[field.name] = val

            params = params_cls(**kwargs)

        return self._validate_params(params)

    def build_job(
        self,
        *,
        force: bool = False,
        trial_data: Any = None,
        trial_ops: Any = None,
        meas_ops: Any = None,
        prop_ops: Any = None,
        block_fn: Callable[..., Any] | None = None,
        prop_kwargs: dict[str, Any] | None = None,
        mesh: Mesh | None = None,
    ) -> Job:
        """
        Assemble a runnable Job from current settings and staged inputs.
        """
        if self._job is not None and not force and (mesh is None or self._job.mesh is mesh):
            return self._job

        staged = self.stage()
        qmc_params = self._make_params()
        self.params = qmc_params

        job = self.setup_fn(
            staged,
            walker_kind=self.walker_kind,
            mesh=mesh,
            mixed_precision=self.mixed_precision,
            params=cast(Any, qmc_params),
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            block_fn=block_fn,
            prop_kwargs=prop_kwargs,
        )
        self._job = job
        return job

    def _coerce_result(self, value: Any) -> Any:
        return float(value)

    def kernel(self, **driver_kwargs: Any) -> tuple[Any, Any]:
        """
        Runs AFQMC, returns (e_tot, e_err), and stores samples.
        """
        print(banner_afqmc())
        print_runtime_provenance()
        mesh = driver_kwargs.get("mesh")
        job = self.build_job(mesh=mesh)
        self.dump_flags(job)

        out = job.kernel(**driver_kwargs)

        if isinstance(out, tuple) and len(out) >= 2:
            e_tot = self._coerce_result(out[0])
            e_err = self._coerce_result(out[1])
            block_e = out[2] if len(out) > 2 else None
            block_w = out[3] if len(out) > 3 else None
        else:
            raise TypeError("Unexpected return from Job.kernel(), expected tuple output.")

        self.e_tot = e_tot
        self.e_err = e_err
        self.block_energies = block_e
        self.block_weights = block_w
        return e_tot, e_err

    run = kernel

    @classmethod
    def _from_staged_common(cls, path: Union[str, Path], **kwargs: Any):
        staged = load_staged(path)
        meta = staged.meta

        af = cls(
            None,
            norb_frozen=meta["norb_frozen"],
            chol_cut=meta["chol_cut"],
            **kwargs,
        )
        af._staged = staged
        af.source_kind = meta["source_kind"]
        af._cache_key = af._key()
        return af

    @classmethod
    def from_staged(
        cls,
        path: Union[str, Path],
        *,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int = 1,
    ) -> Afqmc:
        """
        Returns a new AFQMC object from a previously staged calculations
        (using save_staged method). The number of frozen orbitals, norb_frozen,
        and the choliesky decomposition threshold, chol_cut, cannot be changed.
        Parameters
        ----------
        path: str, pathlib.Path
        The other parameters are identical to the ones in the AFQMC class.
        """
        return cls._from_staged_common(
            path,
            n_eql_blocks=n_eql_blocks,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )


class AfqmcFp(Afqmc):
    params_cls = QmcParamsFp
    job_cls = JobFp
    setup_fn = staticmethod(setup_job_fp)

    def __init__(
        self,
        mf_or_cc: Any,
        *,
        norb_frozen: int | None = None,
        chol_cut: float = 1e-5,
        cache: Union[str, Path] | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_prop_steps: int | None = None,
        n_walkers: int | None = None,
        n_chunks: int = 1,
        ene0: float | None = None,
        n_traj: int | None = None,
    ):
        super().__init__(
            mf_or_cc,
            norb_frozen=norb_frozen,
            chol_cut=chol_cut,
            cache=cache,
            n_eql_blocks=None,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )
        defaults = self.params_cls()
        self.n_prop_steps = defaults.n_prop_steps if n_prop_steps is None else n_prop_steps
        self.n_traj = defaults.n_traj if n_traj is None else n_traj
        self.ene0 = ene0

    def _validate_params(self, params: QmcParamsBase) -> QmcParamsBase:
        assert isinstance(params, QmcParamsFp)
        if params.ene0 is None:
            raise ValueError(
                "The value of the parameter 'ene0' must be set, typically with SCF or CC energy."
            )
        return params

    def _coerce_result(self, value: Any) -> Any:
        return value

    run_fp = Afqmc.kernel

    @classmethod
    def from_staged(
        cls,
        path: Union[str, Path],
        *,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_prop_steps: int | None = None,
        n_walkers: int | None = None,
        n_chunks: int = 1,
    ) -> AfqmcFp:
        """
        Returns a new AFQMC object from a previously staged calculations
        (using save_staged method). The number of frozen orbitals, norb_frozen,
        and the choliesky decomposition threshold, chol_cut, cannot be changed.
        Parameters
        ----------
        path: str, pathlib.Path
        The other parameters are identical to the ones in the AFQMC class.
        """
        return cls._from_staged_common(
            path,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_prop_steps=n_prop_steps,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )


# Backward-compatible aliases
AFQMC = Afqmc
AFQMCFp = AfqmcFp


class AfqmcPt2Ccsd(Afqmc):
    def __init__(
        self,
        mf_or_cc: Any,
        *,
        norb_frozen: int | None = None,
        chol_cut: float = 1e-5,
        cache: Union[str, Path] | None = None,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int | None = None,
    ):

        from pyscf.cc.ccsd import CCSD

        assert isinstance(mf_or_cc, CCSD)

        super().__init__(
            mf_or_cc,
            norb_frozen=norb_frozen,
            chol_cut=chol_cut,
            cache=cache,
            n_eql_blocks=n_eql_blocks,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )

    def kernel(self):
        from ad_afqmc_prototype.core.system import System
        from ad_afqmc_prototype.driver import run_mixed_qmc
        from ad_afqmc_prototype.ham.chol import HamChol
        from ad_afqmc_prototype.meas.pt2ccsd import make_pt2ccsd_meas_ops
        from ad_afqmc_prototype.meas.rhf import make_rhf_meas_ops
        from ad_afqmc_prototype.prop.afqmc import make_prop_ops
        from ad_afqmc_prototype.prop.blocks import block_mixed
        from ad_afqmc_prototype.prop.types import QmcParams
        from ad_afqmc_prototype.staging import StagedMfOrCc, _stage_pt2ccsd_input, stage
        from ad_afqmc_prototype.trial.pt2ccsd import Pt2ccsdTrial
        from ad_afqmc_prototype.trial.rhf import RhfTrial, make_rhf_trial_ops

        import jax.numpy as jnp

        staged_guide = stage(self._scf)
        ham = staged_guide.ham
        sys = System(norb=int(ham.norb), nelec=ham.nelec, walker_kind="restricted")

        ham_data = HamChol(
            jnp.array(ham.h0),
            jnp.array(ham.h1),
            jnp.array(ham.chol),
            basis=ham.basis,
        )

        guide_data = RhfTrial(
            mo_coeff=jnp.array(staged_guide.trial.data["mo"][:, : sys.nup]),
        )

        trial_obj = StagedMfOrCc(self._cc, norb_frozen=None)
        staged_trial = _stage_pt2ccsd_input(trial_obj)

        trial_data = Pt2ccsdTrial(
            mo_t=jnp.asarray(staged_trial.data["mo_t"]),
            t2=jnp.asarray(staged_trial.data["t2"]),
        )

        guide_ops = make_rhf_trial_ops(sys)
        guide_meas_ops = make_rhf_meas_ops(sys)
        guide_prop_ops = make_prop_ops(ham_data.basis, sys.walker_kind)
        trial_meas_ops = make_pt2ccsd_meas_ops(sys, mixed_precision=False)

        params = self._make_params()
        assert isinstance(params, QmcParams)

        print(f"\ndt={params.dt}  n_walkers={params.n_walkers}  n_prop_steps={params.n_prop_steps}")
        print(
            f"Equilibration imaginary time : {params.n_eql_blocks * params.n_prop_steps * params.dt:.2f} a.u."
        )
        print(
            f"Sampling imaginary time      : {params.n_blocks    * params.n_prop_steps * params.dt:.2f} a.u.\n"
        )

        mixed_samples = run_mixed_qmc(
            sys=sys,
            params=params,
            ham_data=ham_data,
            guide_data=guide_data,
            guide_ops=guide_ops,
            guide_prop_ops=guide_prop_ops,
            guide_meas_ops=guide_meas_ops,
            trial_data=trial_data,
            trial_meas_ops=trial_meas_ops,
            mix_block_fn=block_mixed,
        )

        print(
            f"\nGuide  (AFQMC/RHF)    : {mixed_samples.guide_mean_energy.real:.6f} +/- {mixed_samples.guide_stderr_energy.real:.6f} Ha"
        )
        print(
            f"Trial  (AFQMC/pt2CCSD): {mixed_samples.trial_mean_energy.real:.6f} +/- {mixed_samples.trial_stderr_energy.real:.6f} Ha"
        )
        print(f"Reference (CCSD)       : {self._cc.e_tot:.6f} Ha")
