from __future__ import annotations

from .config import configure_once

configure_once()

from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from .core.system import WalkerKind
from .prop.types import QmcParams
from .setup import Job, _filter_kwargs_for
from .setup import setup as setup_job
from .staging import StagedInputs, _is_cc_like
from .staging import dump as dump_staged
from .staging import load as load_staged
from .staging import stage as stage_inputs


def _default_seed() -> int:
    return int(np.random.randint(0, int(1e9)))


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


class AFQMC:
    """
    AFQMC driver object.

    Parameters
    ----------
    mf_or_cc : Any
        Mean-field or coupled-cluster object from which to build Hamiltonian and trial wavefunction.
    norb_frozen : int, optional
        Number of orbitals to freeze (from the bottom), by default 0 or cc.frozen
        if mf_or_cc is a Pyscf SCF or CC instance, respectively. For CC instances,
        norb_frozen cannot be set to a value differing fron cc.frozen.
    chol_cut : float, optional
        Cholesky decomposition cutoff, by default 1e-5
    cache : Union[str, Path], optional
        Path to cache file for staged inputs, by default None
    n_eql_blocks : int, optional
        Number of equilibration blocks if params is not provided, by default 20
    n_blocks : int, optional
        Number of production blocks if params is not provided, by default 200
    seed : Optional[int], optional
        Random seed if params is not provided, by default None
    dt : Optional[float], optional
        Time step if params is not provided, by default None
    n_walkers : Optional[int], optional
        Number of walkers if params is not provided, by default None
    n_chunk : Optional[int], optional
        Number of chunks if params is not provided, by default 1
    """

    def __init__(
        self,
        mf_or_cc: Any,
        *,
        norb_frozen: Optional[int] = None,
        chol_cut: float = 1e-5,
        cache: Optional[Union[str, Path]] = None,
        n_eql_blocks: Optional[int] = None,
        n_blocks: Optional[int] = None,
        seed: Optional[int] = None,
        dt: Optional[float] = None,
        n_walkers: Optional[int] = None,
        n_chunks: Optional[int] = 1,
    ):
        self._obj = mf_or_cc
        self._cc: Optional[Any] = None
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

        self.walker_kind: Optional[WalkerKind] = None  # resolved in kernel
        self.mixed_precision = True

        self.params: Optional[QmcParams] = None  # resolved in kernel
        params = QmcParams()
        self.dt = params.dt if dt is None else dt
        self.n_walkers = params.n_walkers if n_walkers is None else n_walkers
        self.n_blocks = params.n_blocks if n_blocks is None else n_blocks
        self.n_eql_blocks = params.n_eql_blocks if n_eql_blocks is None else n_eql_blocks
        self.seed = params.seed if seed is None else seed
        self.n_chunks = params.n_chunks if n_chunks is None else n_chunks

        self._staged: Optional[StagedInputs] = None
        self._job: Optional[Job] = None
        self._cache_key: Optional[tuple] = None

        self.e_tot: Optional[float] = None
        self.e_err: Optional[float] = None
        self.block_energies: Any = None
        self.block_weights: Any = None

    @property
    def staged(self) -> Optional[StagedInputs]:
        return self._staged

    @property
    def job(self) -> Optional[Job]:
        return self._job

    def dump_flags(self, job) -> None:
        meta = job.staged.meta
        src = meta["source_kind"]
        chol_cut = meta["chol_cut"]
        sys = job.sys
        nchol = job.staged.ham.chol.shape[0]
        params = job.params
        trial = job.staged.trial
        print("******** AFQMC ********")
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
        print(" QmcParams:")
        print(f"  dt             = {params.dt}")
        print(f"  n_walkers      = {params.n_walkers}")
        print(f"  n_chunk        = {params.n_chunks}")
        print(f"  n_eql_blocks   = {params.n_eql_blocks}")
        print(f"  n_blocks       = {params.n_blocks}")
        print(f"  seed           = {params.seed}\n")

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
            norb_frozen=self.norb_frozen if self.norb_frozen is not None else 0,
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

    def _make_params(self) -> Optional[QmcParams]:
        """
        Create QmcParams if user didn't provide one.
        """
        if self.params is not None:
            return self.params

        kwargs: dict[str, Any] = {
            "n_eql_blocks": self.n_eql_blocks,
            "n_blocks": self.n_blocks,
            "seed": _default_seed() if self.seed is None else int(self.seed),
        }
        if self.dt is not None:
            kwargs["dt"] = float(self.dt)
        if self.n_walkers is not None:
            kwargs["n_walkers"] = int(self.n_walkers)
        if self.n_chunks is not None:
            kwargs["n_chunks"] = int(self.n_chunks)

        kwargs = _filter_kwargs_for(QmcParams, kwargs)
        return QmcParams(**kwargs)

    def build_job(
        self,
        *,
        force: bool = False,
        trial_data: Any = None,
        trial_ops: Any = None,
        meas_ops: Any = None,
        prop_ops: Any = None,
        block_fn: Optional[Callable[..., Any]] = None,
        prop_kwargs: Optional[dict[str, Any]] = None,
    ) -> Job:
        """
        Assemble a runnable Job from current settings and staged inputs.
        """
        if self._job is not None and not force:
            return self._job

        staged = self.stage()
        qmc_params = self._make_params()
        self.params = qmc_params

        job = setup_job(
            staged,
            walker_kind=self.walker_kind,
            mixed_precision=self.mixed_precision,
            params=qmc_params,
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            block_fn=block_fn,
            prop_kwargs=prop_kwargs,
        )
        self._job = job
        return job

    def kernel(self, **driver_kwargs: Any) -> tuple[float, float]:
        """
        Runs AFQMC, returns (e_tot, e_err), and stores samples.
        """
        print(banner_afqmc())
        job = self.build_job()
        self.dump_flags(job)

        out = job.kernel(**driver_kwargs)

        if isinstance(out, tuple) and len(out) >= 2:
            e_tot = float(out[0])
            e_err = float(out[1])
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


def from_staged(
    path: Union[str, Path],
    *,
    n_eql_blocks: Optional[int] = None,
    n_blocks: Optional[int] = None,
    seed: Optional[int] = None,
    dt: Optional[float] = None,
    n_walkers: Optional[int] = None,
    n_chunks: Optional[int] = 1,
):
    """
    Returns a new AFQMC object from a previously staged calculations
    (using save_staged method). The number of frozen orbitals, norb_frozen,
    and the choliesky decomposition threshold, chol_cut, cannot be changed.
    Parameters
    ----------
    path: str, pathlib.Path
    The other parameters are identical to the ones in the AFQMC class.
    """
    staged = load_staged(path)
    meta = staged.meta

    mf_or_cc = None

    # Cannot be changed as the input has been staged
    norb_frozen = meta["norb_frozen"]
    chol_cut = meta["chol_cut"]

    af = AFQMC(
        mf_or_cc,
        norb_frozen=norb_frozen,
        chol_cut=chol_cut,
        n_eql_blocks=n_eql_blocks,
        n_blocks=n_blocks,
        seed=seed,
        dt=dt,
        n_walkers=n_walkers,
        n_chunks=n_chunks,
    )

    af._staged = staged
    af.source_kind = meta["source_kind"]
    af._cache_key = af._key()

    return af
