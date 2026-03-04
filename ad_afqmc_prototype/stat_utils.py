from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, cast

import numpy as np

if TYPE_CHECKING:
    import jax


def _pick_plateau(
    Bs: np.ndarray,
    SEs: np.ndarray,
    Gs: np.ndarray,
    *,
    min_blocks: int = 20,
    min_rise: float = 0.20,
    flat_tol: float = 0.03,
    k: int = 3,
) -> tuple[int, float, int]:
    assert Bs.size > 0
    Bs, SEs, Gs = map(np.asarray, (Bs, SEs, Gs))
    ok = Gs >= min_blocks
    Bs2, SEs2, Gs2 = Bs[ok], SEs[ok], Gs[ok]
    if Bs2.size == 0:
        return int(Bs[0]), float(SEs[0]), int(Gs[0])
    rise_ok = SEs2 >= (1.0 + min_rise) * SEs2[0]
    for i in range(0, Bs2.size - k):
        if not rise_ok[i]:
            continue
        window = SEs2[i : i + k + 1]
        if np.all(np.abs(np.diff(window)) <= flat_tol * window[:-1]):
            return int(Bs2[i]), float(SEs2[i]), int(Gs2[i])
    jmax = int(np.argmax(SEs2))
    thresh = 0.95 * SEs2[jmax]
    j = int(np.where(SEs2 >= thresh)[0][0])
    return int(Bs2[j]), float(SEs2[j]), int(Gs2[j])


def blocking_analysis_ratio(
    ene: np.ndarray | jax.Array,
    wt: np.ndarray | jax.Array,
    block_grid: Iterable[int] | None = None,
    *,
    min_blocks: int = 20,
    min_rise: float = 0.20,
    flat_tol: float = 0.03,
    k: int = 3,
    bins: int | str = "fd",
    figsize: tuple[float, float] = (12, 4.2),
    title: str | None = None,
    print_q: bool = True,
    plot_q: bool = False,
    exact: float | None = None,
) -> Dict[str, Any]:
    """Blocking analysis for mu = sum(wt*ene)/sum(wt)"""
    ene = np.asarray(ene, float).ravel()
    wt = np.asarray(wt, float).ravel()
    n = ene.size
    assert wt.size == n

    S = wt * ene
    N = wt
    S_tot, N_tot = S.sum(), N.sum()
    mu_full = S_tot / N_tot

    if block_grid is None:
        raw = np.unique(np.rint(np.geomspace(1, max(2, n // min_blocks), 18)).astype(int))
        block_grid = [int(b) for b in raw if b >= 1 and (n // b) >= min_blocks]
        if (n // raw[-1]) >= 5 and raw[-1] not in block_grid:
            block_grid.append(int(raw[-1]))

    Bs_list: list[int] = []
    SEs_list: list[float] = []
    Gs_list: list[int] = []
    LOO_cache: dict[int, tuple[np.ndarray, float, int]] = {}
    for B in block_grid:
        G = n // B
        if G < 5:
            continue
        usable = G * B
        Sg = S[:usable].reshape(G, B).sum(axis=1)
        Ng = N[:usable].reshape(G, B).sum(axis=1)
        St, Nt = Sg.sum(), Ng.sum()

        denom_loo = Nt - Ng
        safe = np.abs(denom_loo) > 1e-18
        mu_loo = np.where(safe, (St - Sg) / denom_loo, St / Nt)

        mu_bar = mu_loo.mean()
        var = (G - 1) / G * np.sum((mu_loo - mu_bar) ** 2)
        se = float(np.sqrt(max(var, 0.0)))

        Bs_list.append(B)
        SEs_list.append(se)
        Gs_list.append(G)
        LOO_cache[B] = (mu_loo, mu_bar, G)

    Bs = np.array(Bs_list, int)
    SEs = np.array(SEs_list, float)
    Gs = np.array(Gs_list, int)
    ci95: tuple[float, float] | None = None
    if Bs.size == 0:
        B_star: int | None = None
        se_star: float | None = None
        G_star: int | None = None
    else:
        B_star, se_star, G_star = _pick_plateau(
            Bs,
            SEs,
            Gs,
            min_blocks=min_blocks,
            min_rise=min_rise,
            flat_tol=flat_tol,
            k=k,
        )
        ci95 = (mu_full - 1.96 * se_star, mu_full + 1.96 * se_star)

    if B_star is None:
        # Blocking analysis not possible
        out = {
            "mu": float(mu_full),
            "block_sizes": None,
            "se_curve": None,
            "n_blocks": None,
            "B_star": None,
            "se_star": None,
            "ci95_star": (None, None),
            "estimator_scale_samples": None,
            "bias": None,
            "z_score": None,
        }
        return out

    se_star = cast(float, se_star)
    ci95 = cast(tuple[float, float], ci95)
    mu_loo, mu_bar, G = LOO_cache[B_star]
    est_samples = mu_full + (G - 1) / np.sqrt(G) * (mu_loo - mu_bar)

    bias = z = None
    if exact is not None and np.isfinite(se_star) and se_star > 0:
        bias = float(mu_full - exact)
        z = float((mu_full - exact) / se_star)

    out = {
        "mu": float(mu_full),
        "block_sizes": Bs,
        "se_curve": SEs,
        "n_blocks": Gs,
        "B_star": int(B_star),
        "se_star": float(se_star),
        "ci95_star": (float(ci95[0]), float(ci95[1])),
        "estimator_scale_samples": est_samples,
        "bias": bias,
        "z_score": z,
    }
    if print_q:
        print(f"mu: {out['mu']:.16g}  SE*: {out['se_star']:.16g}  95% CI: {out['ci95_star']}")
        if out["z_score"] is not None:
            print(f"bias: {out['bias']:.16g}  z: {out['z_score']:.6g}")

        # table: block size vs SE, mark chosen B*
        se0 = float(SEs[0]) if SEs.size else float("nan")
        print("\nBlocking SE curve (ratio LOO):")
        print(f"{'':1s}{'B':>6s} {'G':>6s} {'SE':>14s} {'SE/SE(B=1)':>12s}")
        for B, G, se in zip(Bs, Gs, SEs):
            mark = "*" if int(B) == int(B_star) else " "
            rel = (float(se) / se0) if (se0 > 0 and np.isfinite(se0)) else float("nan")
            print(f"{mark}{int(B):6d} {int(G):6d} {float(se):14.6e} {rel:12.3f}")
        print("")  # trailing newline

    if plot_q:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # SE curve
        ax1.plot(Bs, SEs, marker="o", lw=1.6)
        ax1.axvline(B_star, ls="--", color="k", alpha=0.85, label=f"chosen B = {B_star}")
        if exact is not None:
            ax1.set_title(
                (title or "Blocking SE for ratio estimator")
                + "\n"
                + rf"$\mu$={mu_full:.6f}, SE*={se_star:.3e}, bias={bias:.3e}, z={z:.2f}"
            )
        else:
            ax1.set_title(title or "Blocking SE for ratio estimator")
        ax1.set_xscale("log")
        ax1.set_xlabel("block size B (walkers)")
        ax1.set_ylabel(r"SE[$\mu$]")
        ax1.grid(True, alpha=0.25)
        ax1.legend()

        # estimator-scale histogram
        ax2.hist(est_samples, bins=bins, density=True, alpha=0.6, edgecolor="white")
        xs = np.linspace(mu_full - 6 * se_star, mu_full + 6 * se_star, 400)
        pdf = (1.0 / (se_star * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((xs - mu_full) / se_star) ** 2
        )
        ax2.plot(xs, pdf, lw=2.0, color="#f58518", label="Normal(SE*)")
        ax2.axvline(mu_full, ls="--", color="k", lw=1.2, label=r"$\hat\mu$")
        ax2.axvline(ci95[0], ls=":", color="k", lw=1.2, label="95% CI")
        ax2.axvline(ci95[1], ls=":", color="k", lw=1.2)
        if exact is not None:
            ax2.axvline(exact, ls="--", color="red", lw=1.4, label="exact/target")
        ax2.set_xlabel("estimator-scale (rescaled LOO)")
        ax2.set_ylabel("density")
        ax2.legend()
        fig.tight_layout()

    return out


def reject_outliers(
    data: np.ndarray | jax.Array,
    obs: int,
    m: float = 10.0,
    min_threshold: float = 1e-5,
) -> tuple[Any, Any]:
    target = data[:, obs]
    median_val = np.median(target)
    d = np.abs(target - median_val)
    mdev = np.median(d)
    q1, q3 = np.percentile(target, [25, 75])
    iqr = q3 - q1
    normalized_iqr = iqr / 1.349
    dispersion = max(mdev, normalized_iqr, min_threshold)
    s = d / dispersion
    mask = s < m
    return data[mask], mask


def jackknife_ratios(
    num: np.ndarray,
    denom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Jackknife mean and standard error for a ratio estimator with array valued numerator.

    Parameters
    ----------
    num : np.ndarray
        Numerator samples, shape (n_samples, *obs_shape).
    denom : np.ndarray
        Denominator samples, shape (n_samples,).

    Returns
    -------
    mean : np.ndarray
        Jackknife estimate of the ratio mean, shape (*obs_shape,).
    sigma : np.ndarray
        Jackknife standard error, shape (*obs_shape,).
    """
    num = np.asarray(num)
    denom = np.asarray(denom).ravel()
    n = num.shape[0]
    assert denom.shape[0] == n

    num_sum = num.sum(axis=0)
    denom_sum = denom.sum()

    # leave one out sums
    loo_num = (num_sum - num) / (n - 1)  # (n, *obs_shape)
    d_shape = (n,) + (1,) * (num.ndim - 1)
    loo_denom = (denom_sum - denom).reshape(d_shape) / (n - 1)  # (n, 1, ...)

    loo_ratio = (loo_num / loo_denom).real  # (n, *obs_shape)
    mean = loo_ratio.mean(axis=0)
    sigma = np.sqrt((n - 1) * np.var(loo_ratio, axis=0))
    return mean, sigma


def rebin_observable(
    obs: np.ndarray,
    weights: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebin block-level observable data into larger super-blocks.

    Parameters
    ----------
    obs : np.ndarray
        Per-block weighted-mean observable, shape ``(n_blocks, *obs_shape)``.
    weights : np.ndarray
        Per-block total weights, shape ``(n_blocks,)``.
    block_size : int
        Number of original blocks per super-block.

    Returns
    -------
    num : np.ndarray
        Super-block numerator sums, shape ``(n_groups, *obs_shape)``.
    denom : np.ndarray
        Super-block denominator sums, shape ``(n_groups,)``.
    """
    obs = np.asarray(obs)
    weights = np.asarray(weights).ravel()
    n = obs.shape[0]
    n_groups = n // block_size
    usable = n_groups * block_size

    w = weights[:usable].reshape(n_groups, block_size)
    w_shape = (n_groups, block_size) + (1,) * (obs.ndim - 1)
    o = obs[:usable].reshape((n_groups, block_size) + obs.shape[1:])

    denom = w.sum(axis=1)  # (n_groups,)
    num = (w.reshape(w_shape) * o).sum(axis=1)  # (n_groups, *obs_shape)
    return num, denom
