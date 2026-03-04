from typing import Any, Dict, Iterable, Tuple

import numpy as np


def _pick_plateau(Bs, SEs, Gs, *, min_blocks=20, min_rise=0.20, flat_tol=0.03, k=3):
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
    ene,
    wt,
    block_grid: Iterable[int] | None = None,
    *,
    min_blocks: int = 20,
    min_rise: float = 0.20,
    flat_tol: float = 0.03,
    k: int = 3,
    bins: int | str = "fd",
    figsize: Tuple[float, float] = (12, 4.2),
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
        raw = np.unique(
            np.rint(np.geomspace(1, max(2, n // min_blocks), 18)).astype(int)
        )
        block_grid = [int(b) for b in raw if b >= 1 and (n // b) >= min_blocks]
        if (n // raw[-1]) >= 5 and raw[-1] not in block_grid:
            block_grid.append(int(raw[-1]))

    Bs, SEs, Gs, LOO_cache = [], [], [], {}
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

        Bs.append(B)
        SEs.append(se)
        Gs.append(G)
        LOO_cache[B] = (mu_loo, mu_bar, G)

    Bs = np.array(Bs, int)
    SEs = np.array(SEs, float)
    Gs = np.array(Gs, int)
    if Bs.size == 0:
        B_star, se_star, G_star = None, None, None
    else:
        B_star, se_star, G_star = _pick_plateau(
            Bs, SEs, Gs, min_blocks=min_blocks, min_rise=min_rise, flat_tol=flat_tol, k=k
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
        print(
            f"mu: {out['mu']:.16g}  SE*: {out['se_star']:.16g}  95% CI: {out['ci95_star']}"
        )
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
        ax1.axvline(
            B_star, ls="--", color="k", alpha=0.85, label=f"chosen B = {B_star}"
        )
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


def reject_outliers(data, obs, m=10.0, min_threshold=1e-5):
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


def jackknife_ratios(num: np.ndarray, denom: np.ndarray):
    r"""Jackknife estimation of standard deviation of the ratio of means.

    Parameters
    ----------
    num : :class:`np.ndarray
        Numerator samples.
    denom : :class:`np.ndarray`
        Denominator samples.

    Returns
    -------
    mean : :class:`np.ndarray`
        Ratio of means.
    sigma : :class:`np.ndarray`
        Standard deviation of the ratio of means.
    """
    n_samples = num.size
    num_mean = np.mean(num)
    denom_mean = np.mean(denom)
    mean = num_mean / denom_mean
    mean_num_all = (num_mean * n_samples - num) / (n_samples - 1)
    mean_denom_all = (denom_mean * n_samples - denom) / (n_samples - 1)
    jackknife_estimates = (mean_num_all / mean_denom_all).real
    mean = np.mean(jackknife_estimates)
    sigma = np.sqrt((n_samples - 1) * np.var(jackknife_estimates))
    return mean, sigma
