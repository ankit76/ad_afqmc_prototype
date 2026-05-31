from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, cast

import numpy as np

if TYPE_CHECKING:
    import jax


def _pick_plateau(
    Bs: np.ndarray, # block sizes
    SEs: np.ndarray, # standard errors for each block
    Gs: np.ndarray, # number of blocks for each block size
    *,
    min_blocks: int = 20,
    min_rise: float = 0.20, # require SE to increase by this fraction
    flat_tol: float = 0.03, # plateau criterion
    k: int = 3,
) -> tuple[int, float, int]:
    assert Bs.size > 0
    Bs, SEs, Gs = map(np.asarray, (Bs, SEs, Gs))

    # Only consider block sizes that still have enough blocks to make jackknife stable.
    ok = Gs >= min_blocks
    Bs2, SEs2, Gs2 = Bs[ok], SEs[ok], Gs[ok]

    # If nothing satisfies the `min_blocks` requirement, just return the smallest B.
    if Bs2.size == 0:
        return int(Bs[0]), float(SEs[0]), int(Gs[0])

    # Ignore the early region where SE may be artificially small.
    # Only start plateau detection once SE has increased by at least `min_rise`
    # relative to the first eligible SE.
    rise_ok = SEs2 >= (1.0 + min_rise) * SEs2[0]

    # Scan forward looking for the first "flat" window of length (k+1),
    # but only starting at indices where the "rise" criterion is met.
    for i in range(0, Bs2.size - k):
        if not rise_ok[i]:
            continue
        window = SEs2[i : i + k + 1] # Candidate plateau window

        # Flatness test:
        # For each adjacent pair, require |ΔSE| <= flat_tol * SE(previous).
        # This enforces that SE has stopped changing appreciably (plateau).
        if np.all(np.abs(np.diff(window)) <= flat_tol * window[:-1]):
            return int(Bs2[i]), float(SEs2[i]), int(Gs2[i])
    
    # If no flat window was found, fall back to "near-maximum" SE:
    finite = np.isfinite(SEs2)
    if not np.any(finite):
        return int(Bs[0]), float(SEs[0]), int(Gs[0])

    # 1) Find maximum SE among eligible points
    jmax = int(np.where(finite, SEs2, -np.inf).argmax())

    # 2) Define a threshold slightly below the max (95% of max by default)
    thresh = 0.95 * SEs2[jmax]

    # 3) Pick the earliest B where SE crosses that threshold
    candidates = np.where(SEs2 >= thresh)[0]
    j = int(candidates[0]) if candidates.size > 0 else jmax
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
    """
    Blocking analysis for mu = sum(wt*ene)/sum(wt).

    Estimates the standard error (SE) of the weighted ratio estimator
        mu_hat = (Σ_i w_i e_i) / (Σ_i w_i)
    in the presence of autocorrelation / serial correlation.

    "blocking" + leave-one-out (LOO) jackknife on *block sums*.
    For each block size B, group the samples into G blocks, compute per-block
    contributions to numerator/denominator, then jackknife the ratio by leaving
    out one block at a time. The SE as a function of B typically increases and
    then plateaus; we pick a plateau point B* heuristically.
    """
    ene = np.asarray(ene, float).ravel()
    wt = np.asarray(wt, float).ravel()
    n = ene.size
    assert wt.size == n
    
    # Define numerator/denominator sample-wise contributions:
    #   S_i = w_i * e_i   and   N_i = w_i
    S = wt * ene
    N = wt

    # Full-sample ratio estimate (no blocking)
    S_tot, N_tot = S.sum(), N.sum()
    mu_full = S_tot / N_tot
    
    # Default block grid: log-spaced block sizes from 1 up to ~n/min_blocks.
    # Keep only B where we still have at least `min_blocks` blocks (G = n//B).
    if block_grid is None:
        raw = np.unique(np.rint(np.geomspace(1, max(2, n // min_blocks), 18)).astype(int))
        block_grid = [int(b) for b in raw if b >= 1 and (n // b) >= min_blocks]

        # Optionally include the largest candidate even if it yields fewer than 
        # `min_blocks` (but at least 5 blocks) to help detect the plateau at large B.
        if (n // raw[-1]) >= 5 and raw[-1] not in block_grid:
            block_grid.append(int(raw[-1]))
    
    # Accumulators for the SE curve and block counts, plus a cache of LOO values
    # so we don't have to recompute for the selected B* later.
    Bs_list: list[int] = [] # candidate block sizes
    SEs_list: list[float] = [] # SE(mu) estimated at each block size
    Gs_list: list[int] = [] # number of blocks G = floor(n/B)
    LOO_cache: dict[int, tuple[np.ndarray, float, int]] = {} # B -> (mu_loo, mu_bar, G)

    # Loop over candidate block sizes
    for B in block_grid:
        G = n // B
        if G < 5:
            # Jackknife variance is unstable with too few blocks; skip.
            continue

        usable = G * B # drop remainder so reshaping is clean

        # Leave-one-out (LOO) ratio estimates:
        # mu^{(-g)} = (St - Sg[g]) / (Nt - Ng[g])
        Sg = S[:usable].reshape(G, B).sum(axis=1)
        Ng = N[:usable].reshape(G, B).sum(axis=1)
        St, Nt = Sg.sum(), Ng.sum()

        denom_loo = Nt - Ng
        safe = np.abs(denom_loo) > 1e-18
        mu_loo = np.where(safe, (St - Sg) / denom_loo, St / Nt)
        
        # Jackknife estimate of variance of mu from the LOO samples.
        # Here we use a standard jackknife variance form:
        #   var_jack = (G-1)/G * Σ_g (mu^{(-g)} - mean(mu^{(-·)}))^2
        mu_bar = mu_loo.mean()
        var = (G - 1) / G * np.sum((mu_loo - mu_bar) ** 2)
        se = float(np.sqrt(max(var, 0.0)))
        
        # Store curve values
        Bs_list.append(B)
        SEs_list.append(se)
        Gs_list.append(G)

        # Cache LOO values so we can build "estimator-scale" samples at B*
        LOO_cache[B] = (mu_loo, mu_bar, G)

    Bs = np.array(Bs_list, int)
    SEs = np.array(SEs_list, float)
    Gs = np.array(Gs_list, int)
    ci95: tuple[float, float] | None = None

    # If no valid block sizes, we can't do blocking analysis.
    if Bs.size == 0:
        B_star: int | None = None
        se_star: float | None = None
        G_star: int | None = None
    else:
        # Choose a "plateau" block size B*:
        # - require enough blocks (>= min_blocks)
        # - require SE has increased by `min_rise` relative to smallest eligible B
        # - then look for a window of k+1 points where SE is locally flat (within flat_tol)
        # - fallback: earliest point reaching 95% of max SE
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

    # Convert LOO samples into "estimator-scale" samples (approximately N(mu_full, SE*^2)).
    # This is a jackknife pseudo-value style rescaling: it inflates the LOO deviations
    # by (G-1)/sqrt(G) to match the jackknife SE convention used above.
    est_samples = mu_full + (G - 1) / np.sqrt(G) * (mu_loo - mu_bar)
    
    # Optional diagnostics if an exact/target value is provided:
    # bias = mu_hat - exact, and z = (mu_hat - exact)/SE*
    bias = z = None
    if exact is not None and np.isfinite(se_star) and se_star > 0:
        bias = float(mu_full - exact)
        z = float((mu_full - exact) / se_star)

    out = {
        "mu": float(mu_full),
        "block_sizes": Bs,              # candidate B values used
        "se_curve": SEs,                # SE(B) curve from jackknife
        "n_blocks": Gs,                 # G(B) = n//B
        "B_star": int(B_star),          # chosen plateau block size
        "se_star": float(se_star),      # chosen plateau standard error
        "ci95_star": (float(ci95[0]), float(ci95[1])), # normal-approx 95% CI
        "estimator_scale_samples": est_samples, # pseudo-samples for histogram/QQ/etc
        "bias": bias,
        "z_score": z,
    }
    if print_q:
        print(f"mu: {out['mu']:.16g}  SE*: {out['se_star']:.16g}  95% CI: {out['ci95_star']}")
        if out["z_score"] is not None:
            print(f"bias: {out['bias']:.16g}  z: {out['z_score']:.6g}")

        # Print a table of the SE curve and mark the chosen B*
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

        # Left panel: SE(B) curve; B* marked as a vertical line.
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

        # Right panel: histogram of estimator-scale pseudo-samples + normal curve using SE*
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
