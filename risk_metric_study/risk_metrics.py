"""
Risk metric helpers.

Custom metric (user-defined):
    R = sigma_neg * (1 + N_neg / N_pos) * (1 + L / L_ref)

where:
    sigma_neg = sample std. of negative daily returns
    N_neg / N_pos = count ratio of down vs up days
    L = longest consecutive losing-day streak
    L_ref = streak scale (default 5 trading days)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _down_mask(returns: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Boolean mask for loss days (strictly below threshold). Zeros excluded."""
    return returns < threshold


def max_losing_streak(returns: pd.Series, threshold: float = 0.0) -> int:
    """Longest consecutive run of returns strictly below threshold."""
    is_loss = _down_mask(returns, threshold).to_numpy()
    best = cur = 0
    for flag in is_loss:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def downside_std(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Std. of returns that fall below threshold.

    Uses sample std (ddof=1). Returns NaN if fewer than 2 loss observations.
    """
    losses = returns[_down_mask(returns, threshold)]
    if len(losses) < 2:
        return np.nan
    return float(losses.std(ddof=1))


def custom_risk_metric(
    returns: pd.Series,
    l_ref: float = 5.0,
    threshold: float = 0.0,
    freq_cap: float = 5.0,
) -> float:
    """
    User risk score:
        R = sigma_neg * (1 + N_neg/N_pos) * (1 + L/l_ref)

    Higher => riskier.
    """
    r = returns.dropna()
    if len(r) < 20:
        return np.nan

    # Exclude flat days from the frequency ratio so zeros do not dilute counts.
    nonzero = r[r != 0]
    n_neg = int((nonzero < threshold).sum())
    n_pos = int((nonzero > threshold).sum())

    sigma_neg = downside_std(r, threshold=threshold)
    if not np.isfinite(sigma_neg):
        return np.nan

    if n_pos == 0:
        freq_term = 1.0 + freq_cap
    else:
        freq_term = 1.0 + min(n_neg / n_pos, freq_cap)

    streak = max_losing_streak(r, threshold=threshold)
    streak_term = 1.0 + (streak / l_ref)

    return float(sigma_neg * freq_term * streak_term)


def max_drawdown(returns: pd.Series) -> float:
    """Peak-to-trough drawdown magnitude as a positive number."""
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    running_peak = wealth.cummax()
    dd = wealth / running_peak - 1.0
    return float(-dd.min()) if len(dd) else np.nan


def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical VaR as a positive loss magnitude (e.g. 0.02 = 2%)."""
    r = returns.dropna()
    if len(r) < 20:
        return np.nan
    # Lower-tail quantile is negative; flip sign so larger = riskier.
    return float(-np.quantile(r, alpha))


def historical_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Expected shortfall / CVaR as positive loss magnitude."""
    r = returns.dropna()
    if len(r) < 20:
        return np.nan
    cutoff = np.quantile(r, alpha)
    tail = r[r <= cutoff]
    if len(tail) == 0:
        return np.nan
    return float(-tail.mean())


def compute_all_risk_metrics(returns: pd.Series, l_ref: float = 5.0) -> dict:
    """Compute custom metric plus common benchmark risk metrics on one series."""
    r = returns.dropna()
    return {
        "custom_R": custom_risk_metric(r, l_ref=l_ref),
        "std": float(r.std(ddof=1)) if len(r) >= 2 else np.nan,
        "downside_std": downside_std(r),
        "max_drawdown": max_drawdown(r),
        "var_5": historical_var(r, 0.05),
        "cvar_5": historical_cvar(r, 0.05),
        "loss_freq": float((r < 0).mean()) if len(r) else np.nan,
        "max_loss_streak": float(max_losing_streak(r)),
    }
