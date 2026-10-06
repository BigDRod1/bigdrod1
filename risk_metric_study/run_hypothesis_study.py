"""
In-sample / out-of-sample hypothesis study:

Does the custom risk metric predict future realized risk better than
standard risk metrics?

Null hypothesis (per benchmark B):
    H0: Spearman(custom_IS, realized_OOS) <= Spearman(B_IS, realized_OOS)
Alternative:
    H1: custom metric has strictly stronger rank correlation with OOS risk

We also report:
    - absolute predictive correlations
    - bootstrap confidence intervals for correlation differences
    - quintile monotonicity of OOS risk when sorted by each IS metric
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from risk_metrics import compute_all_risk_metrics

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Liquid, diversified cross-section spanning styles/sectors/asset classes.
TICKERS = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD",
    "VNQ", "XLE", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "XOM", "UNH",
    "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO", "COST",
    "WMT", "BAC", "CRM", "AMD", "NFLX", "DIS", "PFE", "T", "INTC", "BA",
]

BENCHMARK_METRICS = ["std", "downside_std", "max_drawdown", "var_5", "cvar_5"]
# Realized OOS targets we want a good risk metric to forecast.
OOS_TARGETS = ["oos_max_drawdown", "oos_std", "oos_cvar_5", "oos_custom_R"]

IS_DAYS = 252          # ~1y formation / in-sample window
OOS_DAYS = 63          # ~1q evaluation / out-of-sample window
STEP_DAYS = 21         # roll forward ~1 month between experiments
MIN_ASSETS = 25
N_BOOT = 2000
SEED = 42
L_REF = 5.0


def download_prices(tickers: list[str], start: str = "2015-01-01") -> pd.DataFrame:
    """Download adjusted close prices; drop tickers with too many missing points."""
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    # yfinance returns MultiIndex columns when multiple tickers are requested.
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].copy()
    else:
        px = raw[["Close"]].copy()
        px.columns = tickers[:1]

    # Require mostly complete history so IS/OOS windows are comparable.
    ok = px.notna().mean() >= 0.95
    px = px.loc[:, ok].sort_index().ffill(limit=5)
    return px.dropna(how="all")


def window_metrics(returns_panel: pd.DataFrame, start_i: int, end_i: int) -> pd.DataFrame:
    """Compute risk metrics for every asset on returns[start_i:end_i]."""
    rows = []
    slice_df = returns_panel.iloc[start_i:end_i]
    for ticker in slice_df.columns:
        m = compute_all_risk_metrics(slice_df[ticker], l_ref=L_REF)
        m["ticker"] = ticker
        rows.append(m)
    return pd.DataFrame(rows).set_index("ticker")


def build_walk_forward_table(returns_panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each rebalance date:
      IS metrics on past IS_DAYS
      OOS realized risk on next OOS_DAYS
    """
    n = len(returns_panel)
    records = []

    # i = end of IS window (exclusive end index in iloc terms after +IS_DAYS)
    start = IS_DAYS
    while start + OOS_DAYS <= n:
        is_start = start - IS_DAYS
        is_end = start
        oos_end = start + OOS_DAYS

        is_m = window_metrics(returns_panel, is_start, is_end)
        oos_m = window_metrics(returns_panel, is_end, oos_end)

        # Align assets present in both windows with finite custom metric.
        joined = is_m.join(
            oos_m.add_prefix("oos_"),
            how="inner",
        ).dropna(subset=["custom_R", "oos_max_drawdown", "oos_std", "oos_cvar_5"])

        if len(joined) >= MIN_ASSETS:
            joined = joined.copy()
            joined["asof"] = returns_panel.index[is_end - 1]
            joined["oos_end"] = returns_panel.index[oos_end - 1]
            records.append(joined.reset_index())

        start += STEP_DAYS

    if not records:
        raise RuntimeError("No valid walk-forward windows produced.")
    return pd.concat(records, ignore_index=True)


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return np.nan
    rho, _ = stats.spearmanr(x[mask], y[mask])
    return float(rho)


def bootstrap_delta_rho(
    custom: np.ndarray,
    bench: np.ndarray,
    target: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """
    Bootstrap the difference:
        delta = rho(custom, target) - rho(bench, target)

    One-sided p-value for H1: delta > 0 (custom beats benchmark).
    """
    rng = np.random.default_rng(seed)
    n = len(target)
    base_c = stats.spearmanr(custom, target).statistic
    base_b = stats.spearmanr(bench, target).statistic
    base_delta = float(base_c - base_b)

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rc = stats.spearmanr(custom[idx], target[idx]).statistic
        rb = stats.spearmanr(bench[idx], target[idx]).statistic
        deltas[i] = rc - rb

    # One-sided: fraction of bootstrap deltas <= 0 under observed sampling variability.
    p_better = float(np.mean(deltas <= 0))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "rho_custom": float(base_c),
        "rho_bench": float(base_b),
        "delta": base_delta,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_custom_better": p_better,
    }


def pooled_tests(wf: pd.DataFrame) -> pd.DataFrame:
    """Pool all asset-date rows and compare custom vs each benchmark for each OOS target."""
    rows = []
    for target in OOS_TARGETS:
        y = wf[target].to_numpy(dtype=float)
        custom = wf["custom_R"].to_numpy(dtype=float)
        for bench in BENCHMARK_METRICS:
            x_b = wf[bench].to_numpy(dtype=float)
            mask = np.isfinite(custom) & np.isfinite(x_b) & np.isfinite(y)
            res = bootstrap_delta_rho(custom[mask], x_b[mask], y[mask])
            rows.append(
                {
                    "oos_target": target,
                    "benchmark": bench,
                    **res,
                    "n": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def per_date_average_correlations(wf: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional Spearman on each asof date, then average over time.
    This avoids pooling bias from repeated assets.
    """
    metrics = ["custom_R"] + BENCHMARK_METRICS
    rows = []
    for asof, g in wf.groupby("asof"):
        for target in OOS_TARGETS:
            for m in metrics:
                rows.append(
                    {
                        "asof": asof,
                        "oos_target": target,
                        "metric": m,
                        "rho": spearman_corr(g[m], g[target]),
                        "n_assets": len(g),
                    }
                )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["oos_target", "metric"])["rho"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_rho", "median": "med_rho", "std": "std_rho", "count": "n_dates"})
    )
    return detail, summary


def quintile_oos_risk(wf: pd.DataFrame, metric: str, target: str = "oos_max_drawdown") -> pd.DataFrame:
    """Average OOS risk by in-sample metric quintile (1=low predicted risk, 5=high)."""
    pieces = []
    for asof, g in wf.groupby("asof"):
        # rank(method='first') avoids ties breaking qcut.
        g = g.copy()
        g["q"] = pd.qcut(g[metric].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        tmp = g.groupby("q", observed=True)[target].mean().rename("avg_oos_risk").reset_index()
        tmp["asof"] = asof
        tmp["metric"] = metric
        tmp["target"] = target
        pieces.append(tmp)
    detail = pd.concat(pieces, ignore_index=True)
    summary = (
        detail.groupby(["metric", "target", "q"], observed=True)["avg_oos_risk"]
        .mean()
        .reset_index()
    )
    return summary


def train_test_split_dates(wf: pd.DataFrame, train_frac: float = 0.6):
    """Chronological split of walk-forward dates into IS meta-train / OOS meta-test."""
    dates = np.array(sorted(wf["asof"].unique()))
    cut = int(len(dates) * train_frac)
    train_dates = set(dates[:cut])
    test_dates = set(dates[cut:])
    train = wf[wf["asof"].isin(train_dates)].copy()
    test = wf[wf["asof"].isin(test_dates)].copy()
    return train, test


def main() -> None:
    print("Downloading price history...")
    prices = download_prices(TICKERS)
    returns = prices.pct_change().dropna(how="all")
    print(f"Prices shape={prices.shape}, returns shape={returns.shape}")
    print(f"Date range: {returns.index.min().date()} -> {returns.index.max().date()}")
    print(f"Assets kept: {list(returns.columns)}")

    print("Building walk-forward IS/OOS table...")
    wf = build_walk_forward_table(returns)
    wf.to_csv(RESULTS / "walk_forward_panel.csv", index=False)
    print(f"Walk-forward rows={len(wf)}, dates={wf['asof'].nunique()}, assets~{wf.groupby('asof').size().median():.0f}")

    # Chronological meta split: calibrate narrative on earlier windows, confirm on later ones.
    train, test = train_test_split_dates(wf, train_frac=0.6)
    train.to_csv(RESULTS / "panel_in_sample_period.csv", index=False)
    test.to_csv(RESULTS / "panel_out_of_sample_period.csv", index=False)

    print("Running pooled bootstrap hypothesis tests (full sample)...")
    full_tests = pooled_tests(wf)
    full_tests.to_csv(RESULTS / "hypothesis_tests_full.csv", index=False)

    print("Running IS-period and OOS-period hypothesis tests...")
    train_tests = pooled_tests(train)
    test_tests = pooled_tests(test)
    train_tests.to_csv(RESULTS / "hypothesis_tests_in_sample_period.csv", index=False)
    test_tests.to_csv(RESULTS / "hypothesis_tests_out_of_sample_period.csv", index=False)

    print("Average cross-sectional correlations over time...")
    _, corr_summary_full = per_date_average_correlations(wf)
    _, corr_summary_train = per_date_average_correlations(train)
    _, corr_summary_test = per_date_average_correlations(test)
    corr_summary_full.to_csv(RESULTS / "avg_cs_correlations_full.csv", index=False)
    corr_summary_train.to_csv(RESULTS / "avg_cs_correlations_in_sample_period.csv", index=False)
    corr_summary_test.to_csv(RESULTS / "avg_cs_correlations_out_of_sample_period.csv", index=False)

    print("Quintile monotonicity checks...")
    q_rows = []
    for metric in ["custom_R"] + BENCHMARK_METRICS:
        q_rows.append(quintile_oos_risk(test, metric, target="oos_max_drawdown"))
        q_rows.append(quintile_oos_risk(test, metric, target="oos_cvar_5"))
    quintiles = pd.concat(q_rows, ignore_index=True)
    quintiles.to_csv(RESULTS / "quintile_oos_risk_out_of_sample_period.csv", index=False)

    # Compact headline summary for humans.
    headline = {
        "n_assets": int(returns.shape[1]),
        "n_days": int(returns.shape[0]),
        "n_wf_dates": int(wf["asof"].nunique()),
        "n_train_dates": int(train["asof"].nunique()),
        "n_test_dates": int(test["asof"].nunique()),
        "is_days": IS_DAYS,
        "oos_days": OOS_DAYS,
        "step_days": STEP_DAYS,
        "l_ref": L_REF,
        "formula": "R = downside_std * (1 + N_neg/N_pos) * (1 + L/5)",
        "null": "Spearman(custom, OOS_risk) <= Spearman(benchmark, OOS_risk)",
        "alternative": "custom has higher Spearman correlation with OOS risk",
    }

    # Focus table: predicting OOS max drawdown and OOS CVaR on the true holdout period.
    focus_targets = ["oos_max_drawdown", "oos_cvar_5", "oos_std"]
    focus = test_tests[test_tests["oos_target"].isin(focus_targets)].copy()
    focus["custom_wins_5pct"] = focus["p_custom_better"] < 0.05
    focus["custom_wins_directionally"] = focus["delta"] > 0

    with open(RESULTS / "headline.json", "w") as f:
        json.dump(headline, f, indent=2, default=str)

    focus.to_csv(RESULTS / "oos_holdout_focus.csv", index=False)

    print("\n=== HOLD-OUT (later period) focus: custom vs benchmarks ===")
    print(focus.to_string(index=False))

    # Save a markdown report.
    report = build_markdown_report(
        headline, corr_summary_train, corr_summary_test, train_tests, test_tests, focus, quintiles
    )
    (RESULTS / "REPORT.md").write_text(report)
    print(f"\nWrote results to {RESULTS}")


def build_markdown_report(
    headline: dict,
    corr_train: pd.DataFrame,
    corr_test: pd.DataFrame,
    train_tests: pd.DataFrame,
    test_tests: pd.DataFrame,
    focus: pd.DataFrame,
    quintiles: pd.DataFrame,
) -> str:
    def corr_block(df: pd.DataFrame, target: str) -> str:
        sub = df[df["oos_target"] == target].sort_values("avg_rho", ascending=False)
        lines = ["| metric | avg Spearman rho |", "|---|---:|"]
        for _, r in sub.iterrows():
            lines.append(f"| {r['metric']} | {r['avg_rho']:.4f} |")
        return "\n".join(lines)

    wins = int(focus["custom_wins_directionally"].sum())
    sig_wins = int(focus["custom_wins_5pct"].sum())
    total = len(focus)

    lines = [
        "# Custom Risk Metric Hypothesis Study",
        "",
        "## Formula under test",
        "",
        f"`{headline['formula']}`",
        "",
        "## Design",
        "",
        f"- Assets: {headline['n_assets']}",
        f"- Daily history length: {headline['n_days']}",
        f"- In-sample window: {headline['is_days']} trading days",
        f"- Out-of-sample window: {headline['oos_days']} trading days",
        f"- Walk-forward step: {headline['step_days']} trading days",
        f"- Walk-forward dates: {headline['n_wf_dates']} (train {headline['n_train_dates']} / holdout {headline['n_test_dates']})",
        "",
        "## Hypotheses",
        "",
        f"- **H0**: `{headline['null']}`",
        f"- **H1**: `{headline['alternative']}`",
        "",
        "Benefit is defined as **stronger rank correlation** between the in-sample risk score and realized out-of-sample risk (max drawdown, CVaR, volatility).",
        "",
        "## Average cross-sectional predictive correlations",
        "",
        "### Train period (earlier dates) — target: OOS max drawdown",
        corr_block(corr_train, "oos_max_drawdown"),
        "",
        "### Holdout period (later dates) — target: OOS max drawdown",
        corr_block(corr_test, "oos_max_drawdown"),
        "",
        "### Holdout period — target: OOS CVaR(5%)",
        corr_block(corr_test, "oos_cvar_5"),
        "",
        "### Holdout period — target: OOS std",
        corr_block(corr_test, "oos_std"),
        "",
        "## Bootstrap tests on holdout period",
        "",
        f"Custom beats benchmark directionally in **{wins}/{total}** comparisons; statistically significant at 5% one-sided in **{sig_wins}/{total}**.",
        "",
        "| OOS target | benchmark | rho_custom | rho_bench | delta | 95% CI | p(custom better) |",
        "|---|---|---:|---:|---:|---|---:|",
    ]

    for _, r in focus.sort_values(["oos_target", "benchmark"]).iterrows():
        lines.append(
            f"| {r['oos_target']} | {r['benchmark']} | {r['rho_custom']:.4f} | {r['rho_bench']:.4f} | "
            f"{r['delta']:.4f} | [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] | {r['p_custom_better']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Holdout quintiles (avg OOS max drawdown by predicted-risk quintile)",
            "",
            "| metric | Q1 (low) | Q2 | Q3 | Q4 | Q5 (high) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    qsub = quintiles[quintiles["target"] == "oos_max_drawdown"]
    for metric, g in qsub.groupby("metric"):
        vals = g.set_index("q")["avg_oos_risk"]
        lines.append(
            "| {m} | {a:.4f} | {b:.4f} | {c:.4f} | {d:.4f} | {e:.4f} |".format(
                m=metric,
                a=vals.get(1, np.nan),
                b=vals.get(2, np.nan),
                c=vals.get(3, np.nan),
                d=vals.get(4, np.nan),
                e=vals.get(5, np.nan),
            )
        )

    lines.extend(
        [
            "",
            "## How to read this",
            "",
            "- If custom is useful, its holdout Spearman should be **higher** than std / downside_std / VaR / CVaR / max_drawdown, and quintiles should rise smoothly from Q1 to Q5.",
            "- A positive delta with p < 0.05 means custom significantly out-predicts that benchmark for that OOS target.",
            "- If deltas are near zero or negative, the extras (loss ratio + streak) are not buying predictive power beyond ordinary downside volatility.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
