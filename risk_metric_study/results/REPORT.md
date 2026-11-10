# Custom Risk Metric Hypothesis Study

## Formula under test

`R = downside_std * (1 + N_neg/N_pos) * (1 + L/5)`

## Design

- Assets: 50
- Daily history length: 2955
- In-sample window: 252 trading days
- Out-of-sample window: 63 trading days
- Walk-forward step: 21 trading days
- Walk-forward dates: 126 (train 75 / holdout 51)

## Hypotheses

- **H0**: `Spearman(custom, OOS_risk) <= Spearman(benchmark, OOS_risk)`
- **H1**: `custom has higher Spearman correlation with OOS risk`

Benefit is defined as **stronger rank correlation** between the in-sample risk score and realized out-of-sample risk (max drawdown, CVaR, volatility).

## Average cross-sectional predictive correlations

### Train period (earlier dates) — target: OOS max drawdown
| metric | avg Spearman rho |
|---|---:|
| std | 0.5799 |
| var_5 | 0.5682 |
| cvar_5 | 0.5528 |
| downside_std | 0.5295 |
| custom_R | 0.5225 |
| max_drawdown | 0.4527 |

### Holdout period (later dates) — target: OOS max drawdown
| metric | avg Spearman rho |
|---|---:|
| std | 0.6342 |
| var_5 | 0.6212 |
| cvar_5 | 0.6162 |
| downside_std | 0.5927 |
| custom_R | 0.5841 |
| max_drawdown | 0.5579 |

### Holdout period — target: OOS CVaR(5%)
| metric | avg Spearman rho |
|---|---:|
| std | 0.7901 |
| cvar_5 | 0.7757 |
| var_5 | 0.7649 |
| downside_std | 0.7592 |
| custom_R | 0.7206 |
| max_drawdown | 0.6632 |

### Holdout period — target: OOS std
| metric | avg Spearman rho |
|---|---:|
| std | 0.8645 |
| cvar_5 | 0.8504 |
| var_5 | 0.8432 |
| downside_std | 0.8276 |
| custom_R | 0.7955 |
| max_drawdown | 0.7376 |

## Bootstrap tests on holdout period

Custom beats benchmark directionally in **3/15** comparisons; statistically significant at 5% one-sided in **3/15**.

| OOS target | benchmark | rho_custom | rho_bench | delta | 95% CI | p(custom better) |
|---|---|---:|---:|---:|---|---:|
| oos_cvar_5 | cvar_5 | 0.6082 | 0.6512 | -0.0431 | [-0.0544, -0.0321] | 1.0000 |
| oos_cvar_5 | downside_std | 0.6082 | 0.6463 | -0.0382 | [-0.0506, -0.0253] | 1.0000 |
| oos_cvar_5 | max_drawdown | 0.6082 | 0.5167 | 0.0914 | [0.0739, 0.1086] | 0.0000 |
| oos_cvar_5 | std | 0.6082 | 0.6603 | -0.0521 | [-0.0659, -0.0387] | 1.0000 |
| oos_cvar_5 | var_5 | 0.6082 | 0.6419 | -0.0338 | [-0.0491, -0.0184] | 1.0000 |
| oos_max_drawdown | cvar_5 | 0.4850 | 0.5082 | -0.0233 | [-0.0356, -0.0112] | 0.9995 |
| oos_max_drawdown | downside_std | 0.4850 | 0.4949 | -0.0100 | [-0.0232, 0.0041] | 0.9170 |
| oos_max_drawdown | max_drawdown | 0.4850 | 0.4405 | 0.0445 | [0.0248, 0.0629] | 0.0000 |
| oos_max_drawdown | std | 0.4850 | 0.5220 | -0.0370 | [-0.0518, -0.0227] | 1.0000 |
| oos_max_drawdown | var_5 | 0.4850 | 0.5174 | -0.0325 | [-0.0487, -0.0158] | 1.0000 |
| oos_std | cvar_5 | 0.6985 | 0.7400 | -0.0415 | [-0.0516, -0.0309] | 1.0000 |
| oos_std | downside_std | 0.6985 | 0.7270 | -0.0285 | [-0.0399, -0.0161] | 1.0000 |
| oos_std | max_drawdown | 0.6985 | 0.6078 | 0.0907 | [0.0747, 0.1074] | 0.0000 |
| oos_std | std | 0.6985 | 0.7498 | -0.0513 | [-0.0635, -0.0388] | 1.0000 |
| oos_std | var_5 | 0.6985 | 0.7362 | -0.0377 | [-0.0517, -0.0231] | 1.0000 |

## Holdout quintiles (avg OOS max drawdown by predicted-risk quintile)

| metric | Q1 (low) | Q2 | Q3 | Q4 | Q5 (high) |
|---|---:|---:|---:|---:|---:|
| custom_R | 0.0646 | 0.0871 | 0.1047 | 0.1171 | 0.1658 |
| cvar_5 | 0.0628 | 0.0867 | 0.1026 | 0.1174 | 0.1698 |
| downside_std | 0.0628 | 0.0883 | 0.1026 | 0.1168 | 0.1687 |
| max_drawdown | 0.0670 | 0.0873 | 0.1045 | 0.1154 | 0.1651 |
| std | 0.0610 | 0.0851 | 0.1035 | 0.1169 | 0.1728 |
| var_5 | 0.0620 | 0.0861 | 0.1039 | 0.1193 | 0.1680 |

## How to read this

- If custom is useful, its holdout Spearman should be **higher** than std / downside_std / VaR / CVaR / max_drawdown, and quintiles should rise smoothly from Q1 to Q5.
- A positive delta with p < 0.05 means custom significantly out-predicts that benchmark for that OOS target.
- If deltas are near zero or negative, the extras (loss ratio + streak) are not buying predictive power beyond ordinary downside volatility.
