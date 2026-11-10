# Custom Risk Metric Hypothesis Study

Tests whether

`R = downside_std * (1 + N_neg/N_pos) * (1 + L/5)`

predicts future realized risk better than standard metrics
(std, downside std, max drawdown, VaR, CVaR).

## Run

```bash
cd risk_metric_study
pip install -r requirements.txt
python run_hypothesis_study.py
```

Results land in `results/`, including `REPORT.md`.
