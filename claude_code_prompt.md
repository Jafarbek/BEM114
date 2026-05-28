# ML Cross-Sectional Equity Strategy Backtest

Build and run a complete backtest of a machine learning long-short equity strategy on the S&P 500, with SPY as the benchmark. This is for a hedge fund class project, results are due in roughly 5 hours, so prioritize a clean end-to-end run over polish.

## Strategy

Train a LightGBM regressor on cross-sectional features to predict next-month returns for current S&P 500 constituents. Each month in the test window, go long the top 50 predicted returns and short the bottom 50, equal-weighted within each leg, gross exposure 200%, net exposure 0. Rebalance monthly. Charge 10 bps round-trip transaction cost on turnover.

Single train/test split. No rolling windows, no cross-validation. Train on 2010-01-01 through 2019-12-31. Test on 2020-01-01 through 2024-12-31.

## Data

Pull the current S&P 500 ticker list from the first table at `https://en.wikipedia.org/wiki/List_of_S%26P_500_companies`. Replace dots with hyphens in symbols so they work with yfinance (BRK.B becomes BRK-B).

Use yfinance to download daily OHLCV from 2009-01-01 through 2024-12-31 for all constituents. The extra year at the front is warmup for 252-day features. Also pull SPY for the same range as the benchmark. Cache everything to `./cache/` as parquet so reruns are fast. The bulk yfinance pull is the bottleneck, expect 15 to 45 minutes for ~500 tickers over 15 years.

## Features

Compute these at daily frequency on the wide-format price panel, then snapshot at the last trading day of each month:

- Momentum at 21, 63, 126, 252 trading days
- 12-1 momentum: 12-month return excluding the most recent month
- Realized volatility, annualized, over 21 and 63 days
- 5-day reversal (short-term reversal signal)
- Distance from trailing 252-day high
- Volume ratio: 20-day mean volume divided by 252-day mean volume
- 63-day return skewness
- Rolling 252-day beta to SPY

After snapshotting, cross-sectionally z-score each feature within each date and winsorize at plus or minus 3 sigma. This step is important. Without it the model picks up regime drift instead of relative ranking.

## Target

Next month's return, computed month-end-to-month-end on the actual close series (not as a 21-day forward return). It must be strictly forward-looking. Do not include it or any function of it in the features.

## Model

LightGBM regressor with: `n_estimators=500`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=100`, `feature_fraction=0.9`, `random_state=42`, `n_jobs=-1`. Train once on the full training window, predict on all test-period stock-months.

## Outputs (all to `./results/`)

- `metrics.csv`: strategy annual return, annual vol, Sharpe; SPY annual return, vol, Sharpe; excess return; tracking error; information ratio; CAPM alpha (annualized) and beta; alpha t-statistic; max drawdown; hit rate vs SPY; average turnover.
- `metrics.tex`: the same metrics formatted as a LaTeX `tabular` block with `lr` alignment and `\hline` rules, ready to drop into a proposal via `\input{}`.
- `monthly_pnl.csv`: dated monthly gross return, net return, transaction cost, turnover.
- `feature_importance.csv`: feature names and LightGBM importances ranked descending.
- `equity_curve.png`: two stacked panels, cumulative return strategy vs SPY on top, strategy drawdown on bottom.

Also print all metrics to stdout.

## Reference implementation

A working reference is at `./backtest.py`. Read it first. You can run it directly, adapt it, or rewrite from scratch, whichever is fastest. Same outputs either way.

## Setup

```
pip install yfinance lightgbm pandas numpy matplotlib scipy pyarrow lxml
python backtest.py
```

## Verification before reporting back

Sanity checks once the backtest finishes:

1. Information ratio vs SPY should land roughly in [-1, 2]. Anything outside that range is suspicious.
2. Beta to SPY should be close to zero since the portfolio is long-short, but plus or minus 0.3 is fine.
3. Max drawdown should be deep enough to be believable, typically -10% to -40% for long-short.
4. Feature importance should have momentum and volatility ranked high.
5. If CAPM alpha exceeds 20% per year annualized, double-check for look-ahead bias. Manually inspect a few stock-month rows from the test period and confirm features at date t use only information through t, and `fwd_return` is the next month's return.

If anything looks suspicious, print diagnostics and stop before generating final outputs.

## Known limitations to note in any writeup

- Survivorship bias: current S&P 500 constituents only, no delisted firms. Reported performance is an upper bound on what would obtain in a delisting-adjusted universe.
- Yahoo fundamentals are not point-in-time, which is why no fundamental features are used.
- No borrow costs modeled on the short leg.

## Report back with

- The full metrics table (paste from stdout).
- A one-sentence read on statistical significance (alpha t-stat > 2 means significant at 95%).
- Path to `equity_curve.png`.
- Any anomalies, concerns, or bugs found.
- Suggested next steps if results look weak (e.g., more features, hyperparameter tuning, different universe).

Don't summarize the code, I'll read it directly.
