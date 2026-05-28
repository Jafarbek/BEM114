"""
ML cross-sectional equity strategy backtest.

Universe: current S&P 500 constituents.
Train window: 2010-01-01 through 2019-12-31.
Test window:  2020-01-01 through 2024-12-31.
Model: LightGBM regressor predicting next-month return.
Portfolio: top 50 long, bottom 50 short, equal-weighted, monthly rebalance.
Benchmark: SPY.

Outputs (written to ./results/):
    metrics.csv            -- summary performance metrics
    metrics.tex            -- same metrics formatted as a LaTeX tabular
    monthly_pnl.csv        -- monthly gross/net returns, turnover, t-cost
    feature_importance.csv -- LightGBM feature importances
    equity_curve.png       -- strategy vs SPY cumulative return + drawdown

Cached data goes to ./cache/ so reruns are fast.
"""

import os
import warnings
from io import StringIO
from urllib.request import Request, urlopen
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ---------- Config ----------
START_DATE = "2009-01-01"      # 1 year of warmup before TRAIN_START
TRAIN_END  = "2019-12-31"
TEST_START = "2020-01-01"
END_DATE   = "2024-12-31"

N_LONG  = 50
N_SHORT = 50
TC_BPS  = 10                   # round-trip transaction cost in bps on turnover

CACHE_DIR   = "cache"
RESULTS_DIR = "results"

os.makedirs(CACHE_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------- Data ----------
def get_sp500_tickers():
    cache = f"{CACHE_DIR}/sp500_tickers.csv"
    if os.path.exists(cache):
        return pd.read_csv(cache)["ticker"].tolist()
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urlopen(req).read().decode("utf-8")
    tickers = pd.read_html(StringIO(html))[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    pd.DataFrame({"ticker": tickers}).to_csv(cache, index=False)
    print(f"Got {len(tickers)} S&P 500 tickers")
    return tickers


def fetch_prices(tickers, start, end):
    cache = f"{CACHE_DIR}/prices.parquet"
    if os.path.exists(cache):
        print("Loading cached prices")
        df = pd.read_parquet(cache)
    else:
        print(f"Downloading prices for {len(tickers)} tickers (this is the slow step)")
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                          group_by="ticker", threads=True, progress=True)
        frames = []
        for t in tickers:
            try:
                sub = raw[t][["Close", "Volume"]].copy()
                sub.columns = ["close", "volume"]
                sub["ticker"] = t
                frames.append(sub.reset_index())
            except (KeyError, TypeError):
                continue
        df = pd.concat(frames, ignore_index=True).dropna(subset=["close"])
        df.to_parquet(cache)
    for cand in ("index", "Datetime", "date"):
        if cand in df.columns and "Date" not in df.columns:
            df = df.rename(columns={cand: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def fetch_spy(start, end):
    cache = f"{CACHE_DIR}/spy.parquet"
    if os.path.exists(cache):
        spy = pd.read_parquet(cache)
    else:
        spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
        spy.to_parquet(cache)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    return spy


# ---------- Features ----------
def compute_features(prices_long, spy_df):
    print("Computing features")
    prices  = prices_long.pivot(index="Date", columns="ticker", values="close").sort_index()
    volumes = prices_long.pivot(index="Date", columns="ticker", values="volume").sort_index()
    dret    = prices.pct_change()
    sret    = spy_df["Close"].pct_change()

    feats = {}
    feats["mom_21"]    = prices.pct_change(21)
    feats["mom_63"]    = prices.pct_change(63)
    feats["mom_126"]   = prices.pct_change(126)
    feats["mom_252"]   = prices.pct_change(252)
    feats["mom_12_1"]  = prices.shift(21) / prices.shift(252) - 1      # 12-1 momentum
    feats["vol_21"]    = dret.rolling(21).std() * np.sqrt(252)
    feats["vol_63"]    = dret.rolling(63).std() * np.sqrt(252)
    feats["rev_5"]     = prices.pct_change(5)
    feats["dist_high"] = prices / prices.rolling(252).max() - 1
    feats["vol_ratio"] = volumes.rolling(20).mean() / volumes.rolling(252).mean()
    feats["skew_63"]   = dret.rolling(63).skew()
    rcov = dret.rolling(252).cov(sret)
    rvar = sret.rolling(252).var()
    feats["beta"]      = rcov.div(rvar, axis=0)

    # Stack features into long format
    long_pieces = [df.stack().rename(name) for name, df in feats.items()]
    panel = pd.concat(long_pieces, axis=1).reset_index()
    panel.columns = ["Date", "ticker"] + list(feats.keys())
    panel["Date"] = pd.to_datetime(panel["Date"])

    # Keep last trading day of each (ticker, month) only
    panel["month"] = panel["Date"].dt.to_period("M")
    last_dt = panel.groupby(["ticker", "month"])["Date"].transform("max")
    panel = panel[panel["Date"] == last_dt].copy()

    # Forward monthly return as target: this month-end -> next month-end
    monthly_close = prices.resample("ME").last()
    fwd_ret = monthly_close.pct_change().shift(-1)
    fwd_long = fwd_ret.stack().rename("fwd_return").reset_index()
    fwd_long["month"] = pd.to_datetime(fwd_long["Date"]).dt.to_period("M")
    fwd_long = fwd_long.drop(columns="Date")

    panel = panel.merge(fwd_long, on=["ticker", "month"], how="left").drop(columns="month")
    print(f"Feature panel: {len(panel):,} stock-month rows, {len(feats)} features")
    return panel, list(feats.keys())


def cross_sectional_normalize(panel, feature_cols):
    """Z-score within each cross-section and winsorize at +/- 3 sigma."""
    for c in feature_cols:
        panel[c] = panel.groupby("Date")[c].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() and not np.isnan(x.std()) else x * 0
        )
        panel[c] = panel[c].clip(-3, 3)
    return panel


# ---------- Model ----------
def train_model(panel, feature_cols, train_end):
    train = panel[(panel["Date"] <= train_end)].dropna(subset=feature_cols + ["fwd_return"])
    X, y = train[feature_cols], train["fwd_return"]
    print(f"Training LightGBM on {len(X):,} stock-months")
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


# ---------- Backtest ----------
def backtest(panel, model, feature_cols, test_start, long_gross=1.0, short_gross=1.0):
    """Generic long-short backtest. Set short_gross=0 for long-only.
    Spec strategy is long_gross=1, short_gross=1 (dollar-neutral, gross 200%)."""
    test = panel[panel["Date"] >= test_start].dropna(subset=feature_cols + ["fwd_return"]).copy()
    test["pred"] = model.predict(test[feature_cols])
    rows, prior = [], pd.Series(dtype=float)
    for date in sorted(test["Date"].unique()):
        snap = test[test["Date"] == date].sort_values("pred", ascending=False)
        if len(snap) < N_LONG + (N_SHORT if short_gross > 0 else 0):
            continue
        longs  = snap.head(N_LONG)["ticker"].tolist()
        shorts = snap.tail(N_SHORT)["ticker"].tolist() if short_gross > 0 else []
        idx = longs + shorts
        w = pd.Series(0.0, index=idx)
        w[longs]  =  long_gross  / N_LONG
        if short_gross > 0:
            w[shorts] = -short_gross / N_SHORT

        snap_idx = snap.set_index("ticker")
        port_ret = (w * snap_idx["fwd_return"].reindex(w.index).fillna(0)).sum()

        all_idx  = w.index.union(prior.index)
        turnover = (w.reindex(all_idx, fill_value=0) - prior.reindex(all_idx, fill_value=0)).abs().sum()
        # 10 bps round-trip applied to one-way notional traded: Σ|Δw|/2.
        tcost    = (turnover / 2) * TC_BPS / 10000

        rows.append({
            "Date": date, "gross_return": port_ret, "tcost": tcost,
            "net_return": port_ret - tcost, "turnover": turnover,
        })
        prior = w
    return pd.DataFrame(rows).set_index("Date")


# ---------- Metrics ----------
def compute_metrics(strat, spy_m, turnover=None):
    common = strat.index.intersection(spy_m.index)
    s, b   = strat.loc[common], spy_m.loc[common]
    excess = s - b
    eq     = (1 + s).cumprod()

    res = linregress(b.values, s.values)
    alpha_se = res.intercept_stderr
    metrics = {
        "Strategy Annual Return": s.mean() * 12,
        "Strategy Annual Vol":    s.std() * np.sqrt(12),
        "Strategy Sharpe":        s.mean() / s.std() * np.sqrt(12),
        "SPY Annual Return":      b.mean() * 12,
        "SPY Annual Vol":         b.std() * np.sqrt(12),
        "SPY Sharpe":             b.mean() / b.std() * np.sqrt(12),
        "Excess Return (annual)": excess.mean() * 12,
        "Tracking Error":         excess.std() * np.sqrt(12),
        "Information Ratio":      excess.mean() / excess.std() * np.sqrt(12),
        "CAPM Alpha (annual)":    res.intercept * 12,
        "Beta to SPY":            res.slope,
        "Alpha t-stat":           res.intercept / alpha_se if alpha_se > 0 else float("nan"),
        "Max Drawdown":           (eq / eq.cummax() - 1).min(),
        "Hit Rate vs SPY":        (excess > 0).mean(),
    }
    if turnover is not None:
        metrics["Average Turnover"] = float(turnover.mean())
    return metrics


def metrics_to_latex(metrics, path):
    lines = [r"\begin{tabular}{lr}", r"\hline", r"Metric & Value \\", r"\hline"]
    for k, v in metrics.items():
        lines.append(f"{k} & {v:.4f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def plot_results(strat, spy_m, path):
    common = strat.index.intersection(spy_m.index)
    s, b   = strat.loc[common], spy_m.loc[common]
    s_eq, b_eq = (1 + s).cumprod(), (1 + b).cumprod()
    dd     = s_eq / s_eq.cummax() - 1

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(s_eq.index, s_eq.values, label="Strategy (net of costs)", lw=2)
    axes[0].plot(b_eq.index, b_eq.values, label="SPY", lw=2)
    axes[0].set_title("Cumulative Return: Strategy vs SPY")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].fill_between(dd.index, dd.values, 0, alpha=0.5, color="crimson")
    axes[1].set_title("Strategy Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")


def plot_variants(variants_pnl, spy_m, path):
    """Plot cumulative returns of all strategy variants vs SPY on one panel."""
    common = None
    for pnl in variants_pnl.values():
        idx = pnl.index.intersection(spy_m.index)
        common = idx if common is None else common.intersection(idx)
    b_eq = (1 + spy_m.loc[common]).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, pnl in variants_pnl.items():
        eq = (1 + pnl["net_return"].loc[common]).cumprod()
        ax.plot(eq.index, eq.values, label=name, lw=2)
    ax.plot(b_eq.index, b_eq.values, label="SPY", lw=2, ls="--", color="black")
    ax.set_title("Strategy Variants vs SPY (net of costs)")
    ax.set_ylabel("Growth of $1")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")


# ---------- Driver ----------
def main():
    tickers = get_sp500_tickers()
    prices  = fetch_prices(tickers, START_DATE, END_DATE)
    spy     = fetch_spy(START_DATE, END_DATE)

    panel, feature_cols = compute_features(prices, spy)
    panel = cross_sectional_normalize(panel, feature_cols)

    model = train_model(panel, feature_cols, TRAIN_END)

    imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(f"{RESULTS_DIR}/feature_importance.csv", index=False)
    print("\nFeature importance:"); print(imp.to_string(index=False))

    pnl = backtest(panel, model, feature_cols, TEST_START)
    pnl.to_csv(f"{RESULTS_DIR}/monthly_pnl.csv")

    spy_m = spy["Close"].resample("ME").last().pct_change().dropna()
    if isinstance(spy_m, pd.DataFrame):
        spy_m = spy_m.iloc[:, 0]

    # ---------- Variant strategies (extensions, not the spec) ----------
    variants = {
        "Spec L-S (100/100, net 0)":   (1.0, 1.0),
        "Long-only (100/0)":            (1.0, 0.0),
        "130/30 (net +100)":            (1.3, 0.3),
        "150/50 (net +100, gross 200)": (1.5, 0.5),
    }
    variants_pnl, variants_metrics = {}, {}
    for name, (lg, sg) in variants.items():
        if name.startswith("Spec"):
            v_pnl = pnl
        else:
            v_pnl = backtest(panel, model, feature_cols, TEST_START, long_gross=lg, short_gross=sg)
        variants_pnl[name] = v_pnl
        variants_metrics[name] = compute_metrics(v_pnl["net_return"], spy_m, turnover=v_pnl["turnover"])

    print("\n=== Variant Comparison (annualized) ===")
    summary_rows = ["Variant                          Ret    Vol  Sharpe   Alpha  t-stat  MaxDD  Turn"]
    for name, m in variants_metrics.items():
        summary_rows.append(
            f"{name:32s} {m['Strategy Annual Return']*100:5.1f}% {m['Strategy Annual Vol']*100:5.1f}% "
            f"{m['Strategy Sharpe']:6.2f} {m['CAPM Alpha (annual)']*100:5.1f}% {m['Alpha t-stat']:6.2f} "
            f"{m['Max Drawdown']*100:5.1f}% {m['Average Turnover']:4.2f}"
        )
    print("\n".join(summary_rows))
    pd.DataFrame(variants_metrics).to_csv(f"{RESULTS_DIR}/variants_metrics.csv")
    plot_variants(variants_pnl, spy_m, f"{RESULTS_DIR}/variants_equity.png")

    metrics = compute_metrics(pnl["net_return"], spy_m, turnover=pnl["turnover"])
    print("\n=== Performance Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v:>10.4f}")

    pd.Series(metrics, name="value").to_csv(f"{RESULTS_DIR}/metrics.csv")
    metrics_to_latex(metrics, f"{RESULTS_DIR}/metrics.tex")
    plot_results(pnl["net_return"], spy_m, f"{RESULTS_DIR}/equity_curve.png")
    print(f"\nResults saved to ./{RESULTS_DIR}/")


if __name__ == "__main__":
    main()
