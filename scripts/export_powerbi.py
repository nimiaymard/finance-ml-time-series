import argparse, pandas as pd, numpy as np
from src.finml.utils.config import Paths
from src.finml.evaluation.risk import sharpe_ratio, var_historical, var_parametric

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--target", default="Close")
    args = ap.parse_args()

    paths = Paths()
    prices = pd.read_csv(f"{paths.data_raw}/prices.csv", parse_dates=["Date"])
    prices = prices[prices["Ticker"].isin(args.tickers)].sort_values(["Ticker","Date"])

    # Pivot prices: Date x Ticker
    pivot = prices.pivot(index="Date", columns="Ticker", values=args.target).dropna()
    returns = pivot.pct_change().dropna()
    eq_weights = np.repeat(1/len(args.tickers), len(args.tickers))
    port_ret = (returns * eq_weights).sum(axis=1)
    equity = (1 + port_ret).cumprod()

    ds_dir = paths.powerbi_ds
    import os
    os.makedirs(ds_dir, exist_ok=True)

    pivot.to_csv(f"{ds_dir}/prices.csv")
    returns.to_csv(f"{ds_dir}/returns.csv")

    signals = pd.DataFrame({
        "Date": returns.index,
        "Signal": np.sign(port_ret).values  # dummy signal
    })
    signals.to_csv(f"{ds_dir}/signals.csv", index=False)

    portfolio = pd.DataFrame({
        "Date": equity.index,
        "Equity": equity.values,
        "Return": port_ret.values
    })
    # Risk metrics (last 1Y if available)
    last = returns.tail(min(252, len(returns)))
    sr = sharpe_ratio(last.mean(axis=1))  # per-day to annualized inside fn
    var_h = var_historical(last.mean(axis=1))
    var_p = var_parametric(last.mean(axis=1))
    portfolio["Sharpe"] = sr
    portfolio["VaR_hist"] = var_h
    portfolio["VaR_param"] = var_p
    portfolio.to_csv(f"{ds_dir}/portfolio.csv", index=False)

    print("Exported datasets to powerbi/datasets/")
