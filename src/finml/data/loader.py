from __future__ import annotations
import pandas as pd
import yfinance as yf
from ..utils.config import Paths
from ..utils.io import ensure_dir

def fetch_prices(tickers: list[str], start: str, end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    # yfinance returns MultiIndex columns when multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(level=1).rename_axis(index=["Date", "Ticker"]).reset_index()
    else:
        df["Ticker"] = tickers[0]
        df = df.reset_index().rename(columns={"index": "Date"})
    return df

def save_raw_prices(df: pd.DataFrame, paths: Paths = Paths()) -> str:
    ensure_dir(paths.data_raw)
    out = f"{paths.data_raw}/prices.csv"
    df.to_csv(out, index=False)
    return out
