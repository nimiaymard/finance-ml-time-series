import pandas as pd
import numpy as np

def add_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    df = df.sort_values(["Ticker", "Date"]).copy()
    df["Return"] = df.groupby("Ticker")[price_col].pct_change()
    df["LogReturn"] = np.log1p(df["Return"])
    df["Volatility20"] = df.groupby("Ticker")["LogReturn"].rolling(20).std().reset_index(level=0, drop=True) * np.sqrt(252)
    return df

def make_supervised(df: pd.DataFrame, target_col: str = "Close", lags: int = 1) -> pd.DataFrame:
    df = df.sort_values(["Ticker", "Date"]).copy()
    for i in range(1, lags + 1):
        df[f"{target_col}_lag{i}"] = df.groupby("Ticker")[target_col].shift(i)
    df[f"{target_col}_t+1"] = df.groupby("Ticker")[target_col].shift(-1)
    return df.dropna()
