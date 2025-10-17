from __future__ import annotations
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from ..utils.io import save_pickle

def fit_arima(series: pd.Series, order=(5,1,0)):
    model = SARIMAX(series, order=order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res

def forecast_arima(res, steps: int = 1) -> pd.Series:
    return res.forecast(steps=steps)

def save_arima(res, path: str) -> None:
    save_pickle(res, path)
