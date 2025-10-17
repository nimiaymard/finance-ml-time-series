import numpy as np
import pandas as pd
from scipy.stats import norm

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252):
    ex_ret = returns - risk_free/periods_per_year
    mu = ex_ret.mean()*periods_per_year
    sigma = ex_ret.std()*np.sqrt(periods_per_year)
    return float(mu / sigma) if sigma != 0 else np.nan

def var_parametric(returns: pd.Series, alpha: float = 0.95):
    mu, sigma = returns.mean(), returns.std()
    z = norm.ppf(1 - alpha)
    return float(-(mu + z * sigma))

def var_historical(returns: pd.Series, alpha: float = 0.95):
    return float(-np.quantile(returns.dropna(), 1 - alpha))
