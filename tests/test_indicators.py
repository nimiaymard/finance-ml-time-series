import pandas as pd
from src.finml.features.indicators import add_returns
from pytest import approx

def test_add_returns():
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=3, freq="D"),
        "Ticker": ["AAPL"]*3,
        "Close": [100, 101, 102]
    })
    res = add_returns(df)
    assert "Return" in res.columns
    assert res["Return"].iloc[1] == approx(0.01, rel=1e-12, abs=1e-12)
