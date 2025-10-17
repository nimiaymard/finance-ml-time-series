import argparse, pandas as pd
from src.finml.utils.config import Paths
from src.finml.models.arima_model import fit_arima
from src.finml.utils.io import ensure_dir, save_pickle

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--target", default="Close")
    ap.add_argument("--order", nargs=3, type=int, default=[5,1,0])
    args = ap.parse_args()

    paths = Paths()
    df = pd.read_csv(f"{paths.data_raw}/prices.csv", parse_dates=["Date"])
    df = df[df["Ticker"] == args.ticker].sort_values("Date")
    res = fit_arima(df[args.target], order=tuple(args.order))
    ensure_dir(paths.data_models)
    save_pickle(res, f"{paths.data_models}/arima_{args.ticker}_{args.target}.pkl")
    print("ARIMA trained and saved.")
