import argparse, os
import pandas as pd
from src.finml.utils.config import Paths
from src.finml.backtest.backtest import Backtester
from src.finml.evaluation.metrics import rmse, mae

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--target", default="Close")
    ap.add_argument("--model", choices=["arima","lstm"], default="arima")
    ap.add_argument("--order", nargs=3, type=int, default=[5,1,0], help="Ordre ARIMA p d q (défaut 5 1 0)")
    ap.add_argument("--train-size", type=float, default=0.7)
    ap.add_argument("--seq-len", type=int, default=60)
    args = ap.parse_args()

    paths = Paths()
    # IMPORTANT: parser les dates ici
    df = pd.read_csv(f"{paths.data_raw}/prices.csv", parse_dates=["Date"])
    df = df[df["Ticker"] == args.ticker].sort_values("Date")

    # Backtester configuré
    bt = Backtester(train_size=args.train_size, seq_len=args.seq_len, arima_order=tuple(args.order))
    res = bt.walk_forward(df, target_col=args.target, model=args.model)

    # Sauvegarde & métriques
    out_path = f"{paths.data_processed}/backtest_{args.ticker}_{args.model}.csv"
    os.makedirs(paths.data_processed, exist_ok=True)
    res.to_csv(out_path, index=False)
    print("Saved backtest CSV ->", out_path)
    print("RMSE:", rmse(res[args.target], res["Prediction"]))
    print("MAE :", mae(res[args.target], res["Prediction"]))
