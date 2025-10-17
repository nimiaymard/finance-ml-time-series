import argparse, pandas as pd
from src.finml.utils.config import Paths
from src.finml.models.lstm_model import train_lstm
from src.finml.utils.io import ensure_dir, save_pickle

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--target", default="Close")
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    paths = Paths()
    df = pd.read_csv(f"{paths.data_raw}/prices.csv", parse_dates=["Date"])
    df = df[df["Ticker"] == args.ticker].sort_values("Date")
    model, scaler = train_lstm(df[args.target], seq_len=args.seq_len, epochs=args.epochs)
    ensure_dir(paths.data_models)
    save_pickle(model, f"{paths.data_models}/lstm_{args.ticker}_{args.target}.pkl")
    save_pickle(scaler, f"{paths.data_models}/scaler_{args.ticker}_{args.target}.pkl")
    print("LSTM trained and saved.")
