import argparse, pandas as pd
from src.finml.data.loader import fetch_prices, save_raw_prices

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", default=None)
    p.add_argument("--interval", default="1d")
    args = p.parse_args()

    df = fetch_prices(args.tickers, start=args.start, end=args.end, interval=args.interval)
    path = save_raw_prices(df)
    print(f"Saved: {path}")
