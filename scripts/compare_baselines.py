import argparse, os
import numpy as np
import pandas as pd
from src.finml.utils.config import Paths
from src.finml.backtest.backtest import Backtester
from src.finml.evaluation.metrics import rmse, mae

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    # éviter les divisions par ~0 (prix très petits)
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)

def main():
    ap = argparse.ArgumentParser(description="Comparer baseline naïve vs ARIMA sur un ticker.")
    ap.add_argument("--ticker", required=True, help="Ex: AAPL")
    ap.add_argument("--target", default="Close", help='Colonne cible (ex: "Close" ou "Adj Close")')
    ap.add_argument("--train-size", type=float, default=0.8, help="Ratio d'entraînement (défaut 0.8)")
    ap.add_argument("--order", nargs=3, type=int, default=[5,1,0], help="Ordre ARIMA p d q (défaut 5 1 0)")
    ap.add_argument("--plot", action="store_true", help="Sauvegarde une figure de comparaison")
    args = ap.parse_args()

    paths = Paths()
    data_path = f"{paths.data_raw}/prices.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} introuvable. Lance d'abord: python -m scripts.fetch_data ...")

    # 1) Charger données et filtrer le ticker
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df[df["Ticker"] == args.ticker].sort_values("Date").reset_index(drop=True)
    if args.target not in df.columns:
        raise KeyError(f'Colonne cible "{args.target}" absente. Colonnes dispo: {list(df.columns)}')

    # 2) Split temporel identique aux autres scripts
    n = len(df)
    split = int(n * args.train_size)
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
    if len(test) < 2:
        raise ValueError("Jeu de test trop court après split. Réduis --train-size.")

    # 3) Baseline naïve = persistance (prédit le prix d’hier)
    #    Pour que la 1re prédiction test utilise le *dernier* prix du train,
    #    on décale sur la série complète puis on récupère la partie test.
    full = df[[ "Date", args.target ]].copy()
    full["Naive"] = full[args.target].shift(1)
    naive = full.iloc[split:][["Date", "Naive"]].copy()

    # 4) ARIMA via Backtester (n’importe pas TensorFlow)
    bt = Backtester(train_size=args.train_size, seq_len=60, arima_order=tuple(args.order))
    arima_res = bt.walk_forward(df[["Date", args.target]].copy(), target_col=args.target, model="arima")
    arima = arima_res[["Date", "Prediction"]].rename(columns={"Prediction": "ARIMA"})

    # 5) Fusion des résultats sur la période test
    out = test[["Date", args.target]].copy()
    out = out.merge(naive, on="Date", how="left")
    out = out.merge(arima, on="Date", how="left")

    # drop la 1re ligne test qui n’a pas de naïve (car shift)
    out = out.dropna(subset=["Naive"])

    # 6) Métriques
    y_true = out[args.target].values
    y_naive = out["Naive"].values
    y_arima = out["ARIMA"].values

    metrics = {
        "NAIVE_RMSE": rmse(y_true, y_naive),
        "NAIVE_MAE":  mae(y_true, y_naive),
        "NAIVE_MAPE": mape(y_true, y_naive),
        "ARIMA_RMSE": rmse(y_true, y_arima),
        "ARIMA_MAE":  mae(y_true, y_arima),
        "ARIMA_MAPE": mape(y_true, y_arima),
    }

    # 7) Sauvegardes
    os.makedirs(paths.data_processed, exist_ok=True)
    compare_csv = f"{paths.data_processed}/compare_{args.ticker}_naive_arima.csv"
    out.to_csv(compare_csv, index=False)
    print("Saved comparison CSV ->", compare_csv)

    print("\n=== Metrics (test set) ===")
    print(f"NAIVE  -> RMSE: {metrics['NAIVE_RMSE']:.6f} | MAE: {metrics['NAIVE_MAE']:.6f} | MAPE: {metrics['NAIVE_MAPE']:.3f}%")
    print(f"ARIMA  -> RMSE: {metrics['ARIMA_RMSE']:.6f} | MAE: {metrics['ARIMA_MAE']:.6f} | MAPE: {metrics['ARIMA_MAPE']:.3f}%")

    # 8) (Optionnel) Figure de comparaison
    if args.plot:
        import matplotlib.pyplot as plt
        os.makedirs(paths.reports_fig, exist_ok=True)
        fig_path = f"{paths.reports_fig}/compare_{args.ticker}_naive_arima.png"
        plt.figure(figsize=(10,5))
        plt.plot(out["Date"], out[args.target], label="Actual")
        plt.plot(out["Date"], out["Naive"], label="Naive")
        plt.plot(out["Date"], out["ARIMA"], label="ARIMA")
        plt.title(f"Comparison — {args.ticker} — target={args.target}")
        plt.xlabel("Date"); plt.ylabel(args.target); plt.legend(); plt.tight_layout()
        plt.savefig(fig_path, dpi=144)
        print("Saved comparison figure ->", fig_path)

if __name__ == "__main__":
    main()
