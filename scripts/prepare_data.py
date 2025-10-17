
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_data.py — affichage & préparation des données avant entraînement

Usage (à exécuter depuis la racine du projet) :
  python -m scripts.fetch_data --tickers AAPL MSFT --start 2018-01-01 --end 2025-10-01
  python prepare_data.py --ticker AAPL --target Close --start 2018-01-01 --end 2025-10-01 --winsor 0.01 --impute ffill

Place ensuite ce fichier dans la racine de ton repo ou dans scripts/ (et adapte l'import des Paths).
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Option 2 (autonome) : rendre le script utilisable sans dépendre du module src ---
class Paths:
    data_raw: str = "data/raw"
    data_interim: str = "data/interim"
    data_processed: str = "data/processed"
    data_models: str = "data/models"
    reports_fig: str = "data/reports/figures"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def add_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    df = df.sort_values(["Ticker", "Date"]).copy()
    df["Return"] = df.groupby("Ticker")[price_col].pct_change()
    df["LogReturn"] = np.log1p(df["Return"])
    df["Volatility20"] = (
        df.groupby("Ticker")["LogReturn"]
          .rolling(20).std().reset_index(level=0, drop=True) * np.sqrt(252)
    )
    return df

# ------------------ fonctions utilitaires ------------------
def validate_columns(df: pd.DataFrame, required=("Date", "Ticker", "Close")) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}. Colonnes disponibles: {list(df.columns)}")

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def winsorize(series: pd.Series, alpha: float = 0.01) -> pd.Series:
    if alpha <= 0: 
        return series
    lo, hi = series.quantile(alpha), series.quantile(1 - alpha)
    return series.clip(lower=lo, upper=hi)

def impute_missing(df: pd.DataFrame, cols: list[str], how: str = "ffill") -> pd.DataFrame:
    if how == "none":
        return df
    if how == "ffill":
        df[cols] = df.groupby("Ticker")[cols].ffill()
    elif how == "bfill":
        df[cols] = df.groupby("Ticker")[cols].bfill()
    else:
        raise ValueError("Valeur de --impute invalide. Choisir parmi: none, ffill, bfill")
    return df

def overview(df: pd.DataFrame, ticker: Optional[str], target: str, paths: Paths) -> None:
    print("\n=== Aperçu des données ===")
    print(df.head(10).to_string(index=False))
    print("\nShape:", df.shape)
    print("\nColonnes:", list(df.columns))
    print("\nStatistiques de base (sur la cible) :")
    print(df[target].describe())
    # petit graphique (sauvegardé) : prix
    ensure_dir(paths.reports_fig)
    try:
        fig_path = os.path.join(paths.reports_fig, f"overview_{ticker or 'ALL'}.png")
        plt.figure()
        df_plot = df.sort_values("Date")
        plt.plot(df_plot["Date"], df_plot[target])
        plt.title(f"{ticker or 'ALL'} - {target}")
        plt.xlabel("Date"); plt.ylabel(target)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=144)
        print(f"\nFigure sauvegardée: {fig_path}")
    except Exception as e:
        print(f"[WARN] Graphique non généré: {e}")

def prepare(
    ticker: Optional[str] = None,
    target: str = "Close",
    start: Optional[str] = None,
    end: Optional[str] = None,
    impute: str = "ffill",
    winsor: float = 0.0,
    save: bool = True,
    paths: Paths = Paths(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Chargement
    raw_path = os.path.join(paths.data_raw, "prices.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Fichier non trouvé: {raw_path}. Lance d'abord fetch_data.py.")
    df = pd.read_csv(raw_path)
    validate_columns(df, required=("Date", "Ticker", target))
    df = parse_dates(df)

    # Filtrage ticker + dates
    if ticker:
        df = df[df["Ticker"] == ticker].copy()
    if start:
        df = df[df["Date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["Date"] <= pd.to_datetime(end)]

    # Tri & doublons
    before = len(df)
    df = df.drop_duplicates(subset=["Date", "Ticker"]).sort_values(["Ticker", "Date"])
    after = len(df)
    if after < before:
        print(f"[INFO] Doublons supprimés: {before - after} lignes")

    # Imputation de base sur la cible (optionnel)
    df = impute_missing(df, cols=[target], how=impute)

    # Winsorisation légère de la cible (anti-outliers) si demandé
    if winsor and winsor > 0:
        df[target] = winsorize(df[target], alpha=winsor)

    # Ajout de features
    feat = add_returns(df[["Date", "Ticker", target]].copy(), price_col=target)

    # Drop NaN en début de série (à cause des returns/rolling)
    clean = feat.dropna().reset_index(drop=True)

    # Affichage / aperçu
    overview(clean, ticker=ticker, target=target, paths=paths)

    # Sauvegardes
    if save:
        ensure_dir(paths.data_interim)
        ensure_dir(paths.data_processed)
        clean_path = os.path.join(paths.data_interim, f"{ticker or 'ALL'}_clean.csv")
        feat_path  = os.path.join(paths.data_processed, f"{ticker or 'ALL'}_features.csv")
        clean.to_csv(clean_path, index=False)
        clean.to_csv(feat_path, index=False)
        print(f"\nSauvegardé : {clean_path}")
        print(f"Sauvegardé : {feat_path}")

    return df, clean

def split_train_test(df: pd.DataFrame, ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split = int(n * ratio)
    train, test = df.iloc[:split], df.iloc[split:]
    print(f"\nSplit temporel — train: {train.shape}, test: {test.shape}")
    print(f"Train range: {train['Date'].min().date()} → {train['Date'].max().date()}")
    print(f"Test  range: {test['Date'].min().date()} → {test['Date'].max().date()}")
    return train, test

def main():
    ap = argparse.ArgumentParser(description="Affichage & préparation des données avant entraînement")
    ap.add_argument("--ticker", type=str, default=None, help="Ticker à filtrer (ex: AAPL)")
    ap.add_argument("--target", type=str, default="Close", help="Colonne cible (Close, Adj Close...)")
    ap.add_argument("--start", type=str, default=None, help="Date de début (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default=None, help="Date de fin (YYYY-MM-DD)")
    ap.add_argument("--impute", type=str, default="ffill", choices=["none", "ffill", "bfill"], help="Imputation des valeurs manquantes de la cible")
    ap.add_argument("--winsor", type=float, default=0.0, help="Winsorisation anti-outliers (ex: 0.01)")
    ap.add_argument("--split", type=float, default=0.8, help="Ratio train (0-1) pour split temporel")
    ap.add_argument("--no-save", action="store_true", help="Ne pas sauvegarder les fichiers nettoyés")
    args = ap.parse_args()

    _, clean = prepare(
        ticker=args.ticker,
        target=args.target,
        start=args.start,
        end=args.end,
        impute=args.impute,
        winsor=args.winsor,
        save=not args.no_save,
        paths=Paths(),
    )

    # Split train/test et sauvegarde optionnelle
    train, test = split_train_test(clean, ratio=args.split)

    if not args.no_save:
        ensure_dir(Paths().data_processed)
        tkr = args.ticker or "ALL"
        train.to_csv(os.path.join(Paths().data_processed, f"{tkr}_train.csv"), index=False)
        test.to_csv(os.path.join(Paths().data_processed, f"{tkr}_test.csv"), index=False)
        print(f"Sauvegardé : data/processed/{tkr}_train.csv")
        print(f"Sauvegardé : data/processed/{tkr}_test.csv")

if __name__ == "__main__":
    main()
