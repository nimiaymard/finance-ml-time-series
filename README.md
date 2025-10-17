# Finance ML Time Series — Modélisation financière (ARIMA, LSTM, VaR, Sharpe) & Power BI

Projet pédagogique et reproductible pour :
- Télécharger des **séries de prix** (yfinance)
- Construire des **features** (retours, volatilité, indicateurs simples)
- Entrainer des **modèles** ARIMA & LSTM pour la **prédiction de prix / rendements**
- Évaluer & backtester (MSE/RMSE, **Value-at-Risk**, **Sharpe Ratio**)
- **Exporter des datasets** prêts pour un **rapport Power BI** (visualisation interactive)

## 🚀 Démarrage rapide

```bash
# 1) Créer l'environnement
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
# .venv\Scripts\activate                           # (Windows)

pip install -r requirements.txt

# 2) Télécharger des données (par ex. AAPL, MSFT)
python scripts/fetch_data.py --tickers AAPL MSFT --start 2018-01-01 --end 2025-10-01

# 3) Entraîner ARIMA sur AAPL (clôture)
python scripts/train_arima.py --ticker AAPL --target Close --order 5 1 0

# 4) Entraîner LSTM sur AAPL
python scripts/train_lstm.py --ticker AAPL --target Close --seq-len 60 --epochs 5

# 5) Backtest (walk-forward) & métriques
python scripts/backtest.py --ticker AAPL --model arima --target Close

# 6) Export pour Power BI (portefeuille égal-pondéré AAPL/MSFT)
python scripts/export_powerbi.py --tickers AAPL MSFT --target Close
```

Le dossier `powerbi/` contient des CSV à **charger dans Power BI Desktop**. Vous pouvez créer vos pages (rendements, drawdown, VaR, Sharpe, prédictions vs réel, signaux de trading, etc.).

## 📦 Structure du projet

```
finance-ml-time-series/
├─ README.md
├─ requirements.txt
├─ environment.yml
├─ .gitignore
├─ Makefile
├─ src/finml/
│  ├─ __init__.py
│  ├─ utils/config.py
│  ├─ utils/io.py
│  ├─ data/loader.py
│  ├─ features/indicators.py
│  ├─ models/arima_model.py
│  ├─ models/lstm_model.py
│  ├─ evaluation/metrics.py
│  ├─ evaluation/risk.py
│  └─ backtest/backtest.py
├─ scripts/
│  ├─ fetch_data.py
│  ├─ train_arima.py
│  ├─ train_lstm.py
│  ├─ backtest.py
│  └─ export_powerbi.py
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/
│  ├─ models/
│  └─ reports/figures/
├─ powerbi/
│  ├─ datasets/  # CSV exportés pour Power BI
│  └─ README.md
├─ notebooks/
│  ├─ 01_exploration.ipynb
│  ├─ 02_arima.ipynb
│  ├─ 03_lstm.ipynb
│  └─ 04_portfolio_risk.ipynb
└─ tests/
   ├─ test_indicators.py
   └─ test_metrics.py
```

## 🧰 Environnements

- Python ≥ 3.10
- Voir `requirements.txt` ou `environment.yml`

## 📊 Métriques & Risque

- **MSE/RMSE/MAE**, R²
- **Sharpe ratio**, **VaR** (Historique & Paramétrique)
- Backtest **walk-forward** simple

## 🧱 Architecture technique

- `src/finml/data`: chargement des séries (yfinance) + sauvegarde CSV
- `src/finml/features`: création des features (retours, log-returns, vol, etc.)
- `src/finml/models`: ARIMA (statsmodels), LSTM (TensorFlow/Keras)
- `src/finml/evaluation`: métriques régression & risque (VaR, Sharpe)
- `src/finml/backtest`: walk-forward, génération de signaux et PnL

## 📈 Power BI

Le script `export_powerbi.py` produit des CSV dans `powerbi/datasets/` :
- `prices.csv` : prix par ticker
- `signals.csv` : signaux/prédictions par modèle
- `portfolio.csv` : courbe d'équité, drawdown, Sharpe, VaR

Importez ces CSV dans Power BI Desktop et créez vos visuels.

## ✅ Tests

```bash
pytest -q
```

## 📜 Licence
MIT
