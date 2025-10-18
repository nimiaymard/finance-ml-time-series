from __future__ import annotations
import numpy as np, pandas as pd
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


# ==============================================================
# Helpers
# ==============================================================
def _series_to_logreturns(prices: pd.Series) -> pd.Series:
    """log-return_t = log(Price_t / Price_{t-1}); enlève le premier NaN."""
    s = prices.astype(float)
    logret = np.log(s / s.shift(1))
    return logret.dropna()


def _build_sequences(arr: np.ndarray, seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Construit (X, y) à partir d'un vecteur 1D : X[t] = arr[t-seq_len:t], y[t] = arr[t]."""
    X, y = [], []
    for i in range(seq_len, len(arr)):
        X.append(arr[i-seq_len:i])
        y.append(arr[i])
    X, y = np.array(X), np.array(y)
    return X[..., np.newaxis], y  # (samples, timesteps, 1)


def _time_series_train_val_split(X, y, val_ratio=0.2):
    """Split chronologique (pas de shuffle)."""
    n = len(X)
    split = max(int(n * (1 - val_ratio)), 1)
    return X[:split], y[:split], X[split:], y[split:]


# ==============================================================
# Entraînement LSTM (log-returns ou prix)
# ==============================================================
def train_lstm(
    prices: pd.Series,
    seq_len: int = 60,
    epochs: int = 30,
    batch_size: int = 32,
    mode: str = "logret",  # "logret" (recommandé) ou "price"
):
    """
    Entraîne un LSTM pour prédire le pas suivant.
    - mode="logret": entraîne sur les log-returns (fortement recommandé)
    - mode="price" : entraîne sur les prix normalisés (moins stable)
    Retourne: (model, scaler, mode)
    """
    if mode not in {"logret", "price"}:
        raise ValueError("mode must be 'logret' or 'price'")

    if mode == "logret":
        target_series = _series_to_logreturns(prices)
    else:
        target_series = prices.astype(float).copy()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(target_series.values.reshape(-1, 1)).ravel()

    X, y = _build_sequences(scaled, seq_len=seq_len)
    if len(X) < 10:
        raise ValueError("Série trop courte pour construire des séquences LSTM avec ce seq_len.")

    X_tr, y_tr, X_val, y_val = _time_series_train_val_split(X, y, val_ratio=0.2)

    model = Sequential([
        LSTM(64, input_shape=(seq_len, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=0
    )

    return model, scaler, mode


# ==============================================================
# Prédiction LSTM
# ==============================================================
def lstm_predict(
    model,
    scaler,
    prices: pd.Series,
    seq_len: int = 60,
    steps: int = 1,
    mode: str = "logret",
):
    """
    Multi-step forecast.
    - Si mode='logret': prédit un log-return, puis reconstruit le prix
      via p_{t+1} = p_t * exp(logret_pred)
    - Si mode='price': inverse_transform simple
    """
    s = prices.astype(float)
    if len(s) < seq_len:
        raise ValueError("La fenêtre fournie à lstm_predict est plus courte que seq_len.")

    if mode == "logret":
        # Log-returns sur la fenêtre
        logret = _series_to_logreturns(s)
        base = logret.values[-seq_len:].reshape(-1, 1)
        base_scaled = scaler.transform(base).ravel().tolist()

        last_price = s.iloc[-1]
        preds_prices = []

        for _ in range(steps):
            X = np.array(base_scaled[-seq_len:])[..., np.newaxis].reshape(1, seq_len, 1)
            pred_scaled = model.predict(X, verbose=0).ravel()[0]
            pred_logret = scaler.inverse_transform(np.array([[pred_scaled]])).ravel()[0]
            next_price = last_price * np.exp(pred_logret)
            preds_prices.append(next_price)
            last_price = next_price
            base_scaled.append(pred_scaled)

        return np.array(preds_prices)

    else:  # mode == "price"
        arr = scaler.transform(s.values.reshape(-1, 1)).ravel()
        last = arr[-seq_len:].tolist()
        out_scaled = []
        for _ in range(steps):
            X = np.array(last[-seq_len:])[..., np.newaxis].reshape(1, seq_len, 1)
            pred_scaled = model.predict(X, verbose=0).ravel()[0]
            out_scaled.append(pred_scaled)
            last.append(pred_scaled)
        inv = scaler.inverse_transform(np.array(out_scaled).reshape(-1, 1)).ravel()
        return inv
