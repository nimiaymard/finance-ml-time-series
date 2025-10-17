from __future__ import annotations
import numpy as np, pandas as pd, os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from ..utils.io import save_pickle

def build_sequences(series: np.ndarray, seq_len: int = 60):
    X, y = [], []
    for i in range(seq_len, len(series)):
        X.append(series[i-seq_len:i])
        y.append(series[i])
    X, y = np.array(X), np.array(y)
    return X[..., np.newaxis], y  # (samples, timesteps, 1)

def train_lstm(prices: pd.Series, seq_len: int = 60, epochs: int = 10, batch_size: int = 32):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.values.reshape(-1,1)).ravel()
    X, y = build_sequences(scaled, seq_len=seq_len)

    model = Sequential([
        LSTM(64, input_shape=(seq_len, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
    model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
    return model, scaler

def lstm_predict(model, scaler, prices: pd.Series, seq_len: int = 60, steps: int = 1):
    arr = scaler.transform(prices.values.reshape(-1,1)).ravel()
    out = []
    last = arr[-seq_len:].tolist()
    for _ in range(steps):
        X = np.array(last)[-seq_len:][..., np.newaxis].reshape(1, seq_len, 1)
        pred = model.predict(X, verbose=0).ravel()[0]
        out.append(pred)
        last.append(pred)
    inv = scaler.inverse_transform(np.array(out).reshape(-1,1)).ravel()
    return inv
