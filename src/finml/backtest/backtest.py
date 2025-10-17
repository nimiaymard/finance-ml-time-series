from __future__ import annotations
import pandas as pd
from typing import Literal, Tuple
from ..models.arima_model import fit_arima  # ⚠️ pas d'import LSTM ici (lazy import plus bas)

def _ensure_time_index(series: pd.Series) -> pd.Series:
    """
    Force un DatetimeIndex + une fréquence pour statsmodels.
    Si la fréquence est inconnue, on impose 'B' (jours ouvrés) puis on ffill.
    """
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    freq = pd.infer_freq(s.index)
    if freq is None:
        s = s.asfreq("B").ffill()
    else:
        s = s.asfreq(freq)
    return s

class Backtester:
    """
    Backtesting ARIMA et LSTM en walk-forward, robuste et sans import TensorFlow
    quand on ne l’utilise pas.
    """
    def __init__(self, train_size: float = 0.7, seq_len: int = 60, arima_order: Tuple[int,int,int]=(5,1,0)):
        self.train_size = train_size
        self.seq_len = seq_len
        self.arima_order = arima_order

    # ---------- ARIMA ----------
    def _walk_forward_arima(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        # Séries temporelles proprement indexées
        train_s = train_df.set_index("Date")[target_col].astype(float)
        test_s  = test_df.set_index("Date")[target_col].astype(float)
        train_s = _ensure_time_index(train_s)
        test_s  = _ensure_time_index(test_s)

        # Fit initial
        res = fit_arima(train_s, order=self.arima_order)
        preds = []

        # Walk-forward : forecast 1 pas -> append vraie obs
        for dt, y_true in test_s.items():
            # 1) forecast
            try:
                y_hat = res.forecast(steps=1).iloc[0]
            except Exception:
                # en cas de pépin, refit rapide
                res = fit_arima(train_s, order=self.arima_order)
                y_hat = res.forecast(steps=1).iloc[0]
            preds.append((dt, y_hat))

            # 2) append observation réelle (même index)
            try:
                res = res.append(pd.Series([y_true], index=[pd.to_datetime(dt)]), refit=False)
            except Exception:
                # Fallback sûr : on étend l'historique et on refit
                train_s = pd.concat([train_s, pd.Series([y_true], index=[pd.to_datetime(dt)])])
                train_s = _ensure_time_index(train_s)
                res = fit_arima(train_s, order=self.arima_order)

        # Sortie alignée
        out = test_df.copy()
        pred_series = pd.Series([p[1] for p in preds], index=[p[0] for p in preds], name="Prediction")
        out = out.set_index("Date").join(pred_series).reset_index()
        return out

    # ---------- LSTM ----------
    def _walk_forward_lstm(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        # Import "paresseux" pour ne pas initialiser TensorFlow si on est en ARIMA
        from ..models.lstm_model import train_lstm, lstm_predict
        mdl, scaler = train_lstm(train_df[target_col], seq_len=self.seq_len, epochs=5)
        # Historique continu : queue du train + test
        hist = pd.concat([train_df[target_col].tail(self.seq_len), test_df[target_col]])
        preds = []
        for i in range(len(test_df)):
            window = hist.iloc[i: i + self.seq_len]
            pred = lstm_predict(mdl, scaler, window, seq_len=self.seq_len, steps=1)[0]
            preds.append(pred)
        out = test_df.copy()
        out["Prediction"] = preds
        return out

    # ---------- API publique ----------
    def walk_forward(self, df: pd.DataFrame, target_col: str, model: Literal["arima","lstm"]="arima") -> pd.DataFrame:
        df = df.sort_values("Date").copy()
        df["Date"] = pd.to_datetime(df["Date"])
        n = len(df)
        split = int(n * self.train_size)
        train, test = df.iloc[:split], df.iloc[split:]
        if model == "arima":
            return self._walk_forward_arima(train, test, target_col)
        else:
            return self._walk_forward_lstm(train, test, target_col)

# Compatibilité avec l’ancien appel
def walk_forward(df: pd.DataFrame, target_col: str, model: Literal["arima","lstm"]="arima",
                 train_size: float = 0.7, seq_len: int = 60) -> pd.DataFrame:
    backtester = Backtester(train_size=train_size, seq_len=seq_len)
    return backtester.walk_forward(df, target_col, model)
