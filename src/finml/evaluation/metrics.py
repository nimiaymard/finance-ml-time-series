import numpy as np

def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Input arrays must not be empty.")
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Input arrays must not be empty.")
    return float(np.mean(np.abs(y_true - y_pred)))

#  deux nouvelles métriques
def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE) : erreur moyenne en pourcentage.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = 1e-9  # pour éviter la division par zéro
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

def r2_score(y_true, y_pred):
    """
    Coefficient de détermination R² : proportion de variance expliquée.
    1 = parfait, 0 = nul, <0 = pire que la moyenne.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)
