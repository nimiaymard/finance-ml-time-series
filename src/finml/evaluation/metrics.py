import numpy as np

def mse(y_true, y_pred):
    return float(np.mean((np.array(y_true) - np.array(y_pred))**2))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))
