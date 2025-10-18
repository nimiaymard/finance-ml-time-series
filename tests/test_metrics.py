import numpy as np
import math
import pytest
from src.finml.evaluation.metrics import mse, rmse, mae, mape, r2_score

def test_metrics_values():
    y_true = np.array([100, 102, 101, 105], dtype=float)
    y_pred = np.array([ 99, 103, 100, 106], dtype=float)

    assert math.isclose(mse(y_true, y_pred), 1.0, rel_tol=1e-9)
    assert math.isclose(rmse(y_true, y_pred), 1.0, rel_tol=1e-9)
    assert math.isclose(mae(y_true, y_pred), 1.0, rel_tol=1e-9)
    assert 0.5 <= mape(y_true, y_pred) <= 2.0
    assert 0.6 <= r2_score(y_true, y_pred) <= 0.9

def test_input_validation():
    # shape mismatch
    with pytest.raises(ValueError):
        mse(np.array([1,2]), np.array([1]))
    # empty arrays
    with pytest.raises(ValueError):
        mae(np.array([]), np.array([]))
