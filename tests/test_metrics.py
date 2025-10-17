from src.finml.evaluation.metrics import mse, rmse, mae

def test_metrics():
    y_true = [1,2,3]
    y_pred = [1,2,4]
    assert mse(y_true, y_pred) == 1/3
    assert round(rmse(y_true, y_pred), 6) == round((1/3)**0.5, 6)
    assert mae(y_true, y_pred) == 1/3
