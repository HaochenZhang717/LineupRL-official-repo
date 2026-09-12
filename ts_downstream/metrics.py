from __future__ import annotations

import numpy as np


def classification_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def forecasting_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    return {
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
    }


def seasonal_naive_forecast(history: np.ndarray, horizon: int, period: int = 1) -> np.ndarray:
    history = np.asarray(history, dtype=np.float64)
    if period <= 1 or history.shape[-1] < period:
        return np.repeat(history[..., -1:], horizon, axis=-1)
    reps = int(np.ceil(horizon / period))
    tail = history[..., -period:]
    return np.tile(tail, reps)[..., :horizon]


def naive_skill(y_true, y_pred, y_naive) -> float:
    m = forecasting_metrics(y_true, y_pred)["mse"]
    n = forecasting_metrics(y_true, y_naive)["mse"]
    return float(1.0 - m / n) if n > 0 else 0.0
