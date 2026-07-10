"""Regression metrics with support for log-transformed targets."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def to_dollar_scale(values: np.ndarray, *, log_scale: bool) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if log_scale:
        return np.expm1(array)
    return array


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    log_scale: bool,
) -> dict[str, float]:
    y_true_dollars = to_dollar_scale(y_true, log_scale=log_scale)
    y_pred_dollars = to_dollar_scale(y_pred, log_scale=log_scale)

    return {
        "mae": float(mean_absolute_error(y_true_dollars, y_pred_dollars)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_dollars, y_pred_dollars))),
        "r2": float(r2_score(y_true_dollars, y_pred_dollars)),
    }
