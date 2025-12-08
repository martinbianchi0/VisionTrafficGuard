import math

import numpy as np
import pandas as pd


def evaluate_predictions(df, pred_col="pred_speed", gt_col="gt_speed", verbose=False):
    """
    Calcula métricas MAE, RMSE, MAPE y R^2 para predicciones de velocidad.

    Parámetros:
      - df: DataFrame con columnas de verdad y predicción.
      - pred_col: Nombre de la columna con las velocidades estimadas.
      - gt_col: Nombre de la columna con las velocidades reales.
      - verbose: Si es True, imprime las métricas por pantalla.

    Returns:
      - Dict con métricas: mae, rmse, mape, r2 y n_samples.
    """
    if df.empty:
        metrics = {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "r2": float("nan"),
            "n_samples": 0,
        }
        if verbose:
            print("Sin datos válidos para métricas.")
        return metrics

    y_true = df[gt_col].to_numpy(dtype=float)
    y_pred = df[pred_col].to_numpy(dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        metrics = {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "r2": float("nan"),
            "n_samples": 0,
        }
        if verbose:
            print("Sin datos válidos para métricas.")
        return metrics

    diffs = y_pred - y_true
    abs_diffs = np.abs(diffs)

    mae = float(abs_diffs.mean())
    rmse = float(math.sqrt(np.mean(diffs ** 2)))
    mape = float(np.mean(abs_diffs / np.abs(y_true)) * 100.0)

    var_y = float(np.var(y_true))
    if var_y == 0:
        r2 = float("nan")
    else:
        ss_res = float(np.sum(diffs ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "n_samples": int(len(y_true)),
    }

    if verbose:
        print(f"MAE (km/h):  {metrics['mae']:.2f}")
        print(f"RMSE (km/h): {metrics['rmse']:.2f}")
        print(f"MAPE (%):    {metrics['mape']:.2f}")
        print(f"R^2:         {metrics['r2']:.3f}")
        print(f"N muestras:  {metrics['n_samples']}")

    return metrics


def calibrate_lane_scales(
    df_eval,
    pred_col="speed_raw",
    gt_col="gt_speed",
    lane_col="lane",
    min_samples=4,
):
    """
    Ajusta un factor de escala K_lane por carril usando los vehículos de un video.

    Se asume gt_speed ≈ K_lane * speed_raw y se calcula K_lane con regresión sin bias.

    Parámetros:
      - df_eval: DataFrame con predicciones crudas y ground truth.
      - pred_col: Nombre de la columna de velocidad cruda.
      - gt_col: Nombre de la columna de velocidad real.
      - lane_col: Nombre de la columna de carril.
      - min_samples: Mínimo de vehículos por carril.

    Returns:
      - Dict {lane -> K_lane}.
    """
    lane_scales = {}
    for lane, df_lane in df_eval.groupby(lane_col):
        x = df_lane[pred_col].to_numpy(dtype=float)
        y = df_lane[gt_col].to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) < min_samples:
            continue

        denom = float(np.sum(x ** 2))
        if denom <= 0:
            continue

        k = float(np.sum(x * y) / denom)
        lane_scales[lane] = k

    return lane_scales
