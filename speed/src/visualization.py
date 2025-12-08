import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scatter_real_vs_est(df, pred_col="pred_speed", gt_col="gt_speed", ax=None):
    """
    Grafica velocidades reales vs estimadas con la recta ideal y = x.

    Parámetros:
      - df: DataFrame con columnas de verdad y predicción.
      - pred_col: Nombre de la columna con la velocidad estimada.
      - gt_col: Nombre de la columna con la velocidad real.
      - ax: Eje de Matplotlib opcional.

    Returns:
      - Eje de Matplotlib con el gráfico.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    if df.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        return ax

    x = df[gt_col].to_numpy(dtype=float)
    y = df[pred_col].to_numpy(dtype=float)

    ax.scatter(x, y, alpha=0.7)

    min_v = min(float(x.min()), float(y.min())) * 0.9
    max_v = max(float(x.max()), float(y.max())) * 1.1

    ax.plot([min_v, max_v], [min_v, max_v], "--")
    ax.set_xlabel("Velocidad real (radar) [km/h]")
    ax.set_ylabel("Velocidad estimada [km/h]")
    ax.grid(True)
    return ax


def plot_speed_error_hist(df, pred_col="pred_speed", gt_col="gt_speed", ax=None, bins=20):
    """
    Muestra un histograma de errores de velocidad estimada vs real.

    Parámetros:
      - df: DataFrame con columnas de verdad y predicción.
      - pred_col: Nombre de la columna con la velocidad estimada.
      - gt_col: Nombre de la columna con la velocidad real.
      - ax: Eje de Matplotlib opcional.
      - bins: Cantidad de bins para el histograma.

    Returns:
      - Eje de Matplotlib con el histograma.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if df.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        return ax

    y_true = df[gt_col].to_numpy(dtype=float)
    y_pred = df[pred_col].to_numpy(dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    errors = y_pred - y_true

    ax.hist(errors, bins=bins)
    ax.set_xlabel("Error de velocidad (estimada - real) [km/h]")
    ax.set_ylabel("Cantidad de vehículos")
    ax.grid(True)
    return ax


def plot_speed_by_lane_boxplot(df, speed_col="speed_kmh", lane_col="lane", ax=None):
    """
    Muestra un boxplot de velocidades estimadas por carril.

    Parámetros:
      - df: DataFrame con columnas de carril y velocidad estimada.
      - speed_col: Nombre de la columna con la velocidad estimada.
      - lane_col: Nombre de la columna con el carril.
      - ax: Eje de Matplotlib opcional.

    Returns:
      - Eje de Matplotlib con el boxplot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if df.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        return ax

    df_valid = df[[lane_col, speed_col]].dropna()
    if df_valid.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        return ax

    lanes = sorted(df_valid[lane_col].unique())
    data = [df_valid[df_valid[lane_col] == lane][speed_col].to_numpy(dtype=float) for lane in lanes]

    ax.boxplot(data, positions=range(len(lanes)))
    ax.set_xticks(range(len(lanes)))
    ax.set_xticklabels([str(l) for l in lanes])
    ax.set_xlabel("Carril")
    ax.set_ylabel("Velocidad estimada [km/h]")
    ax.grid(True)
    return ax


def plot_metric_bar(df_metrics, metric="mae", ax=None):
    """
    Muestra un gráfico de barras de una métrica por video.

    Parámetros:
      - df_metrics: DataFrame indexado por nombre de video con métricas por fila.
      - metric: Nombre de la columna de la métrica a graficar.
      - ax: Eje de Matplotlib opcional.

    Returns:
      - Eje de Matplotlib con el gráfico de barras.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if df_metrics.empty or metric not in df_metrics.columns:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        return ax

    videos = list(df_metrics.index)
    values = df_metrics[metric].to_numpy(dtype=float)

    ax.bar(range(len(videos)), values)
    ax.set_xticks(range(len(videos)))
    ax.set_xticklabels(videos, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.grid(True)
    return ax
