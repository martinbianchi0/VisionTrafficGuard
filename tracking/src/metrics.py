from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _get_metric_column(df: pd.DataFrame, pattern: str) -> str | None:
    """
    Busca una columna cuyo nombre contiene un patrón.

    Parámetros:
      - df: DataFrame de métricas de Ultralytics.
      - pattern: Substring a buscar en los nombres de columnas.

    Returns:
      - Nombre de la columna encontrada o None.
    """
    for col in df.columns:
        if pattern in col:
            return col
    return None


def collect_experiments_metrics(
    experiments: Dict[str, Dict[str, Any]],
    runs_base_dir: str | Path = "runs/detect",
) -> pd.DataFrame:
    """
    Carga results.csv de cada experimento y arma un DataFrame.

    Parámetros:
      - experiments: Dict con info de cada experimento (incluye run_name).
      - runs_base_dir: Carpeta base donde Ultralytics guarda los runs.

    Returns:
      - DataFrame con una fila por experimento y columnas de métricas.
    """
    rows: List[Dict[str, Any]] = []
    runs_base_dir = Path(runs_base_dir)

    for name, cfg in experiments.items():
        run_name = cfg.get("run_name")
        if run_name is None:
            print(f"[WARNING] El experimento '{name}' no tiene run_name, se salta.")
            continue

        run_dir = runs_base_dir / run_name
        results_csv = run_dir / "results.csv"

        if not results_csv.exists():
            print(f"[WARNING] No se encontró results.csv para '{name}': {results_csv}")
            continue

        df = pd.read_csv(results_csv)
        if df.empty:
            print(f"[WARNING] results.csv vacío para '{name}'")
            continue

        last = df.iloc[-1]

        col_map_5095 = _get_metric_column(df, "metrics/mAP50-95")
        col_map_50 = _get_metric_column(df, "metrics/mAP50(")
        col_box = _get_metric_column(df, "train/box_loss")
        col_cls = _get_metric_column(df, "train/cls_loss")
        col_dfl = _get_metric_column(df, "train/dfl_loss")

        row = {
            "experiment": name,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "epochs": int(last.get("epoch", len(df) - 1)),
            "mAP50_95": float(last[col_map_5095]) if col_map_5095 else np.nan,
            "mAP50": float(last[col_map_50]) if col_map_50 else np.nan,
            "box_loss": float(last[col_box]) if col_box else np.nan,
            "cls_loss": float(last[col_cls]) if col_cls else np.nan,
            "dfl_loss": float(last[col_dfl]) if col_dfl else np.nan,
        }
        rows.append(row)

    if not rows:
        print("No se pudieron recolectar métricas de ningún experimento.")
        return pd.DataFrame()

    metrics_df = (
        pd.DataFrame(rows)
        .set_index("experiment")
        .sort_values("mAP50_95", ascending=False)
    )
    return metrics_df


def plot_experiments_metrics(metrics_df: pd.DataFrame) -> None:
    """
    Genera figuras comparativas de mAP y pérdidas finales.

    Parámetros:
      - metrics_df: DataFrame devuelto por collect_experiments_metrics().
    """
    if metrics_df.empty:
        print("metrics_df está vacío, no hay nada para plotear.")
        return

    experiments = metrics_df.index.tolist()
    x = np.arange(len(experiments))

    plt.figure(figsize=(10, 5))
    plt.bar(x, metrics_df["mAP50_95"], tick_label=experiments)
    plt.ylabel("mAP50-95")
    plt.title("Comparación de mAP50-95 entre preprocesamientos")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    width = 0.25
    plt.bar(x - width, metrics_df["box_loss"], width=width, label="box_loss")
    plt.bar(x, metrics_df["cls_loss"], width=width, label="cls_loss")
    if "dfl_loss" in metrics_df.columns:
        plt.bar(x + width, metrics_df["dfl_loss"], width=width, label="dfl_loss")

    plt.ylabel("Loss (última época)")
    plt.title("Pérdidas finales por preprocesamiento")
    plt.xticks(x, experiments, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
