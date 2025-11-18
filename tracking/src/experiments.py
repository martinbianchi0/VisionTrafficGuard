from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

from .loading import load_dataset_subset
from .training import create_yolo_subset_config, finetune_yolo_model, DEFAULT_CLASS_NAMES
from .visualization import plot_yolo_predictions_grid


# ------------------------------------------------------------------
# 1) DEFINICIÓN DE EXPERIMENTOS
# ------------------------------------------------------------------
# Cada experimento tiene:
#   - mixed_dir: dataset DETRAC_mixed_50_50 (raw o preprocesado)
#   - archive_dir: dataset archive correspondiente (raw o preprocesado)
#   - yaml_path: ruta donde se va a guardar el YAML de MIXED
#   - run_name: nombre del experimento en runs/detect (se completa luego)
# ------------------------------------------------------------------

EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "raw": {
        "mixed_dir": r"C:\Users\bianc\Vision\tpf\DETRAC_mixed_50_50",
        "archive_dir": r"C:\Users\bianc\Vision\tpf\archive",
        "yaml_path": r"C:\Users\bianc\Vision\tpf\configs\detrac_mixed_raw.yaml",
    },
    "clahe": {
        "mixed_dir": r"C:\Users\bianc\Vision\tpf\preproc_datasets\mixed_clahe",
        "archive_dir": r"C:\Users\bianc\Vision\tpf\preproc_datasets\archive_clahe",
        "yaml_path": r"C:\Users\bianc\Vision\tpf\configs\detrac_mixed_clahe.yaml",
    },
    "smooth": {
        "mixed_dir": r"C:\Users\bianc\Vision\tpf\preproc_datasets\mixed_smooth",
        "archive_dir": r"C:\Users\bianc\Vision\tpf\preproc_datasets\archive_smooth",
        "yaml_path": r"C:\Users\bianc\Vision\tpf\configs\detrac_mixed_smooth.yaml",
    },
    "unsharp": {
        "mixed_dir": r"C:\Users\bianc\Vision\tpf\preproc_datasets\mixed_unsharp",
        "archive_dir": r"C:\Users\bianc\Vision\tpf\preproc_datasets\archive_unsharp",
        "yaml_path": r"C:\Users\bianc\Vision\tpf\configs\detrac_mixed_unsharp.yaml",
    },
    "clahe_unsharp": {
        "mixed_dir": r"C:\Users\bianc\Vision\tpf\preproc_datasets\mixed_clahe_unsharp",
        "archive_dir": r"C:\Users\bianc\Vision\tpf\preproc_datasets\archive_clahe_unsharp",
        "yaml_path": r"C:\Users\bianc\Vision\tpf\configs\detrac_mixed_clahe_unsharp.yaml",
    },
    "smooth_unsharp": {
        "mixed_dir": r"C:\Users\bianc\Vision\tpf\preproc_datasets\mixed_smooth_unsharp",
        "archive_dir": r"C:\Users\bianc\Vision\tpf\preproc_datasets\archive_smooth_unsharp",
        "yaml_path": r"C:\Users\bianc\Vision\tpf\configs\detrac_mixed_smooth_unsharp.yaml",
    },
}


# ------------------------------------------------------------------
# 2) CREAR TODOS LOS YAML PARA LOS DATASETS MIXED_*
# ------------------------------------------------------------------

def prepare_mixed_yamls(
    experiments: Dict[str, Dict[str, Any]] = EXPERIMENTS,
    class_names: Dict[int, str] = DEFAULT_CLASS_NAMES,
    train_ratio: float = 0.8,
) -> None:
    """
    Crea los data.yaml y train/val.txt para TODOS los experimentos sobre
    los datasets MIXED (raw y preprocesados).

    Completa en cada experimento:
      - 'yaml_path' final (por si cambia)
      - 'train_txt'
      - 'val_txt'
    """
    for name, cfg in experiments.items():
        base_dir = cfg["mixed_dir"]
        yaml_path = Path(cfg["yaml_path"])

        subset = load_dataset_subset(
            base_dir=base_dir,
            split=None,
            images_subdir="images",
            labels_subdir="labels",
            percent=100.0,
            shuffle=True,
        )

        image_paths = subset["image_paths"]

        yaml_path_final, train_txt, val_txt = create_yolo_subset_config(
            image_paths=image_paths,
            yaml_path=yaml_path,
            class_names=class_names,
            train_ratio=train_ratio,
        )

        cfg["yaml_path"] = str(yaml_path_final)
        cfg["train_txt"] = str(train_txt)
        cfg["val_txt"] = str(val_txt)

        print(f"[{name}] YAML listo:")
        print(f"   yaml:  {yaml_path_final}")
        print(f"   train: {train_txt}")
        print(f"   val:   {val_txt}")


# ------------------------------------------------------------------
# 3) ENTRENAR TODOS LOS EXPERIMENTOS (MIXED_* como train/val)
# ------------------------------------------------------------------

def train_all_experiments(
    experiments: Dict[str, Dict[str, Any]] = EXPERIMENTS,
    base_weights: str = "yolo11s.pt",
    epochs: int = 5,
    imgsz: int = 640,
    batch: int = 4,
    project: str = "runs",
) -> None:
    """
    Hace fine-tuning de YOLO11 para cada experimento (raw, clahe, etc.)
    usando como dataset el MIXED correspondiente.

    Completa en cada experimento:
      - 'run_name' (nombre de la carpeta en runs/detect)
    """
    for name, cfg in experiments.items():
        print(f"\n================= Entrenando experimento: {name} =================")

        yaml_path = cfg["yaml_path"]
        run_name = f"exp_detrac_mixed_{name}"

        model = YOLO(base_weights)

        _ = finetune_yolo_model(
            model=model,
            data_yaml_path=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=run_name,
        )

        cfg["run_name"] = run_name
        print(f"[{name}] entrenamiento terminado. Run: {project}/detect/{run_name}")


# ------------------------------------------------------------------
# 4) RECOLECTAR MÉTRICAS DE CADA EXPERIMENTO (results.csv)
# ------------------------------------------------------------------

def collect_experiments_metrics(
    experiments: Dict[str, Dict[str, Any]] = EXPERIMENTS,
    runs_base_dir: str = r"runs/detect",
) -> pd.DataFrame:
    """
    Lee el results.csv de cada experimento y arma una tabla comparativa
    con las métricas principales:
      - precision, recall, F1
      - mAP50, mAP50-95
      - val_box_loss, val_cls_loss, val_dfl_loss
    """
    rows: List[Dict[str, Any]] = []

    for name, cfg in experiments.items():
        run_name = cfg.get("run_name")
        if not run_name:
            print(f"[WARN] {name} no tiene run_name (¿entrenaste este experimento?). Me lo salto.")
            continue

        csv_path = Path(runs_base_dir) / run_name / "results.csv"
        if not csv_path.exists():
            print(f"[WARN] No se encontró {csv_path} para {name}. Me lo salto.")
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"[WARN] {csv_path} está vacío para {name}. Me lo salto.")
            continue

        last = df.iloc[-1]  # último epoch

        prec = float(last.get("metrics/precision(B)", np.nan))
        rec = float(last.get("metrics/recall(B)", np.nan))
        if (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = np.nan

        row = {
            "exp": name,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "mAP50": float(last.get("metrics/mAP50(B)", np.nan)),
            "mAP50_95": float(last.get("metrics/mAP50-95(B)", np.nan)),
            "val_box_loss": float(last.get("val/box_loss", np.nan)),
            "val_cls_loss": float(last.get("val/cls_loss", np.nan)),
            "val_dfl_loss": float(last.get("val/dfl_loss", np.nan)),
        }
        rows.append(row)

    df_all = pd.DataFrame(rows)
    if not df_all.empty:
        df_all = df_all.sort_values("exp").reset_index(drop=True)
    return df_all


# ------------------------------------------------------------------
# 5) PLOT COMPARATIVO DE MÉTRICAS (FIGURA PARA EL INFORME)
# ------------------------------------------------------------------

def plot_experiments_metrics(metrics_df: pd.DataFrame) -> None:
    """
    Genera una figura con:
      - mAP50 y mAP50-95 para cada preprocesamiento
      - pérdidas de validación (box/cls/dfl) para cada preprocesamiento
    """
    if metrics_df.empty:
        print("metrics_df está vacío, nada para graficar.")
        return

    exps = metrics_df["exp"].tolist()
    x = np.arange(len(exps))

    mAP50 = metrics_df["mAP50"].values
    mAP50_95 = metrics_df["mAP50_95"].values

    v_box = metrics_df["val_box_loss"].values
    v_cls = metrics_df["val_cls_loss"].values
    v_dfl = metrics_df["val_dfl_loss"].values

    width = 0.25

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # --- Subplot 1: mAP ---
    ax1 = axes[0]
    ax1.bar(x - width/2, mAP50, width, label="mAP50")
    ax1.bar(x + width/2, mAP50_95, width, label="mAP50-95")

    ax1.set_xticks(x)
    ax1.set_xticklabels(exps, rotation=30)
    ax1.set_ylabel("mAP")
    ax1.set_title("Comparación de mAP por preprocesamiento")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # --- Subplot 2: pérdidas de validación ---
    ax2 = axes[1]
    ax2.bar(x - width, v_box, width, label="val_box_loss")
    ax2.bar(x,       v_cls, width, label="val_cls_loss")
    ax2.bar(x + width, v_dfl, width, label="val_dfl_loss")

    ax2.set_xticks(x)
    ax2.set_xticklabels(exps, rotation=30)
    ax2.set_ylabel("Loss")
    ax2.set_title("Comparación de pérdidas de validación")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 6) COMPARACIÓN CUALITATIVA EN ARCHIVE (GRID DE PREDICCIONES)
# ------------------------------------------------------------------

def qualitative_comparison_on_archive(
    experiments: Dict[str, Dict[str, Any]] = EXPERIMENTS,
    runs_base_dir: str = r"runs/detect",
    n_images: int = 6,
    conf: float = 0.25,
) -> None:
    """
    Para cada experimento:
      - carga el best.pt entrenado en MIXED
      - carga el dataset ARCHIVE correspondiente (raw o preprocesado)
      - muestra un grid 2x3 de predicciones sobre ese dominio sucio real
    """
    for name, cfg in experiments.items():
        run_name = cfg.get("run_name")
        if not run_name:
            print(f"[WARN] {name} no tiene run_name, me lo salto.")
            continue

        best_weights = Path(runs_base_dir) / run_name / "weights" / "best.pt"
        if not best_weights.exists():
            print(f"[WARN] No encontré {best_weights} para {name}, me lo salto.")
            continue

        archive_dir = cfg.get("archive_dir")
        if not archive_dir:
            print(f"[WARN] {name} no tiene archive_dir definido, me lo salto.")
            continue

        subset_arch = load_dataset_subset(
            base_dir=archive_dir,
            split=None,
            images_subdir="images",
            labels_subdir="labels",
            percent=100.0,
            shuffle=True,
        )

        image_paths = subset_arch["image_paths"]
        if not image_paths:
            print(f"[WARN] No hay imágenes en {archive_dir} para {name}.")
            continue

        sample_paths = image_paths[:n_images]

        print(f"\n===== Predicciones en ARCHIVE ({name}) =====")
        print(f"Usando pesos: {best_weights}")

        model = YOLO(str(best_weights))

        plot_yolo_predictions_grid(
            model=model,
            image_paths=sample_paths,
            n_rows=2,
            n_cols=3,
            conf=conf,
            sequence_filter=None,
        )
