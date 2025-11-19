from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from ultralytics import YOLO

from .loading import load_dataset_subset, load_dataset_from_root
from .training import create_yolo_subset_config, DEFAULT_CLASS_NAMES
from .visualization import plot_yolo_predictions_grid
from .metrics import collect_experiments_metrics, plot_experiments_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR_MIXED = PROJECT_ROOT / "DETRAC_mixed_50_50"
DATA_DIR_ARCHIVE = PROJECT_ROOT / "archive"
PREPROC_DIR = PROJECT_ROOT / "preproc_datasets"
CONFIGS_DIR = PROJECT_ROOT / "configs"


EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "raw": {
        "mixed_dir": str(DATA_DIR_MIXED),
        "archive_dir": str(DATA_DIR_ARCHIVE),
        "yaml_path": str(CONFIGS_DIR / "detrac_mixed_raw.yaml"),
        "run_name": None,
    },
    "clahe": {
        "mixed_dir": str(PREPROC_DIR / "mixed_clahe"),
        "archive_dir": str(PREPROC_DIR / "archive_clahe"),
        "yaml_path": str(CONFIGS_DIR / "detrac_mixed_clahe.yaml"),
        "run_name": None,
    },
    "smooth": {
        "mixed_dir": str(PREPROC_DIR / "mixed_smooth"),
        "archive_dir": str(PREPROC_DIR / "archive_smooth"),
        "yaml_path": str(CONFIGS_DIR / "detrac_mixed_smooth.yaml"),
        "run_name": None,
    },
    "unsharp": {
        "mixed_dir": str(PREPROC_DIR / "mixed_unsharp"),
        "archive_dir": str(PREPROC_DIR / "archive_unsharp"),
        "yaml_path": str(CONFIGS_DIR / "detrac_mixed_unsharp.yaml"),
        "run_name": None,
    },
    "clahe_unsharp": {
        "mixed_dir": str(PREPROC_DIR / "mixed_clahe_unsharp"),
        "archive_dir": str(PREPROC_DIR / "archive_clahe_unsharp"),
        "yaml_path": str(CONFIGS_DIR / "detrac_mixed_clahe_unsharp.yaml"),
        "run_name": None,
    },
    "smooth_unsharp": {
        "mixed_dir": str(PREPROC_DIR / "mixed_smooth_unsharp"),
        "archive_dir": str(PREPROC_DIR / "archive_smooth_unsharp"),
        "yaml_path": str(CONFIGS_DIR / "detrac_mixed_smooth_unsharp.yaml"),
        "run_name": None,
    },
}


def prepare_mixed_yamls(
    experiments: Dict[str, Dict[str, Any]] = EXPERIMENTS,
    class_names: Dict[int, str] = DEFAULT_CLASS_NAMES,
    train_ratio: float = 0.8,
) -> None:
    """
    Crea los data.yaml y train/val.txt para TODOS los experimentos
    sobre los datasets MIXED (raw y preprocesados).

    Usa la carpeta raíz (images/ y labels/ sin splits).
    """
    for name, cfg in experiments.items():
        mixed_dir = Path(cfg["mixed_dir"])
        yaml_path = Path(cfg["yaml_path"])

        print(f"\n===== Preparando YAML para experimento: {name} =====")
        print(f"Directorio MIXED: {mixed_dir}")

        subset = load_dataset_from_root(
            base_dir=mixed_dir,
            percent=100.0,
            shuffle=True,
            images_subdir="images",
            labels_subdir="labels",
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



def train_all_experiments(
    experiments: Dict[str, Dict[str, Any]] = EXPERIMENTS,
    base_weights: str = "yolo11s.pt",
    epochs: int = 5,
    imgsz: int = 640,
    batch: int = 4,
    project: str = "runs",
) -> Dict[str, Any]:
    """
    Entrena YOLO11 sobre cada YAML de MIXED_50_50.

    Parámetros:
      - experiments: Diccionario de experimentos.
      - base_weights: Checkpoint base de YOLO11.
      - epochs: Épocas por experimento.
      - imgsz: Tamaño de imagen.
      - batch: Batch size.
      - project: Carpeta raíz de runs en Ultralytics.

    Returns:
      - Dict {nombre_experimento: objeto_resultado_ultralytics}.
    """
    results: Dict[str, Any] = {}

    for name, cfg in experiments.items():
        yaml_path = cfg.get("yaml_path")
        if yaml_path is None:
            raise ValueError(
                f"El experimento '{name}' no tiene 'yaml_path'. "
                f"Llamaste primero a prepare_mixed_yamls()?"
            )

        run_name = f"mixed_{name}"
        cfg["run_name"] = run_name

        print(f"\n===== Entrenando experimento: {name} =====")
        print(f"YAML: {yaml_path}")
        print(f"run_name: {run_name}")

        model = YOLO(base_weights)

        train_result = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=run_name,
            device=0,
        )

        results[name] = train_result

    return results


def run_preprocessing_sweep(
    epochs: int = 10,
    batch: int = 8,
    imgsz: int = 640,
    base_weights: str = "yolo11s.pt",
    project: str = "runs",
) -> pd.DataFrame:
    """
    Ejecuta todo el pipeline sobre los preprocesamientos MIXED_*.

    Parámetros:
      - epochs: Épocas por experimento.
      - batch: Batch size.
      - imgsz: Tamaño de imagen.
      - base_weights: Checkpoint base de YOLO11.
      - project: Carpeta raíz de Ultralytics para los runs.

    Returns:
      - DataFrame con métricas finales de cada experimento.
    """
    prepare_mixed_yamls(experiments=EXPERIMENTS, class_names=DEFAULT_CLASS_NAMES)

    train_all_experiments(
        experiments=EXPERIMENTS,
        base_weights=base_weights,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
    )

    metrics_df = collect_experiments_metrics(
        experiments=EXPERIMENTS,
        runs_base_dir=Path(project) / "detect",
    )

    if not metrics_df.empty:
        plot_experiments_metrics(metrics_df)
    else:
        print("No se pudieron recolectar métricas (metrics_df vacío).")

    return metrics_df


def qualitative_comparison_on_archive(
    experiments: Dict[str, Dict[str, Any]] = EXPERIMENTS,
    runs_base_dir: str | Path = "runs/detect",
    n_images: int = 6,
    conf: float = 0.25,
) -> None:
    """
    Muestra grids de predicciones en ARCHIVE para cada experimento.

    Parámetros:
      - experiments: Diccionario de experimentos con archive_dir y run_name.
      - runs_base_dir: Carpeta base de los runs.
      - n_images: Imágenes a muestrear por experimento.
      - conf: Umbral de confianza para YOLO.

    Muestra:
      - Grids 2x3 con predicciones sobre imágenes sucias reales.
    """
    runs_base_dir = Path(runs_base_dir)

    for name, cfg in experiments.items():
        archive_dir = Path(cfg["archive_dir"])
        if not archive_dir.exists():
            print(f"[WARNING] archive_dir no existe para '{name}': {archive_dir}")
            continue

        run_name = cfg.get("run_name")
        if run_name is None:
            print(f"[WARNING] El experimento '{name}' no tiene run_name, se salta.")
            continue

        run_dir = runs_base_dir / run_name
        best_weights = run_dir / "weights" / "best.pt"
        last_weights = run_dir / "weights" / "last.pt"

        if best_weights.exists():
            weights_path = best_weights
        elif last_weights.exists():
            print(f"[WARNING] No hay best.pt para '{name}', usando last.pt.")
            weights_path = last_weights
        else:
            print(f"[WARNING] No se encontraron pesos para '{name}' en {run_dir}")
            continue

        subset = load_dataset_subset(archive_dir, max_images=n_images)
        image_paths = subset["image_paths"]

        print(f"\n===== Predicciones en ARCHIVE ({name}) =====")
        print(f"Usando pesos: {weights_path}")

        model = YOLO(str(weights_path))

        plot_yolo_predictions_grid(
            model=model,
            image_paths=image_paths,
            n_rows=2,
            n_cols=3,
            conf=conf,
            sequence_filter=None,
        )