from pathlib import Path
import random
from typing import List, Dict, Sequence

from ultralytics import YOLO


def load_dataset_subset(
    base_dir: str | Path,
    split: str | None = "train",
    percent: float = 5.0,
    shuffle: bool = True,
    images_subdir: str = "images",
    labels_subdir: str = "labels",
    img_exts: Sequence[str] = (".jpg", ".jpeg", ".png"),
) -> Dict[str, List[Path]]:
    """
    Carga un subset de un dataset en formato YOLO (imágenes + labels).

    Parámetros:
      - base_dir: Carpeta raíz del dataset.
      - split: Subcarpeta del split ('train', 'val', etc.) o None.
      - percent: Porcentaje de imágenes a usar (0–100).
      - shuffle: Si mezcla antes de recortar.
      - images_subdir: Subcarpeta de imágenes.
      - labels_subdir: Subcarpeta de labels .txt.
      - img_exts: Extensiones de imagen válidas.

    Returns:
      - Dict con listas 'image_paths' y 'label_paths' alineadas.
    """
    base_path = Path(base_dir)

    images_dir = base_path / images_subdir
    labels_dir = base_path / labels_subdir
    if split is not None:
        images_dir = images_dir / split
        labels_dir = labels_dir / split

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"No se encontraron carpetas de imágenes y labels en:\n"
            f"  imágenes: {images_dir}\n"
            f"  labels:   {labels_dir}"
        )

    all_images: List[Path] = []
    for ext in img_exts:
        all_images.extend(images_dir.glob(f"*{ext}"))
    all_images = sorted(all_images)

    if not all_images:
        raise RuntimeError(f"No se encontraron imágenes en {images_dir}")

    pairs: List[tuple[Path, Path]] = []
    for img_path in all_images:
        label_name = img_path.stem + ".txt"
        label_path = labels_dir / label_name
        if label_path.exists():
            pairs.append((img_path, label_path))

    if not pairs:
        raise RuntimeError(
            f"No se encontraron labels en {labels_dir} que hagan match con las imágenes."
        )

    if shuffle:
        random.shuffle(pairs)

    percent = max(0.0, min(100.0, percent))
    n = max(1, int(len(pairs) * (percent / 100.0)))
    pairs_subset = pairs[:n]

    image_paths = [p[0] for p in pairs_subset]
    label_paths = [p[1] for p in pairs_subset]

    split_name = split if split is not None else "all"
    print(f"Total imágenes en {split_name}: {len(pairs)} | Usando: {len(image_paths)} ({percent:.1f}%)")

    return {
        "image_paths": image_paths,
        "label_paths": label_paths,
    }


def load_yolo_detector(model_name: str = "yolo11s.pt") -> YOLO:
    """
    Carga un modelo YOLO11 de Ultralytics para detección o tracking.

    Parámetros:
      - model_name: Nombre o ruta del modelo (ej. 'yolo11s.pt' o 'best.pt').

    Returns:
      - Instancia de YOLO lista para .predict() o .track().
    """
    model = YOLO(model_name)
    print(f"Modelo YOLO cargado: {model_name}")
    return model

def load_dataset_from_root(
    base_dir: str | Path,
    percent: float = 100.0,
    shuffle: bool = True,
    images_subdir: str = "images",
    labels_subdir: str = "labels",
    img_exts: Sequence[str] = (".jpg", ".jpeg", ".png"),
) -> Dict[str, List[Path]]:
    """
    Carga un subconjunto de un dataset YOLO sin splits.

    Parámetros:
      - base_dir: Carpeta raíz del dataset (contiene images/ y labels/).
      - percent: Porcentaje de imágenes a usar (0–100).
      - shuffle: Si True, mezcla las imágenes antes de recortar.
      - images_subdir, labels_subdir: Nombres de las carpetas de imágenes y labels.
      - img_exts: Extensiones de imagen aceptadas.

    Returns:
      - Dict con listas de rutas: {"image_paths": [...], "label_paths": [...]}.
    """
    base_dir = Path(base_dir)
    images_dir = base_dir / images_subdir
    labels_dir = base_dir / labels_subdir

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"No se encontraron carpetas de imágenes y labels en:\n"
            f"  imágenes: {images_dir}\n"
            f"  labels:   {labels_dir}"
        )

    all_images: List[Path] = []
    for ext in img_exts:
        all_images.extend(sorted(images_dir.glob(f"*{ext}")))

    if not all_images:
        raise FileNotFoundError(f"No se encontraron imágenes en {images_dir}")

    if shuffle:
        random.shuffle(all_images)

    n = int(len(all_images) * (percent / 100.0))
    n = max(n, 1)
    selected_images = all_images[:n]

    label_paths: List[Path] = []
    for img_path in selected_images:
        label_paths.append(labels_dir / f"{img_path.stem}.txt")

    return {"image_paths": selected_images, "label_paths": label_paths}

