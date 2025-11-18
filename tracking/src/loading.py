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
    Carga un subset de un dataset en formato YOLO (imágenes + labels .txt).

    Pensado para datasets ya organizados como:
        base_dir/
          images/[split]/...
          labels/[split]/...

    (por ejemplo UA-DETRAC convertido a YOLO, DAWN, etc.)

    Parámetros:
      - base_dir: Carpeta raíz del dataset.
      - split: Nombre del split ('train', 'val', etc.) o None si no hay subcarpetas.
      - percent: Porcentaje de imágenes a usar (0–100).
      - shuffle: Si True, mezcla las imágenes antes de recortar el subset.
      - images_subdir: Subcarpeta donde están las imágenes (por defecto 'images').
      - labels_subdir: Subcarpeta donde están los labels .txt (por defecto 'labels').
      - img_exts: Extensiones de imagen válidas.

    Returns:
      - dict con:
          'image_paths': lista de Paths de imágenes seleccionadas.
          'label_paths': lista de Paths de labels correspondientes (mismo orden).
    """
    base_path = Path(base_dir)

    # Resolvemos carpetas de imágenes y labels
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

    # Todas las imágenes válidas del split
    all_images: List[Path] = []
    for ext in img_exts:
        all_images.extend(images_dir.glob(f"*{ext}"))
    all_images = sorted(all_images)

    if len(all_images) == 0:
        raise RuntimeError(f"No se encontraron imágenes en {images_dir}")

    # Armamos pares (img, label) y filtramos sólo los que tengan label
    pairs: List[tuple[Path, Path]] = []
    for img_path in all_images:
        label_name = img_path.stem + ".txt"
        label_path = labels_dir / label_name
        if label_path.exists():
            pairs.append((img_path, label_path))

    if len(pairs) == 0:
        raise RuntimeError(
            f"No se encontraron labels en {labels_dir} que hagan match con las imágenes."
        )

    if shuffle:
        random.shuffle(pairs)

    # Recorte por porcentaje
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


def load_yolo_detector(
    model_name: str = "yolo11s.pt"
) -> YOLO:
    """
    Carga un modelo YOLO11 de Ultralytics para detección/tracking.

    Parámetros:
      - model_name: Nombre del modelo (por defecto 'yolo11s.pt').
                    Podés usar 'yolo11n.pt', 'yolo11m.pt', o la ruta a tu 'best.pt'.

    Returns:
      - Instancia de YOLO lista para usar con .predict() o .track().
    """
    model = YOLO(model_name)
    print(f"Modelo YOLO cargado: {model_name}")
    return model
