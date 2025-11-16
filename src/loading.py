from pathlib import Path
import random
from typing import List, Dict
from ultralytics import YOLO

def load_dataset_subset(
    base_dir: str,
    split: str = "train",
    percent: float = 5.0,
    shuffle: bool = True
    ) -> Dict[str, List[Path]]:
    """
    Carga un subset del dataset UA-DETRAC en formato YOLO.

    Parámetros:
      - base_dir: Carpeta raíz donde está DETRAC_Upload (por ejemplo: 'C:/Users/bianc/Vision/tpf/DETRAC_Upload').
      - split: 'train' o 'val'.
      - percent: Porcentaje de imágenes a usar (0–100). Ej: 5.0 = 5% del split.
      - shuffle: Si True, mezcla las imágenes antes de recortar el subset.

    Returns:
      - dict con:
          'image_paths': lista de Paths de imágenes seleccionadas.
          'label_paths': lista de Paths de labels correspondientes (mismo orden).
    """
    base_path = Path(base_dir)
    images_dir = base_path / "images" / split
    labels_dir = base_path / "labels" / split

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"No se encontraron carpetas 'images/{split}' y 'labels/{split}' en {base_dir}")

    # Todas las imágenes del split
    all_images = sorted(images_dir.glob("*.jpg"))  # podés agregar *.png si hiciera falta

    if len(all_images) == 0:
        raise RuntimeError(f"No se encontraron imágenes en {images_dir}")

    # Armamos pares (img, label) y filtramos sólo los que tengan label
    pairs = []
    for img_path in all_images:
        label_name = img_path.stem + ".txt"  # MVI_20011_img00001 -> MVI_20011_img00001.txt
        label_path = labels_dir / label_name
        if label_path.exists():
            pairs.append((img_path, label_path))

    if len(pairs) == 0:
        raise RuntimeError(f"No se encontraron labels en {labels_dir} que hagan match con las imágenes.")

    if shuffle:
        random.shuffle(pairs)

    # Recorte por porcentaje
    percent = max(0.0, min(100.0, percent))
    n = max(1, int(len(pairs) * (percent / 100.0)))
    pairs_subset = pairs[:n]

    image_paths = [p[0] for p in pairs_subset]
    label_paths = [p[1] for p in pairs_subset]

    print(f"Total imágenes en {split}: {len(pairs)} | Usando: {len(image_paths)} ({percent:.1f}%)")

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