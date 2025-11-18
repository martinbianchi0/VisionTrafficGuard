from pathlib import Path
from typing import List, Callable, Tuple

import cv2
import numpy as np


# --------------------------------------------------------
# 1) OPERADORES DE PREPROCESAMIENTO CLÁSICOS
# --------------------------------------------------------

def clahe_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Aplica CLAHE (ecualización adaptativa) sobre el canal L de LAB.

    Mejora contraste en zonas oscuras/sobreexpuestas sin destruir tanto el detalle.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out


def smooth_median_bilateral(img_bgr: np.ndarray) -> np.ndarray:
    """
    Suavizado suave:
      1) Filtro mediana (quita ruido tipo sal y pimienta).
      2) Filtro bilateral chico (suaviza pero mantiene bordes).

    Útil para reducir ruido sin borrar contornos de autos/semaforos.
    """
    med = cv2.medianBlur(img_bgr, 3)
    bil = cv2.bilateralFilter(med, d=5, sigmaColor=50, sigmaSpace=50)
    return bil


def unsharp_mask(img_bgr: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """
    Unsharp masking:
      - Blur gaussiano
      - Imagen sharpen = img * (1 + amount) - blur * amount

    Refuerza bordes y detalles finos (coches, líneas, etc.).
    """
    blur = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blur, -amount, 0)
    return sharp


# ----------------- PIPELINES COMBINADOS -----------------

def preproc_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """Solo CLAHE."""
    return clahe_bgr(img_bgr)


def preproc_clahe_unsharp(img_bgr: np.ndarray) -> np.ndarray:
    """CLAHE + Unsharp."""
    tmp = clahe_bgr(img_bgr)
    return unsharp_mask(tmp, amount=0.7)


def preproc_smooth_unsharp(img_bgr: np.ndarray) -> np.ndarray:
    """Median + Bilateral + Unsharp."""
    tmp = smooth_median_bilateral(img_bgr)
    return unsharp_mask(tmp, amount=0.7)


# --------------------------------------------------------
# 2) GENERADOR DE DATASET PREPROCESADO
# --------------------------------------------------------

def generate_preprocessed_dataset(
    image_paths: List[Path],
    label_paths: List[Path],
    output_base_dir: str | Path,
    preproc_fn: Callable[[np.ndarray], np.ndarray],
    suffix: str,
) -> Tuple[List[Path], List[Path]]:
    """
    Genera un dataset YOLO preprocesado a partir de imágenes + labels:

      - Aplica preproc_fn a cada imagen BGR.
      - Guarda en:
          output_base_dir/images
          output_base_dir/labels
      - Copia los .txt de labels sin modificarlos.
      - El nombre de cada imagen/label se extiende con _<suffix>.

    Ej:
      MVI_20011_00001.jpg -> MVI_20011_00001_clahe.jpg

    Returns:
      - (new_image_paths, new_label_paths)
    """
    output_base = Path(output_base_dir)
    out_images = output_base / "images"
    out_labels = output_base / "labels"

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    new_image_paths: List[Path] = []
    new_label_paths: List[Path] = []

    assert len(image_paths) == len(label_paths), "image_paths y label_paths deben tener misma longitud"

    for img_path, lbl_path in zip(image_paths, label_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] No se pudo leer {img_path}")
            continue

        proc = preproc_fn(img)

        # Nuevo nombre con sufijo
        new_name = f"{img_path.stem}_{suffix}{img_path.suffix}"
        out_img_path = out_images / new_name
        out_lbl_path = out_labels / f"{img_path.stem}_{suffix}.txt"

        # Guardar imagen preprocesada
        cv2.imwrite(str(out_img_path), proc)

        # Copiar label TXT correspondiente
        if lbl_path.exists():
            txt = lbl_path.read_text(encoding="utf-8")
            out_lbl_path.write_text(txt, encoding="utf-8")
        else:
            print(f"[WARN] No se encontró label para {img_path}")

        new_image_paths.append(out_img_path)
        new_label_paths.append(out_lbl_path)

    return new_image_paths, new_label_paths
