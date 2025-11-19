from pathlib import Path
from typing import List, Callable, Tuple

import cv2
import numpy as np


def clahe_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Aplica CLAHE sobre el canal L en espacio LAB.

    Parámetros:
      - img_bgr: Imagen original en BGR.

    Returns:
      - Imagen BGR con contraste mejorado.
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
    Suaviza la imagen con mediana y bilateral.

    Parámetros:
      - img_bgr: Imagen original en BGR.

    Returns:
      - Imagen BGR con ruido reducido manteniendo bordes.
    """
    blur_med = cv2.medianBlur(img_bgr, 5)
    blur_bil = cv2.bilateralFilter(blur_med, d=9, sigmaColor=100, sigmaSpace=100)
    return blur_bil


def unsharp_mask(img_bgr: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """
    Aplica unsharp masking para resaltar bordes.

    Parámetros:
      - img_bgr: Imagen original en BGR.
      - amount: Intensidad del sharpen.

    Returns:
      - Imagen BGR con detalles reforzados.
    """
    blur = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blur, -amount, 0)
    return sharp


def preproc_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """
    Aplica sólo CLAHE a una imagen BGR.

    Parámetros:
      - img_bgr: Imagen original.

    Returns:
      - Imagen procesada.
    """
    return clahe_bgr(img_bgr)


def preproc_clahe_unsharp(img_bgr: np.ndarray) -> np.ndarray:
    """
    Aplica CLAHE seguido de unsharp masking.

    Parámetros:
      - img_bgr: Imagen original.

    Returns:
      - Imagen procesada.
    """
    tmp = clahe_bgr(img_bgr)
    return unsharp_mask(tmp, amount=0.7)


def preproc_smooth_unsharp(img_bgr: np.ndarray) -> np.ndarray:
    """
    Aplica mediana + bilateral + unsharp masking.

    Parámetros:
      - img_bgr: Imagen original.

    Returns:
      - Imagen procesada.
    """
    tmp = smooth_median_bilateral(img_bgr)
    return unsharp_mask(tmp, amount=0.7)


def generate_preprocessed_dataset(
    image_paths: List[Path],
    label_paths: List[Path],
    output_base_dir: str | Path,
    preproc_fn: Callable[[np.ndarray], np.ndarray],
    suffix: str,
) -> Tuple[List[Path], List[Path]]:
    """
    Genera un dataset YOLO preprocesado a partir de imágenes + labels.

    Parámetros:
      - image_paths: Lista de imágenes originales.
      - label_paths: Lista de labels correspondientes.
      - output_base_dir: Carpeta raíz del dataset resultante.
      - preproc_fn: Función de preprocesamiento que recibe BGR y devuelve BGR.
      - suffix: Sufijo para nombres de salida (ej. 'clahe').

    Returns:
      - Tupla (new_image_paths, new_label_paths) con paths generados.
    """
    output_base = Path(output_base_dir)
    out_images = output_base / "images"
    out_labels = output_base / "labels"

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    new_image_paths: List[Path] = []
    new_label_paths: List[Path] = []

    if len(image_paths) != len(label_paths):
        raise ValueError("image_paths y label_paths deben tener misma longitud")

    for img_path, lbl_path in zip(image_paths, label_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] No se pudo leer {img_path}")
            continue

        proc = preproc_fn(img)

        new_name = f"{img_path.stem}_{suffix}{img_path.suffix}"
        out_img_path = out_images / new_name
        out_lbl_path = out_labels / f"{img_path.stem}_{suffix}.txt"

        cv2.imwrite(str(out_img_path), proc)

        if lbl_path.exists():
            txt = lbl_path.read_text(encoding="utf-8")
            out_lbl_path.write_text(txt, encoding="utf-8")
        else:
            print(f"[WARN] No se encontró label para {img_path}")

        new_image_paths.append(out_img_path)
        new_label_paths.append(out_lbl_path)

    return new_image_paths, new_label_paths
