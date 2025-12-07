import cv2
import numpy as np
from typing import Dict, List


def preprocess_plate_variant(plate_bgr: np.ndarray, variant: str) -> np.ndarray:
    """
    Aplica un esquema de preprocesamiento sobre una patente.

    Parámetros:
      - plate_bgr: Imagen de patente en BGR.
      - variant: Nombre de la variante a aplicar.

    Returns:
      - Imagen procesada en escala de grises o binaria.
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

    if variant == "none":
        return gray

    if variant == "morph":
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        return closed

    if variant == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        return cl

    if variant == "bilateral_adaptive":
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        th = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            21,
            10,
        )
        return th

    return gray


def generate_preprocessing_versions(
    plate_bgr: np.ndarray,
    variants: List[str],
) -> Dict[str, np.ndarray]:
    """
    Genera múltiples versiones preprocesadas de una patente.

    Parámetros:
      - plate_bgr: Imagen de patente en BGR.
      - variants: Lista de nombres de variantes a aplicar.

    Returns:
      - Dict que mapea nombre de variante a imagen procesada.
    """
    out: Dict[str, np.ndarray] = {}
    for v in variants:
        out[v] = preprocess_plate_variant(plate_bgr, v)
    return out