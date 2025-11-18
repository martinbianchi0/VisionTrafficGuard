from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import random


def apply_dirty_effect(img_bgr: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """
    Aplica un efecto tipo 'arena/polvo' a una imagen BGR.

    Combina cambios de brillo/contraste, color, blur, niebla y una capa de polvo.
    La idea es simular baja visibilidad, colores lavados y ligera dominante cálida.

    Parámetros:
      - img_bgr: Imagen original en BGR (cv2.imread).
      - strength: Intensidad global del efecto, entre 0.0 (casi nada) y 1.0 (muy agresivo).

    Returns:
      - Imagen BGR modificada, mismo tamaño que la original.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    img = img_bgr.astype(np.float32) / 255.0

    # ---------------------------
    # 1) Brillo y contraste
    # ---------------------------
    c_min, c_max = 0.4, 0.9
    alpha_rand = random.uniform(c_min, c_max)
    alpha = (1.0 - strength) * 1.0 + strength * alpha_rand

    beta = random.uniform(0.0, 0.3) * strength

    img_bc = np.clip(alpha * img + beta, 0.0, 1.0)

    # ---------------------------
    # 2) Jitter de color en HSV
    # ---------------------------
    img_hsv = cv2.cvtColor((img_bc * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    max_dh = int(10 * strength)
    max_ds = int(40 * strength)
    max_dv = int(30 * strength)

    dh = random.randint(-max_dh, max_dh) if max_dh > 0 else 0
    ds = random.randint(-max_ds, max_ds) if max_ds > 0 else 0
    dv = random.randint(-max_dv, max_dv) if max_dv > 0 else 0

    h = (h.astype(np.int32) + dh) % 180
    s = np.clip(s.astype(np.int32) + ds, 0, 255)
    v = np.clip(v.astype(np.int32) + dv, 0, 255)

    img_hsv_j = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
    img_jitter = cv2.cvtColor(img_hsv_j, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # ---------------------------
    # 3) Blur suave
    # ---------------------------
    if random.random() < 0.8 * strength + 0.2:
        k = random.choice([3, 5, 7])
        img_blur = cv2.GaussianBlur(img_jitter, (k, k), 0)
    else:
        img_blur = img_jitter

    # ---------------------------
    # 4) Capa de polvo (dust veil)
    # ---------------------------
    h_img, w_img, _ = img_blur.shape

    # Ruido base 2D
    noise = np.random.normal(loc=0.7, scale=0.15, size=(h_img, w_img)).astype(np.float32)
    noise = np.clip(noise, 0.0, 1.0)

    ksize = random.choice([7, 11, 15])
    noise_blur = cv2.GaussianBlur(noise, (ksize, ksize), 0)

    # Aseguramos que tenga canal extra: (H, W, 1)
    if noise_blur.ndim == 2:
        noise_blur = noise_blur[..., None]

    dust_color = np.array([0.85, 0.78, 0.60], dtype=np.float32).reshape(1, 1, 3)
    dust_layer = np.clip(noise_blur * dust_color, 0.0, 1.0)

    dust_alpha = random.uniform(0.3, 0.8) * strength
    img_dust = (1.0 - dust_alpha) * img_blur + dust_alpha * dust_layer

    # ---------------------------
    # 5) Niebla adicional
    # ---------------------------
    if random.random() < 0.6:
        fog_strength = random.uniform(0.15, 0.45) * (0.5 + strength / 2.0)
        fog_color = np.full_like(img_dust, random.uniform(0.75, 0.9))
        img_fog = (1.0 - fog_strength) * img_dust + fog_strength * fog_color
    else:
        img_fog = img_dust

    img_out = np.clip(img_fog * 255.0, 0, 255).astype(np.uint8)
    return img_out


def save_dirty_copies(
    image_paths: List[Path],
    output_dir: str | Path,
    max_images: Optional[int] = None,
    strength: float = 0.6,
) -> List[Path]:
    """
    Genera copias "sucias" de un conjunto de imágenes y las guarda en output_dir.

    Usa apply_dirty_effect() sobre cada imagen y mantiene el mismo nombre de archivo.

    Parámetros:
      - image_paths: Lista de Paths de imágenes originales.
      - output_dir: Carpeta donde se guardarán las copias modificadas.
      - max_images: Si no es None, limita la cantidad de imágenes procesadas.
      - strength: Intensidad global de la suciedad (0.0 a 1.0).

    Returns:
      - Lista de Paths de las nuevas imágenes sucias.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dirty_paths: List[Path] = []

    paths_to_process = image_paths
    if max_images is not None:
        paths_to_process = image_paths[:max_images]

    for img_path in paths_to_process:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] No se pudo leer {img_path}")
            continue

        dirty = apply_dirty_effect(img, strength=strength)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), dirty)
        dirty_paths.append(out_path)

    print(f"Generadas {len(dirty_paths)} imágenes sucias en: {out_dir}")
    return dirty_paths
