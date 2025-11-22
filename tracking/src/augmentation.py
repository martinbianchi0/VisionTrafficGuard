from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import random

def apply_dirty_effect(img_bgr: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """
    Aplica un efecto tipo arena/polvo a una imagen BGR.

    Parámetros:
      - img_bgr: Imagen original en BGR.
      - strength: Intensidad global del efecto (0.0 a 1.0).

    Returns:
      - Imagen BGR modificada con suciedad y niebla.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    img = img_bgr.astype(np.float32) / 255.0

    c_min, c_max = 0.4, 0.9
    alpha_rand = random.uniform(c_min, c_max)
    alpha = (1.0 - strength) + strength * alpha_rand
    beta = random.uniform(0.0, 0.3) * strength
    img_bc = np.clip(alpha * img + beta, 0.0, 1.0)

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

    if random.random() < 0.8 * strength + 0.2:
        k = random.choice([3, 5, 7])
        img_blur = cv2.GaussianBlur(img_jitter, (k, k), 0)
    else:
        img_blur = img_jitter

    h_img, w_img, _ = img_blur.shape
    noise = np.random.normal(loc=0.7, scale=0.15, size=(h_img, w_img)).astype(np.float32)
    noise = np.clip(noise, 0.0, 1.0)

    ksize = random.choice([7, 11, 15])
    noise_blur = cv2.GaussianBlur(noise, (ksize, ksize), 0)
    if noise_blur.ndim == 2:
        noise_blur = noise_blur[..., None]

    dust_color = np.array([0.85, 0.78, 0.60], dtype=np.float32).reshape(1, 1, 3)
    dust_layer = np.clip(noise_blur * dust_color, 0.0, 1.0)

    dust_alpha = random.uniform(0.3, 0.8) * strength
    img_dust = (1.0 - dust_alpha) * img_blur + dust_alpha * dust_layer

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
    Genera copias sucias de un conjunto de imágenes y las guarda en disco.

    Parámetros:
      - image_paths: Lista de imágenes originales.
      - output_dir: Carpeta donde se guardan las copias.
      - max_images: Límite opcional de imágenes a procesar.
      - strength: Intensidad del efecto de suciedad.

    Returns:
      - Lista de Paths de las nuevas imágenes sucias.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dirty_paths: List[Path] = []
    paths_to_process = image_paths if max_images is None else image_paths[:max_images]

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

def apply_fog(img, strength=0.3):
    '''
    Aplica un efecto de niebla a una imagen BGR.
    
    Parámetros:
      - img: Imagen original en BGR.
      - strength: Intensidad de la niebla (0.0 a 1.0).

    Returns:
      - Imagen BGR modificada con niebla.
    '''
    strength = np.clip(strength, 0.0, 1.0)
    h, w = img.shape[:2]
    fog = np.full((h, w, 3), 255, dtype=np.float32)  # capa blanca
    
    noise = cv2.GaussianBlur(
        np.random.normal(loc=1.0, scale=0.5, size=(h, w)).astype(np.float32),
        (51, 51), 0
    )
    if noise.ndim == 2:
        noise = noise[..., None] # (H, W, 1)

    fog_layer = (fog * noise).clip(0, 255)

    alpha = 0.3 + 0.7 * strength
    out = cv2.addWeighted(img.astype(np.float32), 1 - alpha, fog_layer, alpha, 0)
    return out.astype(np.uint8)

def apply_rain(img, strength=0.5):
    '''
    Aplica un efecto de lluvia a una imagen BGR.

    Parámetros:
      - img: Imagen original en BGR.
      - strength: Intensidad de la lluvia (0.0 a 1.0).

    Returns:
      - Imagen BGR modificada con lluvia.
    '''
    strength = np.clip(strength, 0.0, 1.0)
    h, w = img.shape[:2]

    rain_layer = np.zeros((h, w), dtype=np.float32)

    # cantidad de gotas
    drops = int(800 * strength)

    for _ in range(drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(10, 20)
        thickness = 1

        cv2.line(
            rain_layer,
            (x, y),
            (x + np.random.randint(-2, 2), y + length),
            color=1.0,
            thickness=thickness
        )

    # motion blur para que parezca lluvia real
    ksize = int(5 + strength * 10)
    rain_layer = cv2.blur(rain_layer, (ksize, 1))

    rain_layer = np.dstack([rain_layer]*3) * 255

    alpha = 0.2 + 0.3 * strength
    out = cv2.addWeighted(img.astype(np.float32), 1.0, rain_layer.astype(np.float32), alpha, 0)
    return out.astype(np.uint8)


def apply_snow(img, strength=0.5):
    '''
    Aplica un efecto de nieve a una imagen BGR.

    Parámetros:
      - img: Imagen original en BGR.
      - strength: Intensidad de la nieve (0.0 a 1.0).

    Returns:
      - Imagen BGR modificada con nieve.
    '''
    strength = np.clip(strength, 0.0, 1.0)
    h, w = img.shape[:2]

    snow = np.random.normal(loc=200, scale=55, size=(h, w)).astype(np.float32)
    snow = cv2.GaussianBlur(snow, (5, 5), 0)
    snow = np.clip(snow, 180, 255)

    snow = np.dstack([snow]*3)

    alpha = 0.1 + 0.4 * strength
    out = cv2.addWeighted(img.astype(np.float32), 1 - alpha, snow, alpha, 0)
    return out.astype(np.uint8)

def apply_night(img, strength=0.5):
    '''
    Aplica un efecto de noche a una imagen BGR.

    Parámetros:
      - img: Imagen original en BGR.
      - strength: Intensidad del efecto noche (0.0 a 1.0).

    Returns:
      - Imagen BGR modificada con efecto noche.
    '''
    strength = np.clip(strength, 0.0, 1.0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # oscurecer
    v *= (0.4 + 0.6 * (1 - strength))
    v = np.clip(v, 0, 255)

    hsv_dark = cv2.merge([h, s, v]).astype(np.uint8)
    out = cv2.cvtColor(hsv_dark, cv2.COLOR_HSV2BGR)

    # ruido para simular ISO alto
    noise = (np.random.randn(*img.shape) * (10 + 40 * strength)).astype(np.int16)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return out

def apply_random_weather(img):
    '''
    Aplica un efecto climático aleatorio a una imagen BGR.

    Parámetros:
      - img: Imagen original en BGR.
      
    Returns:
      - Imagen BGR modificada con un efecto climático aleatorio.
    '''
    r = random.random()

    if r < 0.25:
        return apply_fog(img, strength=random.uniform(0.3, 0.8)), 'FOG'
    elif r < 0.50:
        return apply_rain(img, strength=random.uniform(0.3, 0.8)), 'RAIN'
    elif r < 0.75:
        return apply_snow(img, strength=random.uniform(0.2, 0.7)), 'SNOW'
    else:
        return apply_night(img, strength=random.uniform(0.3, 0.7)), 'NIGHT'

