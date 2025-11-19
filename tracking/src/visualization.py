from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


def preview_sequence_grid(
    image_paths: List[Path],
    sequence_filter: Optional[str] = None,
    n_rows: int = 2,
    n_cols: int = 3,
    random_sample: bool = False,
) -> None:
    """
    Muestra un grid de imágenes crudas (sin pasar por YOLO) de una secuencia.

    Parámetros:
      - image_paths: lista de Paths de imágenes (por ejemplo subset['image_paths']).
      - sequence_filter: si no es None, filtra por secuencia, ej. 'MVI_20011'.
      - n_rows: cantidad de filas en el grid.
      - n_cols: cantidad de columnas en el grid.
      - random_sample: si True, elige imágenes al azar; si False, toma las primeras K ordenadas.

    Muestra:
      - Una figura matplotlib con n_rows * n_cols imágenes como máximo.
    """
    if not image_paths:
        print("La lista de imágenes está vacía.")
        return

    # Filtrar por secuencia si se pide (por nombre de archivo)
    filtered = image_paths
    if sequence_filter is not None:
        filtered = [p for p in image_paths if sequence_filter in p.name]

    if not filtered:
        print(f"No se encontraron imágenes que contengan '{sequence_filter}' en el nombre.")
        return

    # Ordenar por nombre para que queden en orden temporal
    filtered = sorted(filtered, key=lambda p: p.name)

    # Cantidad máxima a mostrar
    max_imgs = n_rows * n_cols

    if random_sample and len(filtered) > max_imgs:
        # Muestreo aleatorio sin reemplazo
        import random
        filtered = random.sample(filtered, max_imgs)
    else:
        # Tomar primeras K (o todas si hay menos)
        filtered = filtered[:max_imgs]

    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)  # por si matplotlib devuelve 1D

    for idx, img_path in enumerate(filtered):
        row = idx // n_cols
        col = idx % n_cols

        ax = axes[row, col]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            ax.set_title(f"Error leyendo\n{img_path.name}", fontsize=8)
            ax.axis("off")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        ax.imshow(img_rgb)
        ax.set_title(img_path.name, fontsize=8)
        ax.axis("off")

    # Celdas sobrantes (si hay menos imágenes que slots en el grid)
    total_slots = n_rows * n_cols
    for empty_idx in range(len(filtered), total_slots):
        row = empty_idx // n_cols
        col = empty_idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()

def show_preproc_panel(img_path, preprocs_dict, title_prefix=""):
    """
    Muestra raw + 5 preprocesamientos en una grilla 2x3.
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print("No pude leer la imagen:", img_path)
        return

    panels = {
        "raw": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
    }

    for name, fn in preprocs_dict.items():
       proc = fn(img_bgr.copy())
       panels[name] = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

    names = list(panels.keys())
    n = len(names)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    for ax, name in zip(axes, names):
        ax.imshow(panels[name])
        ax.set_title(name, fontsize=9)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_yolo_predictions_grid(
    model: YOLO,
    image_paths: List[Path],
    n_rows: int = 2,
    n_cols: int = 3,
    conf: float = 0.25,
    sequence_filter: Optional[str] = None,
):
    """
    Muestra un grid de predicciones de YOLO sobre imágenes crudas.

    Parámetros:
      - model: Instancia de YOLO ya entrenada / fine-tuneada.
      - image_paths: Lista de Paths de imágenes.
      - n_rows: Filas del grid.
      - n_cols: Columnas del grid.
      - conf: Umbral de confianza para las predicciones.
      - sequence_filter: Si no es None, filtra por nombre, ej. 'MVI_20011'.

    Muestra:
      - Una figura matplotlib con las imágenes anotadas por YOLO.
    """
    if not image_paths:
        print("La lista de imágenes está vacía.")
        return

    filtered = image_paths
    if sequence_filter is not None:
        filtered = [p for p in filtered if sequence_filter in p.name]

    if not filtered:
        print(f"No se encontraron imágenes con filtro '{sequence_filter}'.")
        return

    filtered = sorted(filtered, key=lambda p: p.name)

    max_imgs = n_rows * n_cols
    filtered = filtered[:max_imgs]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, img_path in enumerate(filtered):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # correr YOLO en esta imagen
        results = model.predict(
            source=str(img_path),
            conf=conf,
            verbose=False
        )
        if not results:
            ax.set_title(f"Sin resultados\n{img_path.name}", fontsize=8)
            ax.axis("off")
            continue

        # results[0].plot() devuelve imagen BGR anotada
        annotated_bgr = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        ax.imshow(annotated_rgb)
        ax.set_title(img_path.name, fontsize=8)
        ax.axis("off")

    # apagar ejes sobrantes si hay menos imágenes que slots
    total_slots = n_rows * n_cols
    for empty_idx in range(len(filtered), total_slots):
        row = empty_idx // n_cols
        col = empty_idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()
