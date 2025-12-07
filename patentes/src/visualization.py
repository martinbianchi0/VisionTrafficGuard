from pathlib import Path
from typing import Optional, Sequence, Tuple, List

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from video_utils import load_frame
from preprocessing import generate_preprocessing_versions


GREEN = (0.0, 1.0, 0.0)
RED = (1.0, 0.0, 0.0)


def show_gt_plate(
    frame_idx: int,
    df_gt: pd.DataFrame,
    video_path: Path,
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    Muestra un frame y dibuja las cajas GT de patente (en verde).
    No agrega título; eso va en el caption del informe.
    """
    frame = load_frame(video_path, frame_idx)
    if frame is None:
        return

    subset = df_gt[df_gt["frame"] == frame_idx]
    if subset.empty:
        print(f"No hay GT de patente para el frame {frame_idx}")
        return

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        created_fig = True

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame_rgb)
    ax.axis("off")

    for _, row in subset.iterrows():
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=GREEN,
            facecolor="none",
        )
        ax.add_patch(rect)

    if created_fig:
        plt.tight_layout()
        plt.show()

def _draw_detection_vs_gt_on_ax(ax, frame_bgr, row, add_legend: bool = False):
    """
    Dibuja una imagen con GT en verde, predicción en rojo e IoU.
    Si add_legend es True, agrega también la leyenda de colores adentro del eje.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(frame_rgb)
    ax.axis("off")

    x1g, y1g, x2g, y2g = int(row["x1_gt"]), int(row["y1_gt"]), int(row["x2_gt"]), int(
        row["y2_gt"]
    )
    rect_gt = patches.Rectangle(
        (x1g, y1g),
        x2g - x1g,
        y2g - y1g,
        linewidth=2,
        edgecolor=(0.0, 1.0, 0.0),
        facecolor="none",
    )
    ax.add_patch(rect_gt)

    if not np.isnan(row["x1_pred"]):
        x1p, y1p, x2p, y2p = (
            int(row["x1_pred"]),
            int(row["y1_pred"]),
            int(row["x2_pred"]),
            int(row["y2_pred"]),
        )
        rect_pred = patches.Rectangle(
            (x1p, y1p),
            x2p - x1p,
            y2p - y1p,
            linewidth=2,
            edgecolor=(1.0, 0.0, 0.0),
            facecolor="none",
        )
        ax.add_patch(rect_pred)

    iou = float(row.get("iou", np.nan))
    if not np.isnan(iou):
        ax.text(
            6,
            24,
            f"IoU={iou:.2f}",
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.8, edgecolor="none"),
        )

    if add_legend:
        ax.text(
            0.5,
            0.02,
            "Verde: ground truth   Rojo: detección",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
        )


def plot_detection_examples_grid(df_eval, video_path, example_indices, figsize=(10, 8)):
    """
    Muestra hasta 4 ejemplos en una grilla 2x2.
    En la primera imagen se dibuja también la leyenda de colores.
    """
    indices = list(example_indices)
    if len(indices) == 0:
        return
    if len(indices) > 4:
        indices = indices[:4]

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(nrows, ncols)

    for k in range(nrows * ncols):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        if k < len(indices):
            row = df_eval.loc[indices[k]]
            frame_idx = int(row["frame"])
            frame = load_frame(video_path, frame_idx)
            if frame is not None:
                add_legend = k == 0
                _draw_detection_vs_gt_on_ax(ax, frame, row, add_legend=add_legend)
            else:
                ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_detection_examples_2x1(df_eval, video_path, example_indices, figsize=(10, 10)):
    """
    Muestra hasta dos ejemplos de detección vs GT en una figura 2x1.

    Parámetros:
      - df_eval: DataFrame con GT y predicciones por patente.
      - video_path: Ruta al video original.
      - example_indices: Índices de df_eval a mostrar.
      - figsize: Tamaño de la figura (ancho, alto).

    Returns:
      - None. Solo muestra la figura.
    """
    indices = list(example_indices)
    if len(indices) == 0:
        return
    if len(indices) > 2:
        indices = indices[:2]

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    axes = np.array(axes).reshape(2,)

    for k in range(2):
        ax = axes[k]
        if k < len(indices):
            row = df_eval.loc[indices[k]]
            frame_idx = int(row["frame"])
            frame = load_frame(video_path, frame_idx)
            if frame is not None:
                add_legend = k == 0
                _draw_detection_vs_gt_on_ax(ax, frame, row, add_legend=add_legend)
            else:
                ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_detection_single(df_eval, video_path, example_index, figsize=(8, 6)):
    """
    Muestra un único ejemplo de detección vs GT en grande.

    Parámetros:
      - df_eval: DataFrame con GT y predicciones por patente.
      - video_path: Ruta al video original.
      - example_index: Índice de df_eval a mostrar.
      - figsize: Tamaño de la figura (ancho, alto).

    Returns:
      - None. Solo muestra la figura.
    """
    row = df_eval.loc[example_index]
    frame_idx = int(row["frame"])
    frame = load_frame(video_path, frame_idx)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if frame is not None:
        _draw_detection_vs_gt_on_ax(ax, frame, row, add_legend=True)
    else:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_iou_histogram(
    df_eval: pd.DataFrame,
    bins: int = 20,
    figsize: Tuple[int, int] = (6, 4),
) -> None:
    """
    Histograma de IoU sin título, solo con labels de ejes.

    Parámetros:
      - df_eval: DataFrame con la columna 'iou'.
      - bins: Cantidad de bins del histograma.
      - figsize: Tamaño de la figura (ancho, alto).

    Returns:
      - None. Solo muestra la figura.
    """
    if df_eval.empty:
        print("df_eval vacío, no hay IoU para graficar.")
        return

    plt.figure(figsize=figsize)
    df_eval["iou"].hist(bins=bins)
    plt.xlabel("IoU pred vs GT")
    plt.ylabel("Cantidad de patentes")
    plt.tight_layout()
    plt.show()

def plot_gt_coverage_bars(
    df_eval: pd.DataFrame,
    figsize: tuple[int, int] = (6, 4),
) -> None:
    """
    Muestra barras con el promedio de cobertura de la patente GT por la bbox predicha.
    """
    if df_eval.empty or "gt_covered_frac" not in df_eval.columns:
        print("df_eval vacío o sin columnas de cobertura, no se grafica nada.")
        return

    detected = df_eval[df_eval["has_pred"]].copy()
    if detected.empty:
        print("No hay detecciones para graficar cobertura.")
        return

    labels = []
    values = []

    values.append(100.0 * detected["gt_covered_frac"].mean())
    labels.append("0 px")

    if "gt_covered_frac_tol1" in detected.columns:
        values.append(100.0 * detected["gt_covered_frac_tol1"].mean())
        labels.append("±1 px")

    if "gt_covered_frac_tol2" in detected.columns:
        values.append(100.0 * detected["gt_covered_frac_tol2"].mean())
        labels.append("±2 px")

    plt.figure(figsize=figsize)
    plt.bar(labels, values)
    plt.ylim(0, 105)
    plt.ylabel("% píxeles GT dentro de bbox predicha")
    plt.tight_layout()
    plt.show()

def plot_detection_summary_table(
    df_summary: pd.DataFrame,
    figsize: Tuple[int, int] = (9, 2.2),
    fontsize: int = 9,
) -> None:
    """
    Muestra una tabla prolija con las métricas globales de detección.

    Pensada para una sola fila con columnas:
      - mean_iou_all
      - mean_iou_detected
      - center_inside_pct
      - gt_covered_pct
      - gt_covered_pct_tol1
      - gt_covered_pct_tol2
    """
    if df_summary is None or df_summary.empty:
        return

    row = df_summary.iloc[0]

    data = [[
        f"{row['mean_iou_all']:.2f}",
        f"{row['mean_iou_detected']:.2f}",
        f"{row['center_inside_pct']:.1f}",
        f"{row['gt_covered_pct']:.1f}",
        f"{row['gt_covered_pct_tol1']:.1f}",
        f"{row['gt_covered_pct_tol2']:.1f}",
    ]]

    col_labels = [
        "IoU\n(todos)",
        "IoU\n(detectados)",
        "Centro dentro\nGT [%]",
        "GT cubierto\n0 px [%]",
        "GT cubierto\n±1 px [%]",
        "GT cubierto\n±2 px [%]",
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.1, 1.6)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_edgecolor("#000000")
            cell.set_text_props(weight="bold")
        else:
            cell.set_edgecolor("#000000")

    plt.tight_layout()
    plt.show()


def plot_gt_examples_grid(df_gt, video_path, frame_indices, figsize=(10, 8)):
    """
    Muestra hasta cuatro frames con GT en una figura 2x2.

    Parámetros:
      - df_gt: DataFrame de ground truth de patentes.
      - video_path: Ruta al video original.
      - frame_indices: Lista de números de frame a mostrar.
      - figsize: Tamaño de la figura (ancho, alto).

    Returns:
      - None. Solo muestra la figura.
    """
    frames = list(frame_indices)
    if len(frames) == 0:
        return
    if len(frames) > 4:
        frames = frames[:4]

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(nrows, ncols)

    for k in range(nrows * ncols):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        if k < len(frames):
            show_gt_plate(int(frames[k]), df_gt, video_path, ax=ax)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_gt_examples_2x1(df_gt, video_path, frame_indices, figsize=(10, 10)):
    """
    Muestra hasta dos frames con GT en una figura 2x1.

    Parámetros:
      - df_gt: DataFrame de ground truth de patentes.
      - video_path: Ruta al video original.
      - frame_indices: Lista de números de frame a mostrar.
      - figsize: Tamaño de la figura (ancho, alto).

    Returns:
      - None. Solo muestra la figura.
    """
    frames = list(frame_indices)
    if len(frames) == 0:
        return
    if len(frames) > 2:
        frames = frames[:2]

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    axes = np.array(axes).reshape(2,)

    for k in range(2):
        ax = axes[k]
        if k < len(frames):
            show_gt_plate(int(frames[k]), df_gt, video_path, ax=ax)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_gt_single(df_gt, video_path, frame_index, figsize=(8, 6)):
    """
    Muestra un único frame con GT en grande.

    Parámetros:
      - df_gt: DataFrame de ground truth de patentes.
      - video_path: Ruta al video original.
      - frame_index: Número de frame a mostrar.
      - figsize: Tamaño de la figura (ancho, alto).

    Returns:
      - None. Solo muestra la figura.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    show_gt_plate(int(frame_index), df_gt, video_path, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_plate_preprocessing_grid(
    plate_bgr: np.ndarray,
    variants: List[str],
    figsize: tuple[int, int] = (12, 4),
) -> None:
    """
    Muestra una patente y sus variantes de preprocesamiento en una figura.

    Parámetros:
      - plate_bgr: Imagen de patente en BGR.
      - variants: Lista de nombres de preprocesamiento.
      - figsize: Tamaño de la figura.
    """
    if len(variants) == 0:
        return

    images = generate_preprocessing_versions(plate_bgr, variants)
    n = len(variants)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    axes = np.array(axes).reshape(1, n)

    for ax, v in zip(axes.ravel(), variants):
        img = images[v]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_fastplate_examples_grid(
    df_results: pd.DataFrame,
    video_path: Path,
    n_good: int = 4,
    n_bad: int = 4,
    figsize=(10, 8),
):
    """
    Muestra ejemplos buenos y malos de FastPlate en una figura.

    Parámetros:
      - df_results: DataFrame con resultados de OCR.
      - video_path: Ruta al video original.
      - n_good: Cantidad de ejemplos correctos a mostrar.
      - n_bad: Cantidad de ejemplos incorrectos a mostrar.
      - figsize: Tamaño de la figura (ancho, alto).

    Returns:
      - None. Solo muestra la figura.
    """
    df_fp = df_results[df_results["method"] == "fastplate"].copy()
    df_fp["exact_match"] = df_fp["plate_text_pred"] == df_fp["plate_text_gt"]

    good = df_fp[df_fp["exact_match"]].head(n_good)
    bad = df_fp[~df_fp["exact_match"]].head(n_bad)

    df_show = pd.concat([good, bad], axis=0)
    indices = df_show.index.tolist()

    n = len(indices)
    if n == 0:
        return

    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(nrows, ncols)

    for k, idx in enumerate(indices):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        row = df_show.loc[idx]
        frame_idx = int(row["frame"])
        frame = load_frame(video_path, frame_idx)
        if frame is None:
            ax.axis("off")
            continue

        x1 = int(row["x1_pred"])
        y1 = int(row["y1_pred"])
        x2 = int(row["x2_pred"])
        y2 = int(row["y2_pred"])

        patch = frame[y1:y2, x1:x2]
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        ax.imshow(patch_rgb)
        ax.axis("off")

        text = f"GT: {row['plate_text_gt']}\nPred: {row['plate_text_pred']}"
        ax.text(
            0.5,
            0.02,
            text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color="white",
            bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"),
        )

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()

def _select_fastplate_examples(
    df_results: pd.DataFrame,
    n: int,
    mode: str = "good",
) -> pd.DataFrame:
    """
    Selecciona hasta n ejemplos de fastplate, buenos o malos según mode.
    """
    df_fast = df_results[df_results["method"] == "fastplate"].copy()
    print(f"[_select_fastplate_examples] filas fastplate: {len(df_fast)}")
    if df_fast.empty:
        return df_fast

    df_fast["len_pred"] = df_fast["plate_text_pred"].astype(str).str.len()
    df_fast["good_len"] = df_fast["len_pred"].between(6, 8).astype(int)
    df_fast["score_good"] = (
        df_fast["looks_plate"].astype(int) * 3
        + df_fast["non_empty"].astype(int) * 2
        + df_fast["good_len"] * 1
        + df_fast.get("iou_det", 0.0).fillna(0.0)
    )

    ascending = mode != "good"
    df_sorted = df_fast.sort_values("score_good", ascending=ascending)
    df_unique = df_sorted.drop_duplicates(subset=["frame", "lane", "xml_idx"])
    df_sel = df_unique.head(n)
    print(f"[_select_fastplate_examples] ejemplos seleccionados ({mode}): {len(df_sel)}")
    return df_sel


def _plot_fastplate_grid(
    df_results: pd.DataFrame,
    n: int,
    nrows: int,
    ncols: int,
    mode: str,
    figsize: tuple[int, int],
) -> None:
    """
    Dibuja una grilla de ejemplos de fastplate sobre recortes de patente.
    """
    df_sel = _select_fastplate_examples(df_results, n=n, mode=mode)
    print(
        f"[_plot_fastplate_grid] grid {nrows}x{ncols}, pedidos={n}, "
        f"reales={len(df_sel)}, mode={mode}"
    )
    if df_sel.empty:
        print("[_plot_fastplate_grid] no hay ejemplos para mostrar.")
        return

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(nrows, ncols)

    rows = list(df_sel.itertuples(index=False))
    for k in range(nrows * ncols):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        if k < len(rows):
            row = rows[k]
            patch = getattr(row, "plate_img_bgr", None)
            if patch is None or getattr(patch, "size", 0) == 0:
                ax.axis("off")
                continue

            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            ax.imshow(patch_rgb)
            ax.axis("off")

            gt_text = getattr(row, "plate_text_gt", "") or ""
            pred_text = getattr(row, "plate_text_pred", "") or ""

            ax.text(
                0.5,
                0.02,
                f"FastPlate: {pred_text}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color="yellow",
                bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"),
            )
            ax.text(
                0.5,
                0.18,
                f"GT: {gt_text}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                color="white",
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
            )
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_fastplate_best_2x2(
    df_results: pd.DataFrame,
    mode: str = "good",
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """
    Muestra hasta 4 ejemplos de fastplate en una grilla 2x2.
    """
    _plot_fastplate_grid(
        df_results=df_results,
        n=4,
        nrows=2,
        ncols=2,
        mode=mode,
        figsize=figsize,
    )


def plot_fastplate_best_2x1(
    df_results: pd.DataFrame,
    mode: str = "good",
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Muestra hasta 2 ejemplos de fastplate en una grilla 2x1.
    """
    _plot_fastplate_grid(
        df_results=df_results,
        n=2,
        nrows=2,
        ncols=1,
        mode=mode,
        figsize=figsize,
    )


def plot_fastplate_best_single(
    df_results: pd.DataFrame,
    mode: str = "good",
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """
    Muestra un único ejemplo de fastplate en grande.
    """
    _plot_fastplate_grid(
        df_results=df_results,
        n=1,
        nrows=1,
        ncols=1,
        mode=mode,
        figsize=figsize,
    )

def plot_ocr_summary_bars(
    df_summary: pd.DataFrame,
    metric: str = "looks_plate_pct",
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """
    Barplot de una métrica de OCR por combinación método-preproc.
    """
    print(f"[plot_ocr_summary_bars] filas df_summary: {len(df_summary)}, métrica: {metric}")

    if df_summary is None or df_summary.empty:
        print("[plot_ocr_summary_bars] df_summary vacío, no se grafica nada.")
        return

    required_cols = {"method", "preproc", metric}
    if not required_cols.issubset(df_summary.columns):
        print(
            f"[plot_ocr_summary_bars] columnas faltantes para la métrica '{metric}': "
            f"{required_cols - set(df_summary.columns)}"
        )
        return

    df = df_summary.copy()
    labels = df["method"] + "_" + df["preproc"]
    values = df[metric]

    plt.figure(figsize=figsize)
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")

    ylabel = metric.replace("_", " ")
    if metric.endswith("_pct"):
        ylabel += " (%)"
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.show()


def plot_table_as_image(
    df: pd.DataFrame,
    max_rows: int | None = None,
    figsize: tuple[int, int] = (10, 3),
    fontsize: int = 8,
) -> None:
    """
    Muestra un DataFrame como figura (tabla renderizada en matplotlib).
    """
    if df is None or df.empty:
        print("[plot_table_as_image] DataFrame vacío, no se grafica nada.")
        return

    df_plot = df.copy()

    if "accuracy" in df_plot.columns:
        df_plot = df_plot.drop(columns=["accuracy"])
    if "total" in df_plot.columns:
        df_plot = df_plot.drop(columns=["total"])

    if max_rows is not None and len(df_plot) > max_rows:
        df_plot = df_plot.head(max_rows)

    for col in df_plot.columns:
        if df_plot[col].dtype.kind in ("f", "i"):
            if str(col).endswith("_pct"):
                df_plot[col] = df_plot[col].map(lambda x: f"{x:.1f}")
            else:
                df_plot[col] = df_plot[col].map(lambda x: f"{x:.2f}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=df_plot.values,
        colLabels=df_plot.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", weight="bold")
        elif col == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    plt.show()