from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from detector_luvizon import detect_plates_in_frame
from groundtruth import iou_xyxy


def _add_geometry_metrics(df_eval: pd.DataFrame) -> pd.DataFrame:
    df = df_eval.copy()

    df["cx_gt"] = (df["x1_gt"] + df["x2_gt"]) / 2.0
    df["cy_gt"] = (df["y1_gt"] + df["y2_gt"]) / 2.0
    df["cx_pred"] = (df["x1_pred"] + df["x2_pred"]) / 2.0
    df["cy_pred"] = (df["y1_pred"] + df["y2_pred"]) / 2.0

    dx = df["cx_gt"] - df["cx_pred"]
    dy = df["cy_gt"] - df["cy_pred"]
    df["center_dist"] = np.sqrt(dx**2 + dy**2)

    w_gt = df["x2_gt"] - df["x1_gt"]
    h_gt = df["y2_gt"] - df["y1_gt"]
    df["area_gt"] = w_gt * h_gt
    df["diag_gt"] = np.sqrt(w_gt**2 + h_gt**2)

    w_pred = df["x2_pred"] - df["x1_pred"]
    h_pred = df["y2_pred"] - df["y1_pred"]
    df["area_pred"] = w_pred * h_pred

    df["norm_center_dist"] = df["center_dist"] / df["diag_gt"]
    df.loc[~df["has_pred"], ["center_dist", "norm_center_dist"]] = np.nan

    df["center_inside_gt"] = (
        (df["cx_pred"] >= df["x1_gt"])
        & (df["cx_pred"] <= df["x2_gt"])
        & (df["cy_pred"] >= df["y1_gt"])
        & (df["cy_pred"] <= df["y2_gt"])
        & df["has_pred"]
    )

    df["pred_inside_gt"] = (
        (df["x1_pred"] >= df["x1_gt"])
        & (df["y1_pred"] >= df["y1_gt"])
        & (df["x2_pred"] <= df["x2_gt"])
        & (df["y2_pred"] <= df["y2_gt"])
        & df["has_pred"]
    )

    df["gt_inside_pred"] = (
        (df["x1_gt"] >= df["x1_pred"])
        & (df["y1_gt"] >= df["y1_pred"])
        & (df["x2_gt"] <= df["x2_pred"])
        & (df["y2_gt"] <= df["y2_pred"])
        & df["has_pred"]
    )

    ratio = df["area_pred"] / df["area_gt"]
    df["similar_size"] = (
        df["has_pred"]
        & ratio.notna()
        & (ratio >= 0.5)
        & (ratio <= 2.0)
    )

    gt_cov = np.full(len(df), np.nan, dtype=float)
    gt_cov_tol1 = np.full(len(df), np.nan, dtype=float)
    gt_cov_tol2 = np.full(len(df), np.nan, dtype=float)

    for idx, row in df[df["has_pred"]].iterrows():
        area_gt = row["area_gt"]
        if not np.isfinite(area_gt) or area_gt <= 0:
            continue

        x1g, y1g, x2g, y2g = (
            float(row["x1_gt"]),
            float(row["y1_gt"]),
            float(row["x2_gt"]),
            float(row["y2_gt"]),
        )
        x1p, y1p, x2p, y2p = (
            float(row["x1_pred"]),
            float(row["y1_pred"]),
            float(row["x2_pred"]),
            float(row["y2_pred"]),
        )

        def cov_for_box(x1p_, y1p_, x2p_, y2p_) -> float:
            xA = max(x1g, x1p_)
            yA = max(y1g, y1p_)
            xB = min(x2g, x2p_)
            yB = min(y2g, y2p_)
            inter_w = max(0.0, xB - xA)
            inter_h = max(0.0, yB - yA)
            inter = inter_w * inter_h
            if inter <= 0.0:
                return 0.0
            return inter / area_gt

        gt_cov[idx] = cov_for_box(x1p, y1p, x2p, y2p)
        gt_cov_tol1[idx] = cov_for_box(x1p - 1.0, y1p - 1.0, x2p + 1.0, y2p + 1.0)
        gt_cov_tol2[idx] = cov_for_box(x1p - 2.0, y1p - 2.0, x2p + 2.0, y2p + 2.0)

    df["gt_covered_frac"] = gt_cov
    df["gt_covered_frac_tol1"] = gt_cov_tol1
    df["gt_covered_frac_tol2"] = gt_cov_tol2

    return df


def run_luvizon_detection_on_gt(
    df_gt: pd.DataFrame,
    video_path: Path,
    rois_by_frame: Optional[Dict[int, List[Tuple[int, int, int, int]]]] = None,
    max_frames: Optional[int] = None,
    iou_threshold: float = 0.5,
    verbose: bool = True,
    **luvizon_kwargs,
) -> pd.DataFrame:
    """
    Corre el detector Luvizon-like sobre el video y matchea con GT de patentes.

    Agrega también la columna plate_text_gt copiando el texto del GT
    si hay alguna columna de texto disponible en df_gt.
    """
    if df_gt.empty:
        return pd.DataFrame()

    # Detectar columna de texto GT, si existe
    text_col = None
    for candidate in ["plate_text_gt", "plate", "text", "lp", "license_plate"]:
        if candidate in df_gt.columns:
            text_col = candidate
            break

    unique_frames = sorted(int(f) for f in df_gt["frame"].unique())
    if max_frames is not None:
        unique_frames = unique_frames[:max_frames]

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if verbose:
        print(f"Total frames video: {total_frames}")
        print(f"Frames usados en evaluación: {len(unique_frames)}")

    rows: List[Dict] = []

    for i, frame_idx in enumerate(unique_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            if verbose:
                print(f"[run_luvizon_detection_on_gt] No se pudo leer frame {frame_idx}")
            continue

        rois = None
        if rois_by_frame is not None:
            rois = rois_by_frame.get(int(frame_idx))

        preds: List[Tuple[int, int, int, int]] = detect_plates_in_frame(
            frame, rois=rois, **luvizon_kwargs
        )

        frame_gt = df_gt[df_gt["frame"] == frame_idx]
        for _, gt in frame_gt.iterrows():
            box_gt = (
                int(gt["x1"]),
                int(gt["y1"]),
                int(gt["x2"]),
                int(gt["y2"]),
            )

            if preds:
                best_iou = 0.0
                best_pred: Optional[Tuple[int, int, int, int]] = None
                for (x1_p, y1_p, x2_p, y2_p) in preds:
                    box_pred = (int(x1_p), int(y1_p), int(x2_p), int(y2_p))
                    iou = iou_xyxy(box_gt, box_pred)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = box_pred

                has_pred = best_pred is not None and best_iou > 0.0
                iou = float(best_iou)
                if best_pred is not None:
                    x1_p, y1_p, x2_p, y2_p = best_pred
                else:
                    x1_p = y1_p = x2_p = y2_p = np.nan
            else:
                has_pred = False
                iou = 0.0
                x1_p = y1_p = x2_p = y2_p = np.nan

            gt_text = gt[text_col] if text_col is not None else None

            rows.append(
                {
                    "frame": int(frame_idx),
                    "xml_idx": int(gt["xml_idx"]),
                    "lane": int(gt["lane"]),
                    "x1_gt": box_gt[0],
                    "y1_gt": box_gt[1],
                    "x2_gt": box_gt[2],
                    "y2_gt": box_gt[3],
                    "x1_pred": x1_p,
                    "y1_pred": y1_p,
                    "x2_pred": x2_p,
                    "y2_pred": y2_p,
                    "iou": float(iou),
                    "has_pred": bool(has_pred),
                    "plate_text_gt": gt_text,
                }
            )

        if verbose and (i + 1) % 20 == 0:
            print(f"Procesados {i+1}/{len(unique_frames)} frames...")

    cap.release()
    df_eval = pd.DataFrame(rows)
    df_eval = _add_geometry_metrics(df_eval)
    return df_eval

def summarize_detection_metrics(df_eval: pd.DataFrame):
    if df_eval.empty:
        return {}, pd.DataFrame(), {}

    total = int(len(df_eval))
    with_pred = int(df_eval["has_pred"].sum())
    detected = df_eval[df_eval["has_pred"]]

    overall = {
        "total_gt": total,
        "with_pred": with_pred,
        "with_pred_pct": 100.0 * with_pred / total if total > 0 else 0.0,
        "mean_iou_all": float(df_eval["iou"].mean()),
    }

    if detected.empty:
        per_detected = {}
    else:
        n_det = len(detected)
        per_detected = {
            "n_detected": n_det,
            "center_inside_pct": 100.0 * detected["center_inside_gt"].mean(),
            "pred_inside_gt_pct": 100.0 * detected["pred_inside_gt"].mean(),
            "gt_inside_pred_pct": 100.0 * detected["gt_inside_pred"].mean(),
            "similar_size_pct": 100.0 * detected["similar_size"].mean(),
            "mean_iou_detected": float(detected["iou"].mean()),
            "mean_norm_center_dist_detected": float(
                detected["norm_center_dist"].mean()
            ),
            "mean_gt_covered_pct": float(
                detected["gt_covered_frac"].mean() * 100.0
            ),
            "mean_gt_covered_pct_tol1": float(
                detected["gt_covered_frac_tol1"].mean() * 100.0
            ),
            "mean_gt_covered_pct_tol2": float(
                detected["gt_covered_frac_tol2"].mean() * 100.0
            ),
        }

    lane_stats = pd.DataFrame()
    if not detected.empty:
        lane_stats = (
            detected.groupby("lane")
            .agg(
                n_detected=("iou", "size"),
                center_inside_pct=("center_inside_gt", lambda s: 100.0 * s.mean()),
                pred_inside_gt_pct=("pred_inside_gt", lambda s: 100.0 * s.mean()),
                gt_inside_pred_pct=("gt_inside_pred", lambda s: 100.0 * s.mean()),
                similar_size_pct=("similar_size", lambda s: 100.0 * s.mean()),
                mean_iou_detected=("iou", "mean"),
                mean_norm_center_dist_detected=("norm_center_dist", "mean"),
                mean_gt_covered_pct=("gt_covered_frac", lambda s: 100.0 * s.mean()),
                mean_gt_covered_pct_tol1=(
                    "gt_covered_frac_tol1",
                    lambda s: 100.0 * s.mean(),
                ),
                mean_gt_covered_pct_tol2=(
                    "gt_covered_frac_tol2",
                    lambda s: 100.0 * s.mean(),
                ),
            )
            .reset_index()
        )

    return overall, lane_stats, per_detected


def compute_ocr_metrics(df_ocr: pd.DataFrame) -> pd.DataFrame:
    """
    Resume métricas de OCR por método y preprocesamiento.

    Parámetros:
      - df_ocr: Resultados detallados del OCR.

    Returns:
      - DataFrame con columnas:
        method, preproc, total, accuracy, non_empty_pct,
        looks_plate_pct, avg_len_pred.
    """
    empty_schema = pd.DataFrame(
        columns=[
            "method",
            "preproc",
            "total",
            "accuracy",
            "non_empty_pct",
            "looks_plate_pct",
            "avg_len_pred",
        ]
    )

    print(f"[compute_ocr_metrics] filas de entrada: {len(df_ocr)}")

    if df_ocr.empty:
        print("[compute_ocr_metrics] df_ocr vacío, devuelvo esquema vacío.")
        return empty_schema

    df = df_ocr.copy()

    has_gt = df["plate_text_gt"].astype(str).str.len() > 0
    n_gt = has_gt.sum()
    print(f"[compute_ocr_metrics] filas con GT no vacío: {n_gt}")

    # Caso 1: NO hay GT de texto -> no se puede medir accuracy
    if n_gt == 0:
        grouped = (
            df.groupby(["method", "preproc"])
            .agg(
                total=("plate_text_pred", "size"),
                non_empty_pct=("non_empty", "mean"),
                looks_plate_pct=("looks_plate", "mean"),
                avg_len_pred=("length", "mean"),
            )
            .reset_index()
        )

        grouped["non_empty_pct"] *= 100.0
        grouped["looks_plate_pct"] *= 100.0
        grouped["accuracy"] = np.nan

        grouped = grouped[
            [
                "method",
                "preproc",
                "total",
                "accuracy",
                "non_empty_pct",
                "looks_plate_pct",
                "avg_len_pred",
            ]
        ]

        print(
            "[compute_ocr_metrics] no hay GT de texto, métricas sin accuracy "
            "(solo non_empty_pct / looks_plate_pct / avg_len_pred)."
        )
        return grouped.sort_values(
            ["looks_plate_pct", "non_empty_pct"], ascending=False
        )

    # Caso 2: hay GT -> se puede medir accuracy exacto
    df_gt = df[has_gt].copy()
    df_gt["correct"] = df_gt["plate_text_pred"] == df_gt["plate_text_gt"]

    grouped = (
        df_gt.groupby(["method", "preproc"])
        .agg(
            total=("plate_text_gt", "size"),
            accuracy=("correct", "mean"),
            non_empty_pct=("non_empty", "mean"),
            looks_plate_pct=("looks_plate", "mean"),
            avg_len_pred=("length", "mean"),
        )
        .reset_index()
    )

    grouped["accuracy"] *= 100.0
    grouped["non_empty_pct"] *= 100.0
    grouped["looks_plate_pct"] *= 100.0

    print(f"[compute_ocr_metrics] filas en resumen: {len(grouped)}")
    return grouped.sort_values("accuracy", ascending=False)


def best_ocr_configs(df_summary: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Devuelve las mejores combinaciones de método y preprocesamiento.
    """
    print(f"[best_ocr_configs] filas en df_summary: {len(df_summary)}")

    if df_summary.empty:
        print("[best_ocr_configs] df_summary vacío.")
        return df_summary

    df = df_summary.copy()

    if df["accuracy"].notna().any():
        df = df.sort_values("accuracy", ascending=False)
        print("[best_ocr_configs] ordenando por accuracy.")
    else:
        df = df.sort_values(
            ["looks_plate_pct", "non_empty_pct"], ascending=False
        )
        print(
            "[best_ocr_configs] sin accuracy válida, ordenando por "
            "looks_plate_pct y non_empty_pct."
        )

    return df.head(top_k)
