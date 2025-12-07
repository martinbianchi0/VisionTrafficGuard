from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from video_utils import load_frame

def build_ocr_dataset_from_detection(
    df_eval: pd.DataFrame,
    video_path: Path,
    min_iou: float = 0.3,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """
    Construye el dataset de OCR usando las patentes detectadas.
    """
    print(
        f"[build_ocr_dataset_from_detection] filas df_eval entrada: "
        f"{len(df_eval)}"
    )

    df = df_eval.copy()
    df = df[df["has_pred"]]
    df = df[df["iou"] >= min_iou]
    print(
        "[build_ocr_dataset_from_detection] filas con pred y "
        f"IoU>={min_iou}: {len(df)}"
    )

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(max_samples, random_state=0)
        print(
            "[build_ocr_dataset_from_detection] se submuestrearon filas a: "
            f"{len(df)}"
        )

    text_col = None
    for c in ["plate_text_gt", "plate", "text", "lp", "license_plate"]:
        if c in df.columns:
            text_col = c
            break
    print(
        "[build_ocr_dataset_from_detection] columna de texto GT usada: "
        f"{text_col}"
    )

    rows = []

    for idx, (_, row) in enumerate(df.iterrows()):
        frame_idx = int(row["frame"])
        frame = load_frame(video_path, frame_idx)
        if frame is None:
            print(
                "[build_ocr_dataset_from_detection] frame "
                f"{frame_idx} no se pudo cargar, se saltea."
            )
            continue

        x1 = int(row["x1_pred"])
        y1 = int(row["y1_pred"])
        x2 = int(row["x2_pred"])
        y2 = int(row["y2_pred"])

        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            print(
                "[build_ocr_dataset_from_detection] patch vacío en frame "
                f"{frame_idx}, idx local {idx}, se saltea."
            )
            continue

        gt_text = row[text_col] if text_col is not None else None

        rows.append(
            dict(
                frame=frame_idx,
                lane=int(row.get("lane", -1)),
                xml_idx=int(row.get("xml_idx", -1)),
                x1_pred=x1,
                y1_pred=y1,
                x2_pred=x2,
                y2_pred=y2,
                plate_img_bgr=patch,
                plate_text_gt=gt_text,
                iou_det=float(row["iou"]),
            )
        )

    print(
        "[build_ocr_dataset_from_detection] filas en df_ocr de salida: "
        f"{len(rows)}"
    )
    return pd.DataFrame(rows)
