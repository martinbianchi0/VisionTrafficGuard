import numpy as np
import pandas as pd

from .speed_io import (
    build_paths_for_video,
    load_tracks_csv,
    prepare_tracks_df,
    get_video_fps,
    load_ground_truth_from_xml,
)
from .config import FPS_NOMINAL


def iou_xyxy(box_a, box_b):
    """
    Calcula el IoU entre dos cajas en formato xyxy.

    Parámetros:
      - box_a: Dict con claves x1, y1, x2, y2.
      - box_b: Dict con claves x1, y1, x2, y2.

    Returns:
      - IoU en el rango [0, 1].
    """
    x_a = max(box_a["x1"], box_b["x1"])
    y_a = max(box_a["y1"], box_b["y1"])
    x_b = min(box_a["x2"], box_b["x2"])
    y_b = min(box_a["y2"], box_b["y2"])

    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return float(inter / union)


def compute_speed_raw_for_video(
    df_tracks,
    df_gt,
    fps,
    min_frames=5,
    use_axis="y",
    iou_thresh=0.0,
):
    """
    Calcula velocidades crudas por vehículo matcheando radar con tracks.

    Parámetros:
      - df_tracks: DataFrame de tracks con px, py, vehicle_id y frame.
      - df_gt: DataFrame con gt_id, lane, frame_start, frame_end, gt_speed y bbox GT.
      - fps: FPS del video.
      - min_frames: Mínimo de frames en el intervalo del radar.
      - use_axis: Eje de posición para la regresión ("y", "x" o "norm").
      - iou_thresh: Umbral mínimo de IoU para aceptar un match.

    Returns:
      - DataFrame con una fila por vehículo de radar y velocidad cruda estimada.
    """
    df_tracks = df_tracks.copy()
    df_tracks["frame"] = df_tracks["frame"].astype(int)

    rows = []

    for _, g in df_gt.iterrows():
        gt_id = int(g["gt_id"])
        lane = int(g["lane"])
        frame_start = int(g["frame_start"])
        frame_end = int(g["frame_end"])
        gt_speed = float(g["gt_speed"])

        x1_gt = float(g["x1_gt"])
        y1_gt = float(g["y1_gt"])
        x2_gt = float(g["x2_gt"])
        y2_gt = float(g["y2_gt"])
        gt_box = {"x1": x1_gt, "y1": y1_gt, "x2": x2_gt, "y2": y2_gt}

        df_window = df_tracks[
            (df_tracks["frame"] >= frame_start)
            & (df_tracks["frame"] <= frame_end)
        ].copy()
        if df_window.empty:
            continue

        best_vid = None
        best_iou = 0.0

        for vid, group in df_window.groupby("vehicle_id"):
            best_iou_vid = 0.0
            for _, row in group.iterrows():
                box = {
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                }
                iou_val = iou_xyxy(gt_box, box)
                if iou_val > best_iou_vid:
                    best_iou_vid = iou_val
            if best_iou_vid > best_iou:
                best_iou = best_iou_vid
                best_vid = int(vid)

        if best_vid is None or best_iou <= iou_thresh:
            continue

        df_track_int = df_tracks[
            (df_tracks["vehicle_id"] == best_vid)
            & (df_tracks["frame"] >= frame_start)
            & (df_tracks["frame"] <= frame_end)
        ].copy()

        if len(df_track_int) < min_frames:
            continue

        xs = df_track_int["px"].to_numpy(dtype=float)
        ys = df_track_int["py"].to_numpy(dtype=float)

        if use_axis == "y":
            s = ys
        elif use_axis == "x":
            s = xs
        elif use_axis == "norm":
            s = np.sqrt(xs ** 2 + ys ** 2)
        else:
            raise ValueError("use_axis debe ser 'x', 'y' o 'norm'")

        frames = df_track_int["frame"].to_numpy(dtype=float)
        t = (frames - frame_start) / float(fps)

        mask = np.isfinite(s) & np.isfinite(t)
        s = s[mask]
        t = t[mask]

        if len(s) < min_frames:
            continue

        t_mean = float(t.mean())
        s_mean = float(s.mean())
        denom = float(np.sum((t - t_mean) ** 2))
        if denom <= 0:
            continue

        a = float(np.sum((t - t_mean) * (s - s_mean)) / denom)

        speed_raw = abs(a)
        delta_frames = int(frame_end - frame_start)
        n_points = int(len(s))

        rows.append(
            {
                "gt_id": gt_id,
                "vehicle_id": best_vid,
                "lane": lane,
                "speed_raw": speed_raw,
                "gt_speed": gt_speed,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "delta_frames": delta_frames,
                "n_points": n_points,
                "match_iou": float(best_iou),
            }
        )

    df_eval = pd.DataFrame(rows)
    return df_eval


def process_video(video_name, min_frames=5, use_axis="y"):
    """
    Ejecuta el pipeline completo de estimación de velocidad cruda para un video.

    Parámetros:
      - video_name: Nombre del video, por ejemplo "video01".
      - min_frames: Mínimo de frames en el intervalo del radar.
      - use_axis: Eje de posición para la regresión ("y", "x" o "norm").

    Returns:
      - DataFrame con una fila por vehículo con radar y velocidad cruda estimada.
    """
    tracks_csv, gt_xml, video_path = build_paths_for_video(video_name)

    df_tracks_raw = load_tracks_csv(tracks_csv)
    df_tracks = prepare_tracks_df(df_tracks_raw)

    fps_video = get_video_fps(video_path, fallback=FPS_NOMINAL)

    df_gt = load_ground_truth_from_xml(gt_xml)

    df_eval = compute_speed_raw_for_video(
        df_tracks=df_tracks,
        df_gt=df_gt,
        fps=fps_video,
        min_frames=min_frames,
        use_axis=use_axis,
        iou_thresh=0.0,
    )

    return df_eval
