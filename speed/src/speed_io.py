from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import pandas as pd

from .config import DATA_ROOT, SPEED_ROOT, FPS_NOMINAL
from .speed_detection import run_yolo_tracking_for_video


def load_tracks_csv(csv_path):
    """
    Carga el CSV de tracks de vehículos generado por el detector y seguidor.

    Parámetros:
      - csv_path: Ruta al archivo CSV con los tracks.

    Returns:
      - DataFrame con los tracks sin modificar columnas.
    """
    csv_path = Path(csv_path)
    return pd.read_csv(csv_path)


def prepare_tracks_df(df_tracks):
    """
    Limpia el DataFrame de tracks y agrega columnas útiles para velocidad.

    Parámetros:
      - df_tracks: DataFrame crudo con columnas por frame y bounding box.

    Returns:
      - DataFrame con vehicle_id, frame, lane y columnas px, py para el centro de la caja.
    """
    df = df_tracks.copy()

    if "vehicle_id" not in df.columns:
        raise ValueError("El CSV de tracks debe tener una columna 'vehicle_id'.")

    df = df.dropna(subset=["vehicle_id"]).copy()
    df["vehicle_id"] = df["vehicle_id"].astype("int64")

    if "lane" in df.columns:
        df = df.dropna(subset=["lane"]).copy()
        df["lane"] = df["lane"].astype("int64")
    else:
        df["lane"] = -1

    for col in ["x1", "y1", "x2", "y2"]:
        if col not in df.columns:
            raise ValueError(f"El CSV de tracks debe tener la columna '{col}'.")

    df["px"] = (df["x1"] + df["x2"]) / 2.0
    df["py"] = df["y2"].astype(float)
    df["frame"] = df["frame"].astype(int)

    return df


def get_video_fps(video_path, fallback=FPS_NOMINAL):
    """
    Obtiene el FPS real de un archivo de video usando OpenCV.

    Parámetros:
      - video_path: Ruta al archivo de video.
      - fallback: Valor por defecto si no se puede leer el FPS.

    Returns:
      - FPS del video como float.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return float(fallback)

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not pd.notna(fps) or fps <= 0:
        return float(fallback)

    return float(fps)


def load_ground_truth_from_xml(xml_path):
    """
    Lee el XML de vehículos y radar y arma un DataFrame de ground truth.

    Parámetros:
      - xml_path: Ruta a 'vehicles.xml'.

    Returns:
      - DataFrame con columnas gt_id, lane, iframe, frame_start, frame_end,
        gt_speed y la bounding box en píxeles x1_gt, y1_gt, x2_gt, y2_gt.
    """
    xml_path = Path(xml_path)
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    rows = []
    gt_id = 1

    for veh in root.findall(".//vehicle"):
        radar_flag = veh.get("radar", "False")
        radar_elem = veh.find("radar")
        region = veh.find("region")
        if radar_elem is None or region is None:
            continue

        if str(radar_flag).lower() != "true":
            continue

        lane_attr = veh.get("lane")
        iframe_attr = veh.get("iframe") or veh.get("frame")

        try:
            lane = int(lane_attr) if lane_attr is not None else -1
        except ValueError:
            lane = -1

        iframe = int(iframe_attr) if iframe_attr is not None else None

        x = int(region.get("x"))
        y = int(region.get("y"))
        w = int(region.get("w"))
        h = int(region.get("h"))
        x1_gt = x
        y1_gt = y
        x2_gt = x + w
        y2_gt = y + h

        speed_str = radar_elem.get("speed")
        fs_str = radar_elem.get("frame_start")
        fe_str = radar_elem.get("frame_end")
        if speed_str is None or fs_str is None or fe_str is None:
            continue

        try:
            gt_speed = float(speed_str)
            frame_start = int(fs_str)
            frame_end = int(fe_str)
        except ValueError:
            continue

        rows.append(
            {
                "gt_id": gt_id,
                "lane": lane,
                "iframe": iframe,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "gt_speed": gt_speed,
                "x1_gt": x1_gt,
                "y1_gt": y1_gt,
                "x2_gt": x2_gt,
                "y2_gt": y2_gt,
            }
        )
        gt_id += 1

    df_gt = pd.DataFrame(rows)
    return df_gt


def build_paths_for_video(video_name):
    """
    Devuelve rutas de tracks, XML de radar y video para un nombre de video.

    Si no existe el CSV de tracks, lo genera con YOLO y tracking.

    Parámetros:
      - video_name: Nombre del video, por ejemplo "video01".

    Returns:
      - Tuple (tracks_csv, gt_xml, video_path).
    """
    candidates = []
    candidates.append(DATA_ROOT / video_name / "vehicle_tracks_yolo_with_id.csv")
    candidates.append(SPEED_ROOT / video_name / "vehicle_tracks_yolo_with_id.csv")
    candidates.append(
        SPEED_ROOT / "boundingboxes" / "csv" / f"{video_name}_vehicle_tracks_yolo_with_id.csv"
    )

    tracks_csv = None
    for path in candidates:
        if path.exists():
            tracks_csv = path
            break

    if tracks_csv is None:
        auto_csv = SPEED_ROOT / "boundingboxes" / "csv" / f"{video_name}_vehicle_tracks_yolo_with_id.csv"
        run_yolo_tracking_for_video(video_name, auto_csv)
        tracks_csv = auto_csv

    gt_xml = DATA_ROOT / video_name / "vehicles.xml"
    video_path = DATA_ROOT / video_name / "video.mp4"

    if not gt_xml.exists():
        raise FileNotFoundError(f"No se encontró ground truth XML en {gt_xml}")
    if not video_path.exists():
        raise FileNotFoundError(f"No se encontró el video en {video_path}")

    return tracks_csv, gt_xml, video_path
