from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


def build_gt_plates_df(xml_path: Path) -> pd.DataFrame:
    """
    Lee el XML de ground truth y arma un DataFrame con las bounding boxes
    de las patentes (cuando plate="True") y, si existe, info de radar.

    Columnas:
      - frame: índice de frame donde se anotó la patente (int).
      - xml_idx: índice de <vehicle> dentro del XML (int).
      - lane: carril (int).
      - x1, y1, x2, y2, w, h: bbox de la patente en píxeles.
      - radar_frame_start, radar_frame_end, radar_speed: info de radar o NaN.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    entries = []
    for idx, veh in enumerate(root.findall(".//vehicle")):
        if veh.get("plate", "False").lower() != "true":
            continue

        frame = int(veh.get("iframe"))
        lane = int(veh.get("lane"))

        region = veh.find("region")
        if region is None:
            continue

        x = int(region.get("x"))
        y = int(region.get("y"))
        w = int(region.get("w"))
        h = int(region.get("h"))

        x1, y1 = x, y
        x2, y2 = x + w, y + h

        radar = veh.find("radar")
        if radar is not None:
            radar_start = int(radar.get("frame_start"))
            radar_end = int(radar.get("frame_end"))
            radar_speed = float(radar.get("speed"))
        else:
            radar_start = np.nan
            radar_end = np.nan
            radar_speed = np.nan

        entries.append(
            {
                "frame": frame,
                "xml_idx": idx,
                "lane": lane,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "w": w,
                "h": h,
                "radar_frame_start": radar_start,
                "radar_frame_end": radar_end,
                "radar_speed": radar_speed,
            }
        )

    df = pd.DataFrame(entries)
    return df


def iou_xyxy(boxA: Tuple[int, int, int, int],
             boxB: Tuple[int, int, int, int]) -> float:
    """
    Calcula el IoU entre dos cajas en formato (x1, y1, x2, y2).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = float(areaA + areaB - inter)
    if denom <= 0:
        return 0.0

    return inter / denom
