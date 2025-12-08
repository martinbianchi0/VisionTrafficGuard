from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from .config import DATA_ROOT, YOLO_MODEL_PATH


def load_yolo_model():
    """
    Carga el modelo YOLO definido en la configuración del proyecto.

    Returns:
      - Instancia de YOLO lista para usar.
    """
    model_path = YOLO_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo YOLO en {model_path}. Ajustar YOLO_MODEL_PATH si está en otro lado."
        )
    model = YOLO(str(model_path))
    return model


def run_yolo_tracking_for_video(
    video_name,
    csv_out_path,
    conf=0.25,
    iou=0.45,
    tracker="bytetrack.yaml",
):
    """
    Corre YOLO + tracking sobre un video y guarda un CSV con los tracks.

    Parámetros:
      - video_name: Nombre del video, por ejemplo "video01".
      - csv_out_path: Ruta donde guardar el CSV generado.
      - conf: Umbral de confianza de YOLO.
      - iou: Umbral IoU de YOLO.
      - tracker: Configuración de tracker para ultralytics.

    Returns:
      - DataFrame con los tracks generados.
    """

    video_path = DATA_ROOT / video_name / "video.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"No se encontró el video {video_path}")

    model = load_yolo_model()

    csv_out_path = Path(csv_out_path)
    csv_out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    frame_idx = 0

    results = model.track(
        source=str(video_path),
        stream=True,
        tracker=tracker,
        conf=conf,
        iou=iou,
        persist=True,
    )

    for res in results:
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            frame_idx += 1
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
        clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None

        n = xyxy.shape[0]
        for i in range(n):
            x1, y1, x2, y2 = xyxy[i]
            track_id = int(ids[i]) if ids is not None else -1
            cls_id = int(clss[i]) if clss is not None else -1
            conf_val = float(confs[i]) if confs is not None else 0.0

            rows.append(
                {
                    "frame": frame_idx,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "vehicle_id": track_id,
                    "cls": cls_id,
                    "conf": conf_val,
                }
            )

        frame_idx += 1

    df_tracks = pd.DataFrame(rows)
    df_tracks.to_csv(csv_out_path, index=False)
    return df_tracks
