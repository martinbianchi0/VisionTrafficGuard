from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def load_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    """
    Carga un frame del video dado un índice.
    Devuelve imagen BGR o None si falla.
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_idx < 0 or frame_idx >= total_frames:
        print(f"[load_frame] Frame {frame_idx} fuera de rango (0..{total_frames-1})")
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[load_frame] No se pudo leer el frame {frame_idx}")
        return None

    return frame
