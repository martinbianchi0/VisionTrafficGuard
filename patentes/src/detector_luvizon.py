from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np


# ==========================================================
# 1) Representación de regiones y union-find
# ==========================================================

@dataclass
class EdgeRegion:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


class DSU:
    """
    Union-Find simple para agrupar regiones compatibles.
    """

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ==========================================================
# 2) Etapa de bordes y regiones
# ==========================================================

def vertical_edges_sobel(gray: np.ndarray,
                         sobel_ksize: int = 3,
                         tau: float = 2.0) -> np.ndarray:
    """
    Implementa E(x,y) de la ecuación (3) del paper:
      - Gx: Sobel horizontal 3x3.
      - µ: valor medio de |Gx|.
      - E(x,y)=1 si |Gx| > µ * tau.

    Devuelve una imagen binaria 0/255 (uint8).
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    mag = np.abs(gx)
    mu = float(mag.mean())
    if mu <= 0:
        return np.zeros_like(gray, dtype=np.uint8)

    edges = (mag > (mu * tau)).astype(np.uint8) * 255
    return edges


def filter_edge_components(edge_img: np.ndarray,
                           min_w: int = 4,
                           max_w: int = 120,
                           min_h: int = 1,
                           max_h: int = 9999) -> np.ndarray:
    """
    Filtra componentes conectados en la imagen de bordes:
    descarta cajas con ancho/alto fuera de [min,max].

    Devuelve nueva imagen binaria con solo componentes válidos.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edge_img, connectivity=8
    )

    filtered = np.zeros_like(edge_img, dtype=np.uint8)

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if (min_w <= w <= max_w) and (min_h <= h <= max_h):
            filtered[labels == label] = 255

    return filtered


def dilate_edges(edge_img: np.ndarray,
                 kernel_size: Tuple[int, int] = (1, 7)) -> np.ndarray:
    """
    Dilata los bordes usando un elemento estructurante (1 x 7),
    como en el paper.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated = cv2.dilate(edge_img, kernel, iterations=1)
    return dilated


def build_regions_from_edges(edge_img: np.ndarray,
                             min_w: int = 4,
                             max_w: int = 9999,
                             min_h: int = 1,
                             max_h: int = 9999) -> List[EdgeRegion]:
    """
    A partir de una imagen binaria, obtiene bounding boxes de cada
    componente conectado y filtra por tamaño.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edge_img, connectivity=8
    )
    regions: List[EdgeRegion] = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if (min_w <= w <= max_w) and (min_h <= h <= max_h):
            regions.append(EdgeRegion(int(x), int(y), int(w), int(h)))

    return regions


def compatible_regions(a: EdgeRegion, b: EdgeRegion,
                       t1: float = 0.7,
                       t2: float = 1.1,
                       t3: float = 0.4) -> bool:
    """
    Criterio geométrico de compatibilidad de Retornaz & Marcotegui (ec. 4).
    """
    h1, h2 = a.h, b.h
    h = min(h1, h2)
    if h <= 0:
        return False

    dx = abs(a.cx - b.cx) - (a.w + b.w) / 2.0
    dy = abs(a.cy - b.cy)

    return (abs(h1 - h2) < t1 * h) and (dx < t2 * h) and (dy < t3 * h)


def group_regions(regions: List[EdgeRegion]) -> List[List[EdgeRegion]]:
    """
    Agrupa regiones compatibles mediante union-find.
    Devuelve lista de grupos (cada grupo es una lista de EdgeRegion).
    """
    n = len(regions)
    if n == 0:
        return []

    dsu = DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            if compatible_regions(regions[i], regions[j]):
                dsu.union(i, j)

    groups_dict: Dict[int, List[EdgeRegion]] = {}
    for idx in range(n):
        root = dsu.find(idx)
        groups_dict.setdefault(root, []).append(regions[idx])

    return list(groups_dict.values())


def groups_to_candidate_bboxes(groups: List[List[EdgeRegion]],
                               min_plate_w: int = 32,
                               min_plate_h: int = 10) -> List[Tuple[int, int, int, int]]:
    """
    Convierte grupos de EdgeRegion en bboxes de candidatos
    y filtra por tamaño mínimo de patente.
    """
    candidates: List[Tuple[int, int, int, int]] = []

    for group in groups:
        x1 = min(r.x for r in group)
        y1 = min(r.y for r in group)
        x2 = max(r.x2 for r in group)
        y2 = max(r.y2 for r in group)
        w = x2 - x1
        h = y2 - y1
        if w >= min_plate_w and h >= min_plate_h:
            candidates.append((int(x1), int(y1), int(x2), int(y2)))

    return candidates


# ==========================================================
# 3) Clasificación "texto / no texto"
# ==========================================================

def text_window_score(patch_gray: np.ndarray) -> float:
    """
    Score simple de "parecido a texto" en una ventana:
      - calcula gradientes Gx, Gy
      - usa media y desvío de la magnitud como proxy de textura/contorno.

    Cuanto mayor el score, más probable que haya caracteres.
    """
    gx = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    mean_mag = float(mag.mean())
    std_mag = float(mag.std())
    score = mean_mag * (1.0 + std_mag)

    return score


def classify_region_textlike(gray: np.ndarray,
                             edges_dilated: np.ndarray,
                             bbox: Tuple[int, int, int, int],
                             win_w: int = 48,
                             win_h: int = 24,
                             step: int = 4) -> Tuple[float, int]:
    """
    Aproximación a la etapa T-HOG/SVM del paper (Fig. 12):

      - Recibe la imagen en gris, la imagen de bordes dilatada y
        una bbox candidata (x1,y1,x2,y2).
      - Estima la "línea central" de bordes (centro entre píxel
        más alto y más bajo en cada columna con bordes).
      - Centra ventanas de tamaño fijo a lo largo de esa línea,
        espaciadas cada `step` píxeles.
      - Calcula un score de texto para cada ventana.

    Devuelve:
      - region_score: score global de "texto".
      - n_positive: cantidad de ventanas por encima del umbral.
    """
    x1, y1, x2, y2 = bbox
    H, W = gray.shape

    # Recortamos a la imagen válida
    x1_c = max(x1, 0)
    y1_c = max(y1, 0)
    x2_c = min(x2, W)
    y2_c = min(y2, H)
    if x2_c <= x1_c or y2_c <= y1_c:
        return 0.0, 0

    crop_edges = edges_dilated[y1_c:y2_c, x1_c:x2_c]
    crop_gray = gray[y1_c:y2_c, x1_c:x2_c]
    h, w = crop_gray.shape

    # Línea central de bordes por columna
    center_points: List[Tuple[int, int]] = []
    for j in range(w):
        col = crop_edges[:, j]
        ys = np.where(col > 0)[0]
        if ys.size == 0:
            continue
        cy = int((int(ys[0]) + int(ys[-1])) / 2)
        center_points.append((j, cy))

    if not center_points:
        return 0.0, 0

    # Ventanas centradas en la línea central
    half_w = win_w // 2
    half_h = win_h // 2

    scores: List[float] = []
    for idx, (cx_local, cy_local) in enumerate(center_points[::step]):
        cx = x1_c + cx_local
        cy = y1_c + cy_local

        x0_win = max(cx - half_w, 0)
        x1_win = min(cx + half_w, W - 1)
        y0_win = max(cy - half_h, 0)
        y1_win = min(cy + half_h, H - 1)

        if x1_win <= x0_win or y1_win <= y0_win:
            continue

        patch = gray[y0_win:y1_win, x0_win:x1_win]
        if patch.size == 0:
            continue

        scores.append(text_window_score(patch))

    if not scores:
        return 0.0, 0

    scores_arr = np.array(scores, dtype=np.float32)
    mu = float(scores_arr.mean())
    sigma = float(scores_arr.std())
    thr = mu + 0.3 * sigma

    positive = scores_arr[scores_arr > thr]
    if positive.size == 0:
        return 0.0, 0

    density = positive.size / scores_arr.size
    region_score = float(density * positive.mean())

    return region_score, int(positive.size)


# ==========================================================
# 4) Detección en una ROI y en un frame
# ==========================================================

def detect_plate_region_luvizon(
    roi_bgr: np.ndarray,
    min_edge_w: int = 4,
    max_edge_w: int = 120,
    min_plate_w: int = 32,
    min_plate_h: int = 10,
    tau: float = 2.0,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detector de patente estilo Luvizon para UNA ROI (imagen BGR).

    Devuelve (x1,y1,x2,y2) en coordenadas de la ROI o None si falla.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 1) Bordes verticales
    edges = vertical_edges_sobel(gray, tau=tau)

    # 2) Filtrado de componentes chicos/grandes
    edges_filt = filter_edge_components(
        edges,
        min_w=min_edge_w,
        max_w=max_edge_w,
        min_h=1,
        max_h=9999,
    )

    # 3) Dilatación 1x7
    edges_dil = dilate_edges(edges_filt, kernel_size=(1, 7))

    # 4) Regiones y agrupamiento
    regions = build_regions_from_edges(
        edges_dil,
        min_w=min_edge_w,
        max_w=max_edge_w,
        min_h=1,
        max_h=9999,
    )
    if not regions:
        return None

    groups = group_regions(regions)
    candidates = groups_to_candidate_bboxes(
        groups,
        min_plate_w=min_plate_w,
        min_plate_h=min_plate_h,
    )
    if not candidates:
        return None

    # 5) Clasificación “texto / no texto”
    H, W = gray.shape
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_score = 0.0

    for (x1, y1, x2, y2) in candidates:
        # Clampeamos por las dudas
        x1_c = max(x1, 0)
        y1_c = max(y1, 0)
        x2_c = min(x2, W)
        y2_c = min(y2, H)
        if x2_c <= x1_c or y2_c <= y1_c:
            continue

        score, n_pos = classify_region_textlike(
            gray, edges_dil, (x1_c, y1_c, x2_c, y2_c)
        )
        if n_pos == 0:
            continue

        # Penalizamos regiones muy altas: queremos algo cerca del bottom
        bottom_dist = H - y2_c
        score_adj = float(score * (1.0 + 0.001 * (H - bottom_dist)))

        if score_adj > best_score:
            best_score = score_adj
            best_bbox = (x1_c, y1_c, x2_c, y2_c)

    return best_bbox


def detect_plates_in_frame(
    frame_bgr: np.ndarray,
    rois: Optional[List[Tuple[int, int, int, int]]] = None,
    **luvizon_kwargs,
) -> List[Tuple[int, int, int, int]]:
    """
    Corre el detector Luvizon-like en un frame completo o en una lista de ROIs.

    Parámetros:
      - frame_bgr: imagen BGR.
      - rois: lista de bboxes (x1,y1,x2,y2) donde buscar la patente.
              Si es None, usa el frame entero como ROI.
      - luvizon_kwargs: parámetros extra para detect_plate_region_luvizon().

    Devuelve lista de bboxes de patentes en coords del frame.
    """
    H, W, _ = frame_bgr.shape
    results: List[Tuple[int, int, int, int]] = []

    # Caso 1: sin ROIs -> buscar en todo el frame una sola patente
    if rois is None:
        bbox = detect_plate_region_luvizon(frame_bgr, **luvizon_kwargs)
        if bbox is not None:
            results.append(bbox)
        return results

    # Caso 2: con ROIs (por ejemplo, bboxes de vehículos)
    for (x1, y1, x2, y2) in rois:
        x1_c = max(int(x1), 0)
        y1_c = max(int(y1), 0)
        x2_c = min(int(x2), W)
        y2_c = min(int(y2), H)
        if x2_c <= x1_c or y2_c <= y1_c:
            continue

        roi = frame_bgr[y1_c:y2_c, x1_c:x2_c]
        bbox_roi = detect_plate_region_luvizon(roi, **luvizon_kwargs)
        if bbox_roi is None:
            continue

        # Pasamos a coords del frame
        bx1, by1, bx2, by2 = bbox_roi
        results.append((bx1 + x1_c, by1 + y1_c, bx2 + x1_c, by2 + y1_c))

    return results
