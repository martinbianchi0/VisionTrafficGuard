import cv2
import numpy as np
import os   
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from anycalib import AnyCalib

def quality_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 100, 200)
    return float(edges.mean())

def select_best_frames(
    video_path: str,
    output_dir: str,
    num_frames: int = 10,
    frame_step: int = 30,
    resize_width: int | None = None
 ):

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")


    scores = []
    frames_idx = []

    idx = 0
    first_frame_saved = False
    frame0 = None

    # ---------- LOOP PRINCIPAL ----------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Guardar copia del frame 0 sin condiciones
        if idx == 0:
            frame0 = frame.copy()
            first_frame_saved = True

        # Resizing opcional
        if resize_width is not None:
            h, w = frame.shape[:2]
            scale = resize_width / w
            frame = cv2.resize(frame, (resize_width, int(h * scale)),
                               interpolation=cv2.INTER_AREA)

        # Evaluar solo frames con salto frame_step
        if idx % frame_step == 0:
            s = quality_score(frame)
            scores.append((s, idx))
            frames_idx.append((idx, frame))

        idx += 1

    cap.release()

    if not scores:
        raise RuntimeError("No se evaluó ningún frame. Ajustá frame_step o revisá el video.")

    # Ordenar por calidad
    scores.sort(key=lambda x: x[0], reverse=True)

    # Seleccionar los mejores N basados en score
    selected = {idx for (_, idx) in scores[:num_frames]}

    # --- NUEVO: incluir el frame 0 sí o sí ---
    selected.add(0)

    print(f"Seleccionando {num_frames} mejores frames")

    # Guardar frames
    saved = 0


    if first_frame_saved and frame0 is not None:
        out_path = os.path.join(output_dir, f"anycalib_frame_000000.png")
        cv2.imwrite(out_path, frame0)
        saved += 1

    # Guardar los demás frames seleccionados
    for idx_frame, frame in frames_idx:
        if idx_frame in selected and idx_frame != 0:
            out_path = os.path.join(output_dir, f"anycalib_frame_{idx_frame:06d}.png")
            cv2.imwrite(out_path, frame)
            saved += 1

    print(f"Listo! Se guardaron {saved} frames en '{output_dir}'.")





def run_anycalib_pipeline(
    img_np: np.ndarray,
    ground_z: float = 1.0,
    model_id: str = "anycalib_pinhole",
    cam_id: str = "pinhole"
 ):

    # -----------------------------
    # 1) Dimensiones de la imagen
    # -----------------------------
    H, W, _ = img_np.shape

    # -----------------------------
    # 2) Elegir device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -----------------------------
    # 3) Convertir imagen a tensor
    # -----------------------------
    image_t = (
        torch.tensor(img_np, dtype=torch.float32, device=device)
        .permute(2, 0, 1) / 255.0
    )

    # -----------------------------
    # 4) Instanciar modelo AnyCalib
    # -----------------------------
    model = AnyCalib(model_id=model_id).to(device)
    model.eval()

    # -----------------------------
    # 5) Predicción
    # -----------------------------
    with torch.no_grad():
        output = model.predict(image_t, cam_id=cam_id)

    print("Intrinsics:", output["intrinsics"])
    print("pred_size:", output["pred_size"])
    print("rays shape:", output["rays"].shape)

    intrinsics = output["intrinsics"]

    # -----------------------------
    # 6) Construir rays_dir_full
    # -----------------------------
    H_pred, W_pred = output["pred_size"]

    rays_dir = output["rays"].reshape(H_pred, W_pred, 3)

    # Preparamos para interpolación
    rays_t = (
        torch.tensor(rays_dir, dtype=torch.float32, device=device)
        .permute(2, 0, 1)       # (3, H_pred, W_pred)
        .unsqueeze(0)           # (1, 3, H_pred, W_pred)
    )

    rays_full_t = F.interpolate(
        rays_t,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )

    # Volvemos a numpy: (H, W, 3)
    rays_dir_full = (
        rays_full_t.squeeze(0)      # (3, H, W)
        .permute(1, 2, 0)           # (H, W, 3)
        .cpu()
        .numpy()
    )

    print("rays_dir_full shape:", rays_dir_full.shape)

    # -----------------------------
    # 7) Creamos pixel_to_world
    # -----------------------------
    def pixel_to_world(u: int, v: int, gz: float = ground_z):

        u = int(np.clip(u, 0, W - 1))
        v = int(np.clip(v, 0, H - 1))

        d = rays_dir_full[v, u].astype(np.float32)
        dz = d[2]
        if abs(dz) < 1e-6:
            return None

        t = gz / dz
        if t <= 0:
            return None

        return d * t

    # -----------------------------
    # 8) Creamos pixel_to_world_from_yolo
    # -----------------------------
    def pixel_to_world_from_yolo(cx_orig, cy_bottom_orig, gz: float = ground_z):
        return pixel_to_world(cx_orig, cy_bottom_orig, gz)

    # -----------------------------
    # 9) Retornar todo lo necesario
    # -----------------------------
    return {
        "intrinsics": intrinsics,
        "rays_dir_full": rays_dir_full,
        "pixel_to_world": pixel_to_world,
        "pixel_to_world_from_yolo": pixel_to_world_from_yolo,
        "H": H,
        "W": W
    }


def average_bbox_width_in_roi(
    df: pd.DataFrame,
    class_ids= 0,         
    min_conf: float=0.5,   
   
    x_left_trim_ratio: float = 0.0,   
    x_right_trim_ratio: float = 0.0,  
    y_top_trim_ratio: float = 0.0,    
    y_bottom_trim_ratio: float = 0.0, 
    img_width: float | None = None,
    img_height: float | None = None,
):



    df_filt = df.copy()


    if class_ids is not None and "class_id" in df_filt.columns:
        df_filt = df_filt[df_filt["class_id"].isin(class_ids)]

    if min_conf is not None and "conf" in df_filt.columns:
        df_filt = df_filt[df_filt["conf"] >= min_conf]

    if df_filt.empty:
        raise ValueError("No hay filas después de filtrar por clase/confianza.")


    cx_min_all = df_filt["cx"].min()
    cx_max_all = df_filt["cx"].max()
    if img_width is None:
        img_width = float(cx_max_all - cx_min_all)
        x0 = float(cx_min_all)
    else:
        img_width = float(img_width)

        x0 = 0.0

    cy_min_all = df_filt["cy_bottom"].min()
    cy_max_all = df_filt["cy_bottom"].max()
    if img_height is None:
        img_height = float(cy_max_all - cy_min_all)
        y0 = float(cy_min_all)
    else:
        img_height = float(img_height)
        y0 = 0.0

    cx_min_roi = x0 + x_left_trim_ratio * img_width
    cx_max_roi = x0 + (1.0 - x_right_trim_ratio) * img_width

    cy_min_roi = y0 + y_top_trim_ratio * img_height
    cy_max_roi = y0 + (1.0 - y_bottom_trim_ratio) * img_height


    df_roi = df_filt[
        (df_filt["cx"] >= cx_min_roi) & (df_filt["cx"] <= cx_max_roi) &
        (df_filt["cy_bottom"] >= cy_min_roi) & (df_filt["cy_bottom"] <= cy_max_roi)
    ].copy()

    if df_roi.empty or "bbox_w" not in df_roi.columns:
        raise ValueError("No hay bounding boxes dentro de la ROI definida.")

    mean_w = float(df_roi["bbox_w"].mean())
    median_w = float(df_roi["bbox_w"].median())
    n = int(len(df_roi))

    print(f"ROI X: [{cx_min_roi:.1f}, {cx_max_roi:.1f}]")
    print(f"ROI Y: [{cy_min_roi:.1f}, {cy_max_roi:.1f}]")
    print(f"Ancho promedio bbox_w (ROI): {mean_w:.2f} px")
    print(f"Ancho mediano  bbox_w (ROI): {median_w:.2f} px")
    print(f"Cant. cajas usadas (ROI)    : {n}")

    return mean_w, median_w, n


def estimate_global_scale_from_dataset(
    df: pd.DataFrame,
    pixel_to_world_fn,
    car_width_m: float = 1.9,
    car_class_ids=(0,),       
    min_conf: float = 0.5,
    min_bbox_w_px: float = 40,
    max_bbox_w_px: float | None = None,
    min_frames_per_car: int = 5,
    x_left_trim_ratio: float = 0.0,
    x_right_trim_ratio: float = 0.0,
    y_top_trim_ratio: float = 0.0,
    y_bottom_trim_ratio: float = 0.0,
    img_width: int | float | None = None,
    img_height: int | float | None = None,
    plot_roi: bool = False,
):





    cond = (
        df["class_id"].isin(car_class_ids) &
        (df["conf"] >= min_conf) &
        (df["bbox_w"] >= min_bbox_w_px)
    )
    if max_bbox_w_px is not None:
        cond &= (df["bbox_w"] <= max_bbox_w_px)

    df_cars_basic = df[cond].copy()

    if df_cars_basic.empty:
        raise ValueError(
            "No hay detecciones válidas de autos (class_id, conf, bbox_w) "
        )


    if img_width is None:
        img_width = float(df_cars_basic["cx"].max())
    else:
        img_width = float(img_width)

    if img_height is None:
        img_height = float(df_cars_basic["cy_bottom"].max())
    else:
        img_height = float(img_height)


    x_min = x_left_trim_ratio * img_width
    x_max = (1.0 - x_right_trim_ratio) * img_width

    y_min = y_top_trim_ratio * img_height
    y_max = (1.0 - y_bottom_trim_ratio) * img_height

    df_cars = df_cars_basic[
        (df_cars_basic["cx"] >= x_min) & (df_cars_basic["cx"] <= x_max) &
        (df_cars_basic["cy_bottom"] >= y_min) & (df_cars_basic["cy_bottom"] <= y_max)
    ].copy()

    if df_cars.empty:
        raise ValueError("No hay autos en la ROI definida para estimar la escala.")

    print(f"Detecciones de autos tras filtros básicos + ROI: {len(df_cars)}")
    print(f"Banda X usada: cx ∈ [{x_min:.1f}, {x_max:.1f}] sobre ancho {img_width:.1f}")
    print(f"Banda Y usada: cy_bottom ∈ [{y_min:.1f}, {y_max:.1f}] sobre alto {img_height:.1f}")

    if plot_roi:
        plt.figure(figsize=(6, 8))
        plt.scatter(
            df_cars_basic["cx"],
            df_cars_basic["cy_bottom"],
            s=5, alpha=0.2,
            label="Autos (filtro básico)"
        )
        plt.scatter(
            df_cars["cx"],
            df_cars["cy_bottom"],
            s=8, alpha=0.7,
            label="Autos en ROI"
        )

        # bandas
        plt.axvline(x_min, color="red", linestyle="--", label="Banda X")
        plt.axvline(x_max, color="red", linestyle="--")
        plt.axhline(y_min, color="green", linestyle="--", label="Banda Y")
        plt.axhline(y_max, color="green", linestyle="--")

        plt.gca().invert_yaxis()
        plt.xlabel("cx (pix)")
        plt.ylabel("cy_bottom (pix)")
        plt.title("ROI usada para estimar escala")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if "gt_id" not in df_cars.columns:
        raise ValueError("El DataFrame no tiene columna 'gt_id'.")


    scales_per_car = []

    for gt_id, df_car in df_cars.groupby("gt_id"):
        if len(df_car) < min_frames_per_car:
            continue

        widths_world = []

        for _, row in df_car.iterrows():
            cx = row.cx
            cyb = row.cy_bottom
            half_w = row.bbox_w / 2.0

            P_left = pixel_to_world_fn(cx - half_w, cyb)
            P_right = pixel_to_world_fn(cx + half_w, cyb)

            if P_left is None or P_right is None:
                continue

            P_left = np.array(P_left, dtype=np.float32)
            P_right = np.array(P_right, dtype=np.float32)

            diff_xy = P_right[:2] - P_left[:2]
            width_world = np.linalg.norm(diff_xy)

            if width_world > 0:
                widths_world.append(width_world)

        if len(widths_world) == 0:
            continue

        width_med = np.median(widths_world)
        s_car = car_width_m / width_med
        scales_per_car.append((gt_id, s_car))

    if len(scales_per_car) == 0:
        raise RuntimeError("No se pudo estimar escala para ningún auto.")

    scales_arr = np.array([s for _, s in scales_per_car])

    print("\nEjemplos de escalas por auto:")
    for g, s in scales_per_car[:10]:
        print(f"  gt_id={g}: s_car={s:.3f} m/u")

    q1, q3 = np.percentile(scales_arr, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask_inliers = (scales_arr >= lower) & (scales_arr <= upper)
    scales_inliers = scales_arr[mask_inliers]

    print(f"\nTotal autos usados: {len(scales_per_car)}")
    print(f" inliers: {len(scales_inliers)}")

    if len(scales_inliers) == 0:
        print("Todos los autos quedaron como outliers.")
        scales_inliers = scales_arr

    s_global = float(np.median(scales_inliers))
    print(f"\nEscala global estimada (mediana): s = {s_global:.4f} m/u")

    return s_global, scales_per_car



def compute_speeds_for_all_cars(
    df: pd.DataFrame,
    pixel_to_world_fn,
    s_global: float,
    fps: float,
    car_class_ids=(0,),   
    cy_quantile: float = 0.3,
    min_frames_per_car: int = 5,
):


    registros = []
    resumen = []


    df_auto = df[df["class_id"].isin(car_class_ids)].copy()

    for gt_id, df_car in df_auto.groupby("gt_id"):
        df_car = df_car.sort_values("frame_idx").copy()
        if len(df_car) < min_frames_per_car:
            continue


        cy = df_car["cy_bottom"]
        if not cy.isna().all():
            cy_thr = cy.quantile(cy_quantile)
            df_car = df_car[cy > cy_thr].copy()
        if len(df_car) < 2:
            continue

        points = []
        times = []
        frame_idxs = []


        for _, row in df_car.iterrows():
            P = pixel_to_world_fn(row.cx, row.cy_bottom)
            if P is None:
                continue

            P = np.array(P, dtype=np.float32)
            P_m = s_global * P  # metros en plano AnyCalib

            t = row.frame_idx / fps

            points.append(P_m[:2])     
            times.append(t)
            frame_idxs.append(int(row.frame_idx))

        points = np.array(points)
        times = np.array(times)

        if len(points) < 2:
            continue

        vels_kmh = []


        for i in range(1, len(points)):
            dt = times[i] - times[i-1]
            if dt <= 0:
                continue

            dist = np.linalg.norm(points[i] - points[i-1])  # metros
            v_ms = dist / dt
            v_kmh = v_ms * 3.6
            vels_kmh.append(v_kmh)

            registros.append({
                "gt_id": gt_id,
                "frame_idx": frame_idxs[i],
                "speed_kmh": v_kmh,
            })

        if len(vels_kmh) == 0:
            continue

        resumen.append({
            "gt_id": gt_id,
            "speed_kmh_mean": float(np.mean(vels_kmh)),
            "speed_kmh_median": float(np.median(vels_kmh)),
            "speed_kmh_max": float(np.max(vels_kmh)),
            "n_samples": len(vels_kmh),
        })

    df_vel = pd.DataFrame(registros)
    df_stats = pd.DataFrame(resumen)

    return df_vel, df_stats