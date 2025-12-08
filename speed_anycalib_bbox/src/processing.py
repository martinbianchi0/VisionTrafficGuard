import os
import xml.etree.ElementTree as ET
import csv
import pandas as pd
import numpy as np



def filter_vehicles_with_radar(xml_in: str, xml_out: str) -> None:

    if not os.path.exists(xml_in):
        raise FileNotFoundError(f"No se encontró el XML de entrada: {xml_in}")

    tree = ET.parse(xml_in)
    root = tree.getroot()

    gtruth = root.find("gtruth")
    if gtruth is None:
        raise ValueError("No se encontró el nodo <gtruth> en el XML.")

    vehicles = list(gtruth.findall("vehicle"))

    removed = 0
    for v in vehicles:
        radar_flag = v.get("radar")

        # Si radar es None o "false" (case-insensitive) → eliminar
        if radar_flag is None or radar_flag.lower() == "false":
            gtruth.remove(v)
            removed += 1

    # Reasignar IDs 1..N a los vehículos restantes
    for v_id, v in enumerate(gtruth.findall("vehicle"), start=1):
        v.set("vehicle_id", str(v_id))

    tree.write(xml_out)
    print(f"[filter_vehicles_with_radar] Vehículos totales en {xml_in}: {len(vehicles)}")
    print(f"[filter_vehicles_with_radar] Vehículos sin radar eliminados: {removed}")
    print(f"[filter_vehicles_with_radar] Vehículos restantes con radar: {len(gtruth.findall('vehicle'))}")
    print(f"[filter_vehicles_with_radar] XML filtrado guardado en: {xml_out}")


def convert_xml_to_csv(xml_file: str, csv_file: str) -> None:

    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"No se encuentra el archivo XML: {xml_file}")

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        raise RuntimeError(f"Error de parseo en el XML: {e}")

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Encabezados
        headers = [
            "vehicle_id",
            "lane",
            "speed",
            "frame_start",
            "frame_end",
            "initial_frame",  
            "x", "y", "w", "h",
            "moto", "plate", "radar", "sema"
        ]
        writer.writerow(headers)

        count = 0

        # Recorrer vehículos
        for vehicle in root.findall(".//vehicle"):
            try:
                v_id   = vehicle.get("vehicle_id")
                lane   = vehicle.get("lane")
                iframe = vehicle.get("iframe")
                moto   = vehicle.get("moto")
                plate  = vehicle.get("plate")
                radar_flag = vehicle.get("radar")
                sema   = vehicle.get("sema")

                # Región inicial
                region = vehicle.find("region")
                x = y = w = h = ""
                if region is not None:
                    x = region.get("x")
                    y = region.get("y")
                    w = region.get("w")
                    h = region.get("h")

                # Datos de radar (si existen)
                radar_data = vehicle.find("radar")
                speed = ""
                f_start = ""
                f_end = ""
                if radar_data is not None:
                    speed   = radar_data.get("speed")
                    f_start = radar_data.get("frame_start")
                    f_end   = radar_data.get("frame_end")

                writer.writerow([
                    v_id, lane, speed, f_start, f_end, iframe,
                    x, y, w, h,
                    moto, plate, radar_flag, sema
                ])
                count += 1

            except Exception as e:
                print(f"Error procesando vehículo (id={v_id if 'v_id' in locals() else '?'}) : {e}")
                continue

    print(f"[convert_xml_to_csv] Vehículos procesados: {count}")
    print(f"[convert_xml_to_csv] CSV guardado en: {csv_file}")




def prepare_ground_truth_with_speed(
    input_xml: str = "ground_truth.xml",
    filtered_xml: str = "vehicles_with_speed.xml",
    output_csv: str = "ground_truth_video01.csv",
    ):

    filter_vehicles_with_radar(input_xml, filtered_xml)
    convert_xml_to_csv(filtered_xml, output_csv)



def find_matches(df_yolo, df_gt, frame_tolerance=1):


    yolo_by_frame = {f: g for f, g in df_yolo.groupby('frame_idx')}


    candidates = []

    n_gt_cars = df_gt['vehicle_id'].nunique()


    for _, gt_row in df_gt.iterrows():
        gt_id = int(gt_row['vehicle_id'])


        target_frame = int(gt_row['initial_frame'])


        gt_cx = float(gt_row['x']) + float(gt_row['w']) / 2.0
        gt_cy = float(gt_row['y']) + float(gt_row['h']) / 2.0


        for f in range(target_frame - frame_tolerance,
                       target_frame + frame_tolerance + 1):
            if f not in yolo_by_frame:
                continue

            yolo_frame = yolo_by_frame[f]

            for _, y_row in yolo_frame.iterrows():
                y_id = int(y_row['track_id'])


                if not (y_row['x1'] <= gt_cx <= y_row['x2'] and
                        y_row['y1'] <= gt_cy <= y_row['y2']):
                    continue


                y_cx = (y_row['x1'] + y_row['x2']) / 2.0
                y_cy = (y_row['y1'] + y_row['y2']) / 2.0
                dist = np.hypot(gt_cx - y_cx, gt_cy - y_cy)

                candidates.append((dist, y_id, gt_id))

    if not candidates:
        print(" No se encontraron candidatos.")
        return {}

    # Ordenamos candidatos por distancia creciente
    candidates.sort(key=lambda t: t[0])


    mapping = {}
    used_yolo = set()
    used_gt = set()

    for dist, y_id, gt_id in candidates:
        if y_id in used_yolo or gt_id in used_gt:
            continue

        mapping[y_id] = gt_id
        used_yolo.add(y_id)
        used_gt.add(gt_id)

    print(f"Coincidencias encontradas: {len(mapping)} de {n_gt_cars} vehículos GT")
    return mapping


def filter_and_save(df_yolo, mapping, output_csv):



    df_clean = df_yolo[df_yolo['track_id'].isin(mapping.keys())].copy()


    df_clean['gt_id'] = df_clean['track_id'].map(mapping)

    cols = ['gt_id'] + [c for c in df_clean.columns if c != 'gt_id']
    df_clean = df_clean[cols]


    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_clean.to_csv(output_csv, index=False)




    unique_tracks = df_clean['track_id'].nunique()
    print(f"Total de vehículos únicos preservados: {unique_tracks}")

    return df_clean

def align_yolo_with_ground_truth(
    yolo_tracks_csv="data/processed/video1/video1_track.csv",
    gt_csv="data/processed/video1/gt/gt_video1_processed.csv",
    output_aligned_csv="data/processed/video1/video1_track_id.csv",
    frame_tolerance=1,
):


    if not os.path.exists(yolo_tracks_csv) or not os.path.exists(gt_csv):
        raise FileNotFoundError(
            f"Faltan archivos de entrada.\nYOLO: {yolo_tracks_csv}\nGT:   {gt_csv}"
        )


    df_yolo = pd.read_csv(yolo_tracks_csv)
    df_gt = pd.read_csv(gt_csv)

    mapping = find_matches(df_yolo, df_gt, frame_tolerance=frame_tolerance)


    df_clean = filter_and_save(df_yolo, mapping, output_aligned_csv)

    print("\nProceso de alineación completado.")
    return df_clean, mapping





def filter_track(
    df_track: pd.DataFrame,
    orig_h: int = 1080,
    margin_bottom_px: int = 20,
    margin_top_px: int = 0,
):

    if len(df_track) == 0:
        return df_track


    df_track = df_track.sort_values("frame_idx").copy()


    mask_bottom = df_track["cy_bottom"] < (orig_h - margin_bottom_px)


    if margin_top_px > 0:
        mask_top = df_track["cy_bottom"] > margin_top_px
        mask = mask_bottom & mask_top
    else:
        mask = mask_bottom

    df_track = df_track[mask].copy()

    return df_track



def filter_all_tracks(
    input_csv: str = "aligned_tracks.csv",
    output_csv: str = "aligned_tracks_stable.csv",
    orig_h: int = 1080,
    margin_bottom_px: int = 20,
    margin_top_px: int = 0,
    min_frames_per_track: int = 2,
):

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_csv}")


    df = pd.read_csv(input_csv)

    if "track_id" not in df.columns:
        raise ValueError("El CSV de entrada debe tener una columna 'track_id'.")


    all_tracks = df["track_id"].unique()
    print("Tracks encontrados:", all_tracks)

    stable_tracks = []


    for tid in all_tracks:
        df_track = df[df["track_id"] == tid]
        df_track_stable = filter_track(
            df_track,
            orig_h=orig_h,
            margin_bottom_px=margin_bottom_px,
            margin_top_px=margin_top_px,
        )

        print(f"Track {tid}: {len(df_track)} frames → {len(df_track_stable)} estables")
        stable_tracks.append(df_track_stable)


    if len(stable_tracks) > 0:
        df_all_stable = pd.concat(stable_tracks, ignore_index=True)
    else:
        df_all_stable = pd.DataFrame(columns=df.columns)


    if len(df_all_stable) > 0 and min_frames_per_track is not None and min_frames_per_track > 1:
        counts = df_all_stable.groupby("track_id").size().reset_index(name="n_frames")
        valid_ids = counts[counts["n_frames"] >= min_frames_per_track]["track_id"]

        before = len(df_all_stable)
        df_all_stable = df_all_stable[df_all_stable["track_id"].isin(valid_ids)].copy()
        after = len(df_all_stable)



    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df_all_stable.to_csv(output_csv, index=False)

    print("\nDataset filtrado guardado en:", output_csv)
    print("Total de frames estables:", len(df_all_stable))

    return df_all_stable

def load_tracks_and_gt(
    tracks_csv: str,
    gt_csv: str,
):

    if not os.path.exists(tracks_csv):
        raise FileNotFoundError(f"No se encontró TRACKS_CSV: {tracks_csv}")
    if not os.path.exists(gt_csv):
        raise FileNotFoundError(f"No se encontró GT_CSV: {gt_csv}")

    df_tracks = pd.read_csv(tracks_csv)
    df_gt = pd.read_csv(gt_csv)

    if "gt_id" not in df_tracks.columns:
        raise ValueError(
            "El CSV de tracks no tiene columna 'gt_id'. "
        )


    df_gt_speed = df_gt[["vehicle_id", "speed"]].rename(
        columns={"vehicle_id": "gt_id", "speed": "speed_kmh"}
    )

    return df_tracks, df_gt_speed


def compute_pixel_speeds(
    df_tracks: pd.DataFrame,
    fps: float,
):

    df = df_tracks.copy()

    # Orden por vehículo y frame
    df = df.sort_values(["gt_id", "frame_idx"])

    # Centro de la caja
    if "cx" not in df.columns:
        df["cx"] = (df["x1"] + df["x2"]) / 2.0

    if "cy_bottom" not in df.columns:
        if "cy" in df.columns:
            df["cy_bottom"] = df["cy"]
        else:
            df["cy_bottom"] = df["y2"]

    # Diferencias frame a frame por gt_id
    df["dx"] = df.groupby("gt_id")["cx"].diff()
    df["dy"] = df.groupby("gt_id")["cy_bottom"].diff()


    df["disp_px"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["pix_speed"] = df["disp_px"] * fps

    return df

def get_valid_gt_ids(
    df_tracks: pd.DataFrame,
    df_gt_speed: pd.DataFrame,
 ):

    ids_tracks = df_tracks["gt_id"].unique()
    ids_gt = df_gt_speed["gt_id"].unique()
    valid_ids = np.intersect1d(ids_tracks, ids_gt)

    print(f"Vehículos con tracks:          {len(ids_tracks)}")
    print(f"Vehículos con GT de velocidad: {len(ids_gt)}")
    print(f"Vehículos en intersección:     {len(valid_ids)}")

    return valid_ids



def split_train_test_ids(
    valid_ids: np.ndarray,
    test_size: float = 0.2,
    random_seed: int = 42,
):

    rng = np.random.default_rng(random_seed)
    ids = valid_ids.copy()
    rng.shuffle(ids)

    n_total = len(ids)
    n_test = int(np.round(test_size * n_total))
    n_train = n_total - n_test

    test_ids = set(ids[:n_test])
    train_ids = set(ids[n_test:])

    print("\nSplit por vehículo (IDs):")
    print(f"  Total vehículos: {n_total}")
    print(f"  Train IDs:       {len(train_ids)}")
    print(f"  Test IDs:        {len(test_ids)}")

    return train_ids, test_ids





def build_frame_level_splits(
    df_tracks: pd.DataFrame,
    train_ids: set[int],
    test_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_tracks_train = df_tracks[df_tracks["gt_id"].isin(train_ids)].reset_index(drop=True)
    df_tracks_test = df_tracks[df_tracks["gt_id"].isin(test_ids)].reset_index(drop=True)

    print("\nFilas a nivel frame:")
    print(f"  Train frames: {len(df_tracks_train)}")
    print(f"  Test frames:  {len(df_tracks_test)}")

    return df_tracks_train, df_tracks_test





def save_frame_splits(
    df_tracks_train: pd.DataFrame,
    df_tracks_test: pd.DataFrame,
    train_csv: str = "tracks_train.csv",
    test_csv: str = "tracks_test.csv",
) -> None:
    os.makedirs(os.path.dirname(train_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(test_csv) or ".", exist_ok=True)

    df_tracks_train.to_csv(train_csv, index=False)
    df_tracks_test.to_csv(test_csv, index=False)

    print("\nArchivos generados:")
    print(f"  - {train_csv}")
    print(f"  - {test_csv}")




def prepare_frame_level_train_test(
    tracks_csv: str = "aligned_tracks_stable.csv",
    gt_csv: str = "ground_truth_video01.csv",
    fps: float = 25.0,
    test_size: float = 0.2,
    random_seed: int = 42,
    train_out: str = "tracks_train.csv",
    test_out: str = "tracks_test.csv",
    min_frames_per_gt: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:


    df_tracks, df_gt_speed = load_tracks_and_gt(tracks_csv, gt_csv)


    df_tracks = compute_pixel_speeds(df_tracks, fps=fps)

    if min_frames_per_gt is not None and min_frames_per_gt > 1:
        counts = df_tracks.groupby("gt_id").size().reset_index(name="n_frames")
        valid_gt_ids = counts[counts["n_frames"] >= min_frames_per_gt]["gt_id"]
        before = len(df_tracks)
        df_tracks = df_tracks[df_tracks["gt_id"].isin(valid_gt_ids)].copy()
        after = len(df_tracks)
        print(
            f"\nFiltro por min_frames_per_gt = {min_frames_per_gt}: "
            f"frames antes={before} → después={after}"
        )

    valid_ids = get_valid_gt_ids(df_tracks, df_gt_speed)


    train_ids, test_ids = split_train_test_ids(
        valid_ids,
        test_size=test_size,
        random_seed=random_seed,
    )


    df_tracks_train, df_tracks_test = build_frame_level_splits(
        df_tracks, train_ids, test_ids
    )
    save_frame_splits(df_tracks_train, df_tracks_test, train_out, test_out)

    return df_tracks_train, df_tracks_test
