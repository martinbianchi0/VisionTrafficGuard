from collections import defaultdict
from pathlib import Path
import random
import shutil
import yaml
import tqdm

def split_dataset(source_root:Path, destination_root:Path, train_split=0.7, val_split=0.2, test_split=0.1):
    '''
    Divide un dataset en subconjuntos de entrenamiento, validación y prueba, garantizando que los videos completos permanezcan en un único split.

    Params:
        source_root: Ruta raíz del dataset original, que debe contener las carpetas:
            `images/train/`, `images/val/`, `labels/train/`, `labels/val/`.
        destination_root: Directorio donde se generará la nueva estructura dividida en
            `images/{train,val,test}` y `labels/{train,val,test}`.
        train_split: Proporción de videos que se asignarán al conjunto de entrenamiento.
        val_split: Proporción de videos asignados al conjunto de validación.
        test_split: Proporción de videos asignados al conjunto de prueba.

    Returns:
        None
    '''
    random.seed(42)

    print("Escaneando archivos y agrupando por video...")

    all_images = list((source_root / "images" / "train").glob("*.jpg")) + \
                    list((source_root / "images" / "val").glob("*.jpg"))

    # agrupamos las imagenes por video para evitar data leakage
    video_groups = defaultdict(list)

    for img_path in tqdm(all_images, desc="Agrupando videos"):
        parts = img_path.name.split('_')
        video_id = f"{parts[0]}_{parts[1]}" # ejemplo: 'MVI_20011'
        video_groups[video_id].append(img_path)

    unique_videos = list(video_groups.keys())
    random.shuffle(unique_videos)

    total_videos = len(unique_videos)
    print(f"\nTotal de videos encontrados: {total_videos}")
    print(f"Total de frames encontrados: {len(all_images)}")

    # calculamos los cortes
    train_end = int(total_videos * train_split)
    val_end = train_end + int(total_videos * val_split)

    train_videos = unique_videos[:train_end]
    val_videos = unique_videos[train_end:val_end]
    test_videos = unique_videos[val_end:]

    splits = {
        'train': train_videos,
        'val': val_videos,
        'test': test_videos
    }

    print(f"\nDistribución de VIDEOS:")
    print(f"Train: {len(train_videos)} | Val: {len(val_videos)} | Test: {len(test_videos)}")

    print("\nIniciando copia de archivos a DETRAC_SPLIT...")

    for split_name, videos in splits.items():
        save_img_dir = destination_root / "images" / split_name
        save_lbl_dir = destination_root / "labels" / split_name
        
        save_img_dir.mkdir(parents=True, exist_ok=True)
        save_lbl_dir.mkdir(parents=True, exist_ok=True)

        files_to_copy = []
        for vid in videos:
            files_to_copy.extend(video_groups[vid])

        print(f"Copiando {len(files_to_copy)} frames al conjunto {split_name.upper()}...")
        
        for img_path in tqdm(files_to_copy, desc=f"Copiando {split_name}"):
            shutil.copy(img_path, save_img_dir / img_path.name)
        
            origin_split_folder = img_path.parent.name
            label_name = img_path.stem + ".txt"
            
            label_src = source_root / "labels" / origin_split_folder / label_name
            
            if label_src.exists():
                shutil.copy(label_src, save_lbl_dir / label_name)
            else:
                pass

    print("\n¡Proceso terminado! Nuevo dataset en:", destination_root)

def prepare_data(DATASET_ROOT:Path, images_train_dest:Path, images_val_dest:Path, labels_train_dest:Path, labels_val_dest:Path):
    '''
    Prepara la estructura de datos para un entrenamiento YOLO copiando imágenes y etiquetas en carpetas de entrenamiento y validación.

    Params:
        DATASET_ROOT: Ruta raíz del dataset original, que debe contener carpetas `images/` y `labels/` con la misma estructura interna.
        images_train_dest: Directorio donde se copiarán las imágenes de entrenamiento.
        images_val_dest: Directorio donde se copiarán las imágenes de validación.
        labels_train_dest: Directorio donde se copiarán las etiquetas asociadas a las imágenes de entrenamiento.
        labels_val_dest: Directorio donde se copiarán las etiquetas asociadas a las imágenes de validación.

    Returns:
        None
    '''

    # creamos la estructura de carpetas vacías
    for d in (images_train_dest, images_val_dest, labels_train_dest, labels_val_dest):
        d.mkdir(parents=True, exist_ok=True)


    print("Escaneando disco duro.")

    all_images = list((DATASET_ROOT / "images").rglob("*.jpg"))
    all_labels = list((DATASET_ROOT / "labels").rglob("*.txt"))

    print(f"    -> Encontradas {len(all_images)} imágenes totales.")
    print(f"    -> Encontradas {len(all_labels)} etiquetas totales.")

    label_map = {l.stem: l for l in all_labels}

    train_count = 0
    val_count = 0

    print("Copiando archivos a la carpeta temporal...")

    for i, img_path in enumerate(all_images):
        if i > 0 and i % 5000 == 0:
            print(f"   ... procesados {i}/{len(all_images)}")

        lbl_path = label_map.get(img_path.stem)
        
        if lbl_path is None:
            continue

        parts = [p.lower() for p in img_path.parts]
        
        if 'train' in parts:
            shutil.copy(img_path, images_train_dest / img_path.name)
            shutil.copy(lbl_path, labels_train_dest / lbl_path.name)
            train_count += 1

        elif 'val' in parts or 'test' in parts:
            shutil.copy(img_path, images_val_dest / img_path.name)
            shutil.copy(lbl_path, labels_val_dest / lbl_path.name)
            val_count += 1

    print(f"\nRESUMEN FINAL:")
    print(f"   Train: {train_count} imágenes")
    print(f"   Val:   {val_count} imágenes")

def create_yaml(out_dir:Path, images_train_dest:Path, images_val_dest:Path, labels_dict:dict):
    '''
    Crea un archivo `data.yaml` para entrenar un modelo YOLO, incluyendo rutas de entrenamiento/validación y definiciones de clases.

    Params:
        out_dir: Directorio donde se guardará el archivo `data.yaml`.
        images_train_dest: Ruta al directorio que contiene las imágenes de entrenamiento.
        images_val_dest: Ruta al directorio que contiene las imágenes de validación.
        labels_dict: Diccionario que mapea `class_id` (int) a su nombre de clase (str).

    Returns
        data_yaml_path (pathlib.Path): Ruta completa al archivo `data.yaml` recién creado.
    '''

    nc = len(labels_dict)
    names = labels_dict
    
    print(f"Generando YAML:")
    print(f"    -> Clases ({nc}): {names}")

    data_yaml = {
        "path": str(out_dir.absolute()), 
        "train": str(images_train_dest.absolute()),
        "val": str(images_val_dest.absolute()),
        "nc": nc,
        "names": names
    }

    data_yaml_path = out_dir / "data.yaml"
    
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False) 

    print(f"Archivo creado exitosamente en: {data_yaml_path}")
    return data_yaml_path

