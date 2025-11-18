from pathlib import Path
from typing import List, Tuple, Dict

from ultralytics import YOLO


DEFAULT_CLASS_NAMES: Dict[int, str] = {
    0: "others",
    1: "car",
    2: "van",
    3: "bus",
}


def create_yolo_subset_config(
    image_paths: List[Path],
    yaml_path: str | Path,
    class_names: Dict[int, str] = DEFAULT_CLASS_NAMES,
    train_ratio: float = 0.8,
) -> Tuple[Path, Path, Path]:
    """
    Crea un data.yaml y archivos train.txt / val.txt para entrenar YOLO
    usando sólo las imágenes de 'image_paths'.

    Parámetros:
      - image_paths: Lista de Paths de imágenes a usar (por ej. subset['image_paths']).
      - yaml_path: Ruta donde guardar el .yaml (ej: 'C:/.../ua_detrac_subset.yaml').
      - class_names: Dict {id: nombre_clase}.
      - train_ratio: Porcentaje de imágenes para train (el resto va a val).

    Returns:
      - (yaml_path, train_txt_path, val_txt_path)
    """
    yaml_path = Path(yaml_path)
    yaml_dir = yaml_path.parent
    yaml_dir.mkdir(parents=True, exist_ok=True)

    if not image_paths:
        raise ValueError("La lista de image_paths está vacía.")

    # Ordenamos para que el split sea reproducible
    image_paths = sorted(image_paths, key=lambda p: p.as_posix())

    n_total = len(image_paths)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, n_total - n_train)

    train_imgs = image_paths[:n_train]
    val_imgs = image_paths[n_train:] if n_val > 0 else image_paths[-1:]

    # Archivos .txt para YOLO (lista de paths de imágenes)
    train_txt = yaml_dir / (yaml_path.stem + "_train.txt")
    val_txt = yaml_dir / (yaml_path.stem + "_val.txt")

    def _write_list(txt_path: Path, imgs: List[Path]) -> None:
        with open(txt_path, "w", encoding="utf-8") as f:
            for img in imgs:
                # Usamos paths absolutos en formato POSIX para evitar problemas de barras
                f.write(img.as_posix() + "\n")

    _write_list(train_txt, train_imgs)
    _write_list(val_txt, val_imgs)

    # Armamos el YAML a mano (sin depender de PyYAML)
    # YOLO acepta train/val como paths a .txt con rutas de imágenes
    yaml_lines = [
        "path: .",  # no es tan relevante si train/val son absolutos
        f"train: {train_txt.as_posix()}",
        f"val: {val_txt.as_posix()}",
        "names:",
    ]
    for cid, cname in class_names.items():
        yaml_lines.append(f"  {cid}: {cname}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")

    print(f"Config YOLO creada:")
    print(f"  YAML:   {yaml_path}")
    print(f"  train:  {train_txt} ({len(train_imgs)} imágenes)")
    print(f"  val:    {val_txt} ({len(val_imgs)} imágenes)")

    return yaml_path, train_txt, val_txt


def finetune_yolo_model(
    model: YOLO,
    data_yaml_path: str | Path,
    epochs: int = 3,
    imgsz: int = 640,
    batch: int = 4,
    project: str = "runs",
    name: str = "ua_detrac_subset",
):
    """
    Hace fine-tuning de un modelo YOLO sobre el dataset definido en un data.yaml.

    Parámetros:
      - model: Instancia de YOLO ya cargada (por ej. yolo_model).
      - data_yaml_path: Ruta al data.yaml que define train/val.
      - epochs: Cantidad de épocas de entrenamiento.
      - imgsz: Tamaño de imagen (lado más largo) para el training.
      - batch: Tamaño de batch.
      - project: Carpeta raíz donde YOLO guarda 'runs/'.
      - name: Nombre del experimento (subcarpeta dentro de project).

    Returns:
      - Objeto 'results' devuelto por model.train().
    """
    data_yaml_path = Path(data_yaml_path)

    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=0,  # en Windows conviene 0 para evitar quilombos
        project=project,
        name=name,
    )
    return results
