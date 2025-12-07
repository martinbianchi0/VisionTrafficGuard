from pathlib import Path

# Este archivo se espera en:
#   C:/Users/bianc/Vision/tpf/patentes/src/paths_patentes.py
# Entonces PROJECT_ROOT = .../tpf
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
VIDEO01_DIR = DATA_DIR / "video01"

VIDEO01_MP4 = VIDEO01_DIR / "video.mp4"
XML_VEHICLES_PATH = VIDEO01_DIR / "vehicles.xml"
