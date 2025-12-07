from fast_plate_ocr import LicensePlateRecognizer
import tempfile
import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import pytesseract
import re

from preprocessing import preprocess_plate_variant

try:
    import easyocr
except ImportError:
    easyocr = None

ALPHANUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# ----- MODELOS GLOBALES -----

if easyocr is not None:
    EASYOCR_READER = easyocr.Reader(["en"], gpu=False)
else:
    EASYOCR_READER = None

try:
    fastplate_model = LicensePlateRecognizer("cct-xs-v1-global-model")
except Exception as e:
    print("[WARN] No se pudo inicializar fast-plate-ocr:", e)
    fastplate_model = None


# ----- UTILIDADES DE TEXTO -----

def clean_plate_text(text: str | None) -> str:
    """
    Normaliza el texto de patente a A–Z y 0–9 en mayúsculas.

    Parámetros:
      - text: Texto crudo de OCR.

    Returns:
      - Texto limpio con solo caracteres alfanuméricos.
    """
    if text is None:
        return ""
    text = str(text).upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def looks_like_plate(text: str) -> bool:
    """
    Indica si el texto tiene forma LLLNNNN típica de patente.

    Parámetros:
      - text: Texto de patente ya normalizado.

    Returns:
      - True si cumple patrón, False en caso contrario.
    """
    cleaned = clean_plate_text(text)
    if len(cleaned) != 7:
        return False
    return bool(re.fullmatch(r"[A-Z]{3}[0-9]{4}", cleaned))


# ----- MOTORES OCR -----

def ocr_tesseract(img: np.ndarray, psm: int = 7) -> str:
    """
    Lee una patente con Tesseract y devuelve texto normalizado.

    Parámetros:
      - img: Imagen de patente (gray o BGR).
      - psm: Modo de segmentación de página.

    Returns:
      - Texto leído por Tesseract.
    """
    if img.ndim == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img

    config = f"--oem 1 --psm {psm} -c tessedit_char_whitelist={ALPHANUM}"
    try:
        raw = pytesseract.image_to_string(img_rgb, config=config)
    except Exception:
        raw = ""
    return clean_plate_text(raw)


def ocr_easyocr(img: np.ndarray) -> str:
    """
    Lee una patente con EasyOCR y devuelve texto normalizado.

    Parámetros:
      - img: Imagen de patente (gray o BGR).

    Returns:
      - Texto leído por EasyOCR.
    """
    if EASYOCR_READER is None:
        return ""

    if img.ndim == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    try:
        results = EASYOCR_READER.readtext(img_rgb, detail=0)
        raw = "".join(results)
    except Exception:
        raw = ""
    return clean_plate_text(raw)


def ocr_fastplate(img: np.ndarray) -> str:
    """
    Corre fast-plate-ocr sobre un recorte de patente y devuelve texto limpio.

    Parámetros:
      - img: Imagen de patente (gray o BGR).

    Returns:
      - Texto leído por fast-plate-ocr normalizado.
    """
    if fastplate_model is None or img is None:
        return ""

    if len(img.shape) == 2:
        img_save = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_save = img

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cv2.imwrite(tmp_path, img_save)
        result = fastplate_model.run(tmp_path)
        if isinstance(result, str):
            raw = result
        elif isinstance(result, dict):
            raw = result.get("plate") or result.get("text") or str(result)
        else:
            raw = str(result)
    except Exception as e:
        print("[WARN] Error fast-plate-ocr:", e)
        raw = ""
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return clean_plate_text(raw)

def run_ocr_on_dataset(
    df_ocr: pd.DataFrame,
    variants: List[str],
    methods: List[str],
) -> pd.DataFrame:
    """
    Corre múltiples OCRs y preprocesamientos sobre el dataset de patentes.
    """
    print(f"[run_ocr_on_dataset] filas df_ocr entrada: {len(df_ocr)}")
    print(f"[run_ocr_on_dataset] métodos: {methods}")
    print(f"[run_ocr_on_dataset] variantes: {variants}")

    rows: List[Dict[str, Any]] = []

    for idx, (_, row) in enumerate(df_ocr.iterrows()):
        plate_bgr = row["plate_img_bgr"]
        gt_text = row.get("plate_text_gt", "")
        gt_text_clean = clean_plate_text(gt_text)

        for v in variants:
            img_proc = preprocess_plate_variant(plate_bgr, v)

            for m in methods:
                if m == "tesseract":
                    pred = ocr_tesseract(img_proc, psm=7)
                elif m == "easyocr":
                    pred = ocr_easyocr(img_proc)
                elif m == "fastplate":
                    pred = ocr_fastplate(img_proc)
                else:
                    pred = ""

                pred_clean = clean_plate_text(pred)

                rows.append(
                    dict(
                        frame=int(row["frame"]),
                        lane=int(row.get("lane", -1)),
                        xml_idx=int(row.get("xml_idx", -1)),
                        method=m,
                        preproc=v,
                        plate_text_gt=gt_text_clean,
                        plate_text_pred=pred_clean,
                        length=len(pred_clean),
                        non_empty=(pred_clean != ""),
                        looks_plate=looks_like_plate(pred_clean),
                        iou_det=float(row.get("iou_det", 0.0)),
                    )
                )

    df_out = pd.DataFrame(rows)
    print(
        f"[run_ocr_on_dataset] filas resultados: {len(df_out)} "
        f"(esperadas: {len(df_ocr) * len(variants) * len(methods)})"
    )
    if not df_out.empty:
        print(
            "[run_ocr_on_dataset] métodos presentes:",
            sorted(df_out["method"].unique()),
        )
        print(
            "[run_ocr_on_dataset] preprocs presentes:",
            sorted(df_out["preproc"].unique()),
        )

    return df_out
