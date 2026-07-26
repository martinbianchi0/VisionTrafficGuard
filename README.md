# VisionTrafficGuard

Sistema de fiscalización de tránsito **basado solo en video** que, a partir de una cámara fija cenital, es capaz de:

- Detectar y trackear vehículos (YOLOv11-small + ByteTrack).
- Proyectar las trayectorias al plano real y estimar su **velocidad**.
- Leer la **patente** de cada vehículo (FastPlate-OCR + votación temporal).
- Clasificar automáticamente **infracciones** según el límite de velocidad.

El proyecto se apoya en dos datasets principales:

- **UA-DETRAC**: para entrenar y validar el detector de vehículos.
- **Vehicle-DSM**: para estimación de velocidad con radar como *ground truth* y lectura de patentes.

---

## Resultados

Cada número va con la condición en la que se midió. **La variante más precisa y la más fácil de desplegar no son la misma**, y eso es parte del resultado.

| Componente | Métrica | Medida sobre |
|---|---|---|
| **Velocidad — homografías por carril + calibración con radar** | **MAE 0.77 km/h** | 64 vehículos, ground truth de bucles inductivos. La más precisa; exige calibración física en el lugar |
| **Velocidad — AnyCalib, Modelo B** (segmento de referencia de 4,4 m) | **MAE 3.60 km/h** | 387 vehículos, conjunto de test. Sin radar ni mediciones físicas |
| Detector de vehículos (YOLOv11-small fine-tuneado) | precisión **0.952** · recall **0.979** · mAP@50-95 **0.896** | Vehicle-DSM re-etiquetado, splits por video para evitar fuga entre frames |
| Clasificador de infracciones | precisión 0.973 · recall 0.923 · **F1 0.947** · exactitud 0.956 | matriz de confusión en el informe |
| OCR de patentes (FastPlate-OCR) | 100 % de lecturas no vacías, **93 % con forma de patente válida** — contra 51.3 % de EasyOCR y 40 % de Tesseract | benchmark de 3 motores × 3 preprocesamientos, más votación temporal por track |

**Sobre el 93 %:** es la proporción de lecturas cuyo formato corresponde a una patente válida. **No es accuracy de lectura exacta** — no se midió carácter por carácter contra el ground truth.

El detalle está en `Informe.pdf` y en `Poster.pdf`.

## Estructura del repositorio

La lógica del trabajo está organizada por módulos:

- `tracking/`  
  Entrenamiento y evaluación del detector de vehículos **YOLOv11-small** y el *tracking* con **ByteTrack**.  
  Desde acá salen los pesos y *pipelines* de detección que se usan en los módulos de velocidad y patentes.

- `speed/`  
  Estimación de velocidad mediante **homografías por carril** calibradas con radar.  
  Incluye:
  - Proyección de las posiciones de los vehículos al plano métrico usando trampas de velocidad por carril.
  - Regresión distancia–tiempo para obtener la velocidad promedio dentro de la trampa.
  - Calibración por carril con radar (factor de escala \(k_\ell\)).

- `speed_anycalib_bbox/`  
  Variante de estimación de velocidad basada en **AnyCalib**, usando un *prior* de escala ligado al **ancho de la *bounding box*** del vehículo (Modelo A).  
  Sirve como baseline de calibración totalmente automática sin mediciones físicas explícitas.

- `speed_anycalib_prior/`  
  Segunda variante con AnyCalib (Modelo B): usa un **segmento real de 4.4 m** alineado con la dirección del movimiento como referencia de escala.  
  Esta versión logra un **MAE de 3.60 km/h sobre 387 vehículos** sin necesidad de radar directo (ver Resultados).

- `patentes/`  
  Módulo completo de **detección y lectura de patentes**:
  - Detector basado en **bordes verticales** + filtrado morfológico y fusión temporal por vehículo.
  - Experimentos de OCR con **Tesseract**, **EasyOCR** y **FastPlate-OCR**.
  - Comparación de distintas combinaciones de preprocesamiento (morpho, bilateral + umbral adaptativo, CLAHE, etc.).
  - Pipeline final con **FastPlate-OCR sin preprocesamiento**, más la votación temporal para consolidar una única patente por vehículo.

Cada carpeta contiene *notebooks* y/o scripts que implementan el flujo correspondiente (entrenamiento, evaluación y generación de figuras usadas en el informe).

---

## Instalación del entorno (IMPORTANTE)

No se debe subir el entorno virtual al repositorio.  
Para reproducir el proyecto, cada usuario debe crear su propio entorno localmente.

### Crear un entorno virtual

En **Linux/Mac**:

```bash
python3 -m venv venv
source venv/bin/activate
