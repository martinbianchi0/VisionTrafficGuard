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
  Esta versión logra errores de velocidad más estables (MAE ≈ 3–4 km/h) sin necesidad de radar directo.

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
