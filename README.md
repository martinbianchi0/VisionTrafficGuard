# VisionTrafficGuard

VisionTrafficGuard es un sistema integral de fiscalización de tránsito basado únicamente en video, capaz de estimar velocidades vehiculares, leer patentes y clasificar infracciones a partir de una cámara fija.  

El pipeline combina:

- Detección y seguimiento de vehículos con **YOLOv11-small** + **ByteTrack**.
- Estimación de velocidad en el plano real usando **homografías por carril** calibradas con radar y variantes de auto-calibración con **AnyCalib**.
- Detección y lectura de patentes con un detector basado en bordes verticales y **FastPlate-OCR** con votación temporal por vehículo.
- Clasificación automática de infracciones de velocidad a partir de las velocidades estimadas.

El objetivo es acercarse a la precisión de un radar usando sólo cámaras ya instaladas en la infraestructura urbana.

---

## Estructura del repositorio

- `speed/`  
  Módulo de **estimación de velocidad**:
  - Entrenamiento y *fine-tuning* de YOLOv11 con UA-DETRAC y Vehicle-DSM.
  - Proyección al plano métrico (homografías por carril, modelos AnyCalib).
  - Regresión distancia–tiempo y calibración con radar por carril.
  - Cálculo de métricas y generación de figuras.

- `patentes/`  
  Módulo de **detección y lectura de patentes**:
  - Detector basado en bordes verticales y fusión temporal por vehículo.
  - Experimentos con Tesseract, EasyOCR y FastPlate-OCR.
  - Votación temporal y métricas sobre los *crops* de patentes.

- `tracking/`  
  Módulo de **detección + tracking**:
  - Aplicación de YOLOv11-small sobre los videos del escenario.
  - Seguimiento temporal con ByteTrack para generar *tracks* por vehículo.

Cada carpeta contiene el código y los *notebooks* necesarios para reproducir los experimentos descritos en el informe.

---

## Instalación del entorno (IMPORTANTE)

No se debe subir el entorno virtual al repositorio.  
Para reproducir el proyecto, cada usuario debe crear su propio entorno localmente.

### 1. Crear un entorno virtual

**En Linux / Mac:**

```bash
python3 -m venv venv
source venv/bin/activate
