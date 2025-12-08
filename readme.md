
## Instalación del entorno (IMPORTANTE)

**No se debe subir el entorno virtual al repositorio.**  
Para reproducir el proyecto, cada usuario debe crear su propio entorno localmente.

###Crear un entorno virtual

En Linux/Mac:

```bash
python3 -m venv venv
source venv/bin/activate

python -m venv venv
venv\Scripts\activate

Una vez activado el entorno, instalar todas las dependencias del proyecto:
pip install -r requirements.txt
El archivo requirements.txt incluye todas las librerías necesarias (PyTorch, OpenCV, Ultralytics, AnyCalib, NumPy, etc.), evitando tener que distribuir un entorno virtual que puede pesar varios GB.

el dataset Debe descargarse manualmente desde el siguiente enlace:
[text](https://drive.google.com/drive/folders/1Jw73P62AXJrwQAcUtEAFyyePznWbg2D1)

