

# CTCNet Super-Resolución Facial — Aplicación FastAPI

Sirve tu modelo entrenado `ctcnet_best.pth` como una API REST.

## Estructura del Proyecto

```
ctcnet_fastapi/
├── main.py           # FastAPI app
├── models.py         # CTCNet architecture (same as Colab notebook)
├── requirements.txt
└── ctcnet_best.pth   # ← paste your trained model here
```

## Configuración

```bash
pip install -r requirements.txt
```

Copia tu archivo `.pth` desde Colab a esta carpeta:
```
# In Colab, download it first:
from google.colab import files
files.download('/content/ctcnet_best.pth')
```

Luego colócalo junto a `main.py`.

## Ejecución

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints de la API

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Lista los endpoints |
| GET | `/health` | Estado del modelo y dispositivo |
| GET | `/info` | Metadatos del modelo |
| POST | `/superresolve` | Sube una imagen LR → obtén imagen SR (PNG) |
| POST | `/superresolve/base64` | Sube una imagen LR → obtén JSON en base64 |

### Ejemplo: curl

```bash
# Super-resolve a face image
curl -X POST "http://localhost:8000/superresolve" \
     -H "accept: image/png" \
     -F "file=@my_face_16x16.jpg" \
     --output result_128x128.png
```

### Ejemplo: requests de Python

```python
import requests

with open("my_face_16x16.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/superresolve",
        files={"file": ("face.jpg", f, "image/jpeg")}
    )

with open("result.png", "wb") as out:
    out.write(response.content)

print(response.headers["X-Output-Size"])  # e.g. "128x128"
```

### Ejemplo: ruta base64 (más fácil para frontends)

```python
import requests, base64

with open("face.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/superresolve/base64",
        files={"file": f}
    ).json()

img_bytes = base64.b64decode(resp["image_base64"])
with open("result.png", "wb") as out:
    out.write(img_bytes)
```

## Documentación Interactiva

Visita `http://localhost:8000/docs` para la interfaz de usuario de Swagger generada automáticamente: puedes subir imágenes y probar la API directamente en tu navegador.

## Notas

- Si entrenaste con diferentes valores de `base_channels` o `num_frm`, actualiza esos valores en la llamada a `load_model()` dentro de `main.py`.
- Para la inferencia con GPU, simplemente asegúrate de que CUDA esté disponible: la aplicación lo detecta automáticamente.
- Para producción, ejecuta detrás de nginx y usa `--workers 1` (el modelo no es seguro para subprocesos con múltiples workers compartiendo estado; usa un gestor de procesos como gunicorn con 1 worker por GPU).
