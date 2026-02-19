# CTCNet Face Super-Resolution — FastAPI App

Serves your trained `ctcnet_best.pth` as a REST API.

## Project Structure

```
ctcnet_fastapi/
├── main.py           # FastAPI app
├── models.py         # CTCNet architecture (same as Colab notebook)
├── requirements.txt
└── ctcnet_best.pth   # ← paste your trained model here
```

## Setup

```bash
pip install -r requirements.txt
```

Copy your `.pth` file from Colab into this folder:
```
# In Colab, download it first:
from google.colab import files
files.download('/content/ctcnet_best.pth')
```

Then place it next to `main.py`.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | List endpoints |
| GET | `/health` | Model + device status |
| GET | `/info` | Model metadata |
| POST | `/superresolve` | Upload LR image → get SR image (PNG) |
| POST | `/superresolve/base64` | Upload LR image → get base64 JSON |

### Example: curl

```bash
# Super-resolve a face image
curl -X POST "http://localhost:8000/superresolve" \
     -H "accept: image/png" \
     -F "file=@my_face_16x16.jpg" \
     --output result_128x128.png
```

### Example: Python requests

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

### Example: base64 route (easier for frontends)

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

## Interactive Docs

Visit `http://localhost:8000/docs` for the auto-generated Swagger UI —
you can upload images and test the API directly in your browser.

## Notes

- If you trained with different `base_channels` or `num_frm`, update those values
  in the `load_model()` call inside `main.py`.
- For GPU inference, just ensure CUDA is available — the app detects it automatically.
- For production, run behind nginx and use `--workers 1` (model is not thread-safe with multiple workers sharing state; use a process manager like gunicorn with 1 worker per GPU).
