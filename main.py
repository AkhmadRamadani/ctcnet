"""
CTCNet Face Super-Resolution - FastAPI Application
Auto-detects model architecture from checkpoint keys/shapes.
"""

import io
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from models import CTCNet

app = FastAPI(
    title="CTCNet Face Super-Resolution API",
    description="Upscale low-resolution face images 8× using CTCNet (CNN-Transformer Cooperation Network)",
    version="1.0.0"
)

MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ctcgan_best.pth"


def infer_model_config(state_dict: dict) -> dict:
    """
    Inspect checkpoint keys/shapes to automatically recover
    base_channels, num_frm, sr_head_channels, and num_heads.
    No hardcoding — reads directly from what was saved.
    """
    # base_channels = output channels of shallow_conv
    base_channels = state_dict["shallow_conv.weight"].shape[0]

    # num_frm = count FRM blocks in bottleneck
    num_frm = sum(
        1 for k in state_dict
        if k.startswith("bottleneck.") and k.endswith(".fsau.afdu1.reduction.weight")
    )

    # sr_head: first conv output channels
    sr_head_mid_channels = state_dict["sr_head.0.weight"].shape[0]

    # num_heads from MDTA temperature param shape [heads, 1, 1]
    num_heads = state_dict["enc_lgcm1.transformer.attn.temperature"].shape[0]

    return {
        "base_channels": base_channels,
        "num_frm": num_frm,
        "sr_head_mid_channels": sr_head_mid_channels,
        "num_heads": num_heads,
    }


def load_model():
    """Load CTCNet/CTCGAN from checkpoint, auto-detecting architecture."""
    global MODEL
    print(f"Loading CTCNet from {MODEL_PATH} on {DEVICE}...")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    cfg = infer_model_config(state_dict)
    print(f"  Auto-detected config: {cfg}")

    model = CTCNet(
        base_channels=cfg["base_channels"],
        num_frm=cfg["num_frm"],
        sr_head_mid_channels=cfg["sr_head_mid_channels"],
        num_heads=cfg["num_heads"],
        scale=8,
    )
    model.load_state_dict(state_dict)

    if "epoch" in checkpoint:
        print(f"  Loaded from epoch {checkpoint['epoch']}")

    model.to(DEVICE)
    model.eval()
    MODEL = model
    print("✅ Model ready.")


@app.on_event("startup")
async def startup_event():
    load_model()


# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess(image: Image.Image) -> torch.Tensor:
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def postprocess(tensor: torch.Tensor) -> Image.Image:
    out = tensor.squeeze(0).clamp(0, 1)
    arr = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "CTCNet Face Super-Resolution API",
        "endpoints": {
            "POST /superresolve":        "Upload LR face image → 8× SR image (PNG)",
            "POST /superresolve/base64": "Upload LR face image → base64 JSON",
            "GET  /health":              "Model + device status",
            "GET  /info":                "Model metadata",
        }
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL is not None else "model not loaded",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/info")
def info():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    total_params = sum(p.numel() for p in MODEL.parameters())
    return {
        "model": "CTCGAN Generator (CTCNet)",
        "paper": "CTCNet: A CNN-Transformer Cooperation Network for Face Image Super-Resolution",
        "scale_factor": 8,
        "parameters": f"{total_params:,}",
        "device": str(DEVICE),
    }


@app.post("/superresolve")
async def super_resolve(file: UploadFile = File(...)):
    """Upload a low-resolution face image → receive 8× super-resolved PNG."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        input_w, input_h = image.size

        with torch.no_grad():
            sr_tensor = MODEL(preprocess(image))

        sr_image = postprocess(sr_tensor)
        output_w, output_h = sr_image.size

        buf = io.BytesIO()
        sr_image.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "X-Input-Size":   f"{input_w}x{input_h}",
                "X-Output-Size":  f"{output_w}x{output_h}",
                "X-Scale-Factor": "8",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/superresolve/base64")
async def super_resolve_base64(file: UploadFile = File(...)):
    """Upload a low-resolution face image → receive base64-encoded PNG in JSON."""
    import base64

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    input_w, input_h = image.size

    with torch.no_grad():
        sr_tensor = MODEL(preprocess(image))

    sr_image = postprocess(sr_tensor)
    output_w, output_h = sr_image.size

    buf = io.BytesIO()
    sr_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "input_size":   f"{input_w}x{input_h}",
        "output_size":  f"{output_w}x{output_h}",
        "scale_factor": 8,
        "image_base64": b64,
        "format":       "PNG",
    })


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
