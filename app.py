from __future__ import annotations

import os
import urllib.request
import uuid
from pathlib import Path

import cv2
import insightface
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from insightface.app import FaceAnalysis


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "inswapper_128.onnx"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
GENERATED_DIR = STATIC_DIR / "generated"


class FaceSwapService:
    def __init__(self) -> None:
        self.face_app = FaceAnalysis(name="buffalo_l", root=str(BASE_DIR))
        self.face_app.prepare(ctx_id=self._context_id(), det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model(
            str(MODEL_PATH), download=False, providers=["CPUExecutionProvider"]
        )

    @staticmethod
    def _context_id() -> int:
        return int(os.getenv("FACE_SWAP_CTX_ID", "-1"))

    def swap_faces(self, source_bytes: bytes, target_bytes: bytes) -> bytes:
        source_image = decode_image(source_bytes)
        target_image = decode_image(target_bytes)

        source_faces = self.face_app.get(source_image)
        if not source_faces:
            raise ValueError("No face detected in the source image.")

        target_faces = self.face_app.get(target_image)
        if not target_faces:
            raise ValueError("No face detected in the target image.")

        result = target_image.copy()
        source_face = source_faces[0]

        for target_face in target_faces:
            result = self.swapper.get(result, target_face, source_face, paste_back=True)

        success, encoded = cv2.imencode(".jpg", result)
        if not success:
            raise ValueError("Failed to encode the swapped image.")

        return encoded.tobytes()


def fetch_image_from_url(url: str) -> bytes:
    req = urllib.request.Request(  # noqa: S310
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; FaceSwap/1.0)",
            "Accept": "image/*,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
        return resp.read()


def decode_image(image_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a valid image.")
    return image


def save_result_image(image_bytes: bytes) -> str:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.jpg"
    output_path = GENERATED_DIR / filename
    output_path.write_bytes(image_bytes)
    return f"/static/generated/{filename}"


app = FastAPI(title="Face Swap")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
face_swap_service = FaceSwapService()


@app.get("/")
async def index(request: Request) -> object:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"request": request, "result_url": None, "error": None},
    )


@app.post("/swap")
async def swap(
    request: Request,
    source_image: UploadFile | None = File(None),
    target_image: UploadFile | None = File(None),
    source_url: str | None = Form(None),
    target_url: str | None = Form(None),
) -> object:
    try:
        if source_url:
            source_bytes = fetch_image_from_url(source_url)
        elif source_image:
            source_bytes = await source_image.read()
        else:
            raise ValueError("Provide a source image file or URL.")

        if target_url:
            target_bytes = fetch_image_from_url(target_url)
        elif target_image:
            target_bytes = await target_image.read()
        else:
            raise ValueError("Provide a target image file or URL.")
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"request": request, "result_url": None, "error": str(exc)},
            status_code=400,
        )

    try:
        swapped_bytes = face_swap_service.swap_faces(source_bytes, target_bytes)
        result_url = save_result_image(swapped_bytes)
        return templates.TemplateResponse(
            request,
            "index.html",
            {"request": request, "result_url": result_url, "error": None},
        )
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"request": request, "result_url": None, "error": str(exc)},
            status_code=400,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Face swap failed.") from exc


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/favicon.ico")
async def favicon() -> RedirectResponse:
    return RedirectResponse(url="/")
