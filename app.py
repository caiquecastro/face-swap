from __future__ import annotations

import asyncio
import io
import math
import os
import shutil
import uuid
from pathlib import Path


def positive_setting(name: str, default: int | float) -> int | float:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value) if isinstance(default, int) else float(value)
        valid = math.isfinite(parsed) and parsed > 0
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite number.") from exc
    if not valid:
        raise ValueError(f"{name} must be a positive finite number.")
    return parsed


MAX_REQUEST_BYTES = positive_setting("FACE_SWAP_MAX_REQUEST_BYTES", 21 * 1024 * 1024)
MAX_UPLOAD_BYTES = positive_setting("FACE_SWAP_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
MAX_DOWNLOAD_BYTES = positive_setting("FACE_SWAP_MAX_DOWNLOAD_BYTES", 10 * 1024 * 1024)
MAX_IMAGE_DIMENSION = positive_setting("FACE_SWAP_MAX_IMAGE_DIMENSION", 4096)
MAX_IMAGE_PIXELS = positive_setting("FACE_SWAP_MAX_IMAGE_PIXELS", 16_000_000)
MAX_TARGET_FACES = positive_setting("FACE_SWAP_MAX_TARGET_FACES", 10)
DOWNLOAD_TIMEOUT = positive_setting("FACE_SWAP_DOWNLOAD_TIMEOUT_SECONDS", 15.0)
DOWNLOAD_IDLE_TIMEOUT = positive_setting("FACE_SWAP_DOWNLOAD_IDLE_TIMEOUT_SECONDS", 5.0)

# OpenCV reads these once, when imported; cap its allocation as well as headers.
os.environ["OPENCV_IO_MAX_IMAGE_WIDTH"] = str(MAX_IMAGE_DIMENSION)
os.environ["OPENCV_IO_MAX_IMAGE_HEIGHT"] = str(MAX_IMAGE_DIMENSION)
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(MAX_IMAGE_PIXELS)

import cv2
import httpx
import insightface
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from PIL import Image, UnidentifiedImageError
from starlette.formparsers import MultiPartException

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "inswapper_128.onnx"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
GENERATED_DIR = STATIC_DIR / "generated"


class FaceSwapService:
    def __init__(self) -> None:
        self.face_app = self._load_face_app()
        self.face_app.prepare(ctx_id=self._context_id(), det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model(
            str(MODEL_PATH), download=False, providers=["CPUExecutionProvider"]
        )

    def _load_face_app(self) -> FaceAnalysis:
        buffalo_dir = BASE_DIR / "models" / "buffalo_l"
        for attempt in range(2):
            try:
                return FaceAnalysis(name="buffalo_l", root=str(BASE_DIR))
            except Exception:
                if attempt == 0 and buffalo_dir.exists():
                    shutil.rmtree(buffalo_dir)
        raise RuntimeError("Failed to load buffalo_l model after cleanup retry.")

    @staticmethod
    def _context_id() -> int:
        return int(os.getenv("FACE_SWAP_CTX_ID", "-1"))

    def swap_faces(self, source_bytes: bytes, target_bytes: bytes) -> bytes:
        source_image = decode_image(source_bytes)
        target_image = decode_image(target_bytes)

        # Count detections before running any per-face models or swaps.
        target_boxes, target_landmarks = self.face_app.det_model.detect(
            target_image, max_num=MAX_TARGET_FACES + 1
        )
        if len(target_boxes) > MAX_TARGET_FACES:
            raise ValueError(f"Target image exceeds the limit of {MAX_TARGET_FACES} faces.")
        if len(target_boxes) == 0:
            raise ValueError("No face detected in the target image.")

        # max_num=1 ranks by size/position; detect all to preserve the first face.
        source_boxes, source_landmarks = self.face_app.det_model.detect(source_image, max_num=0)
        if len(source_boxes) == 0:
            raise ValueError("No face detected in the source image.")
        source_faces = self._analyze_faces(source_image, source_boxes[:1], source_landmarks)
        target_faces = self._analyze_faces(target_image, target_boxes, target_landmarks)

        result = target_image.copy()
        source_face = source_faces[0]

        for target_face in target_faces:
            result = self.swapper.get(result, target_face, source_face, paste_back=True)

        success, encoded = cv2.imencode(".jpg", result)
        if not success:
            raise ValueError("Failed to encode the swapped image.")

        return encoded.tobytes()

    def _analyze_faces(self, image, boxes, landmarks):
        faces = []
        for index, box in enumerate(boxes):
            face = Face(
                bbox=box[:4], det_score=box[4],
                kps=landmarks[index] if landmarks is not None else None,
            )
            for task, model in self.face_app.models.items():
                if task != "detection":
                    model.get(image, face)
            faces.append(face)
        return faces


def declared_length(headers) -> int | None:
    value = headers.get("content-length", "")
    try:
        return int(value) if value.isascii() and value.isdecimal() else None
    except ValueError:
        return None


async def fetch_image_from_url(url: str) -> bytes:
    try:
        async with asyncio.timeout(DOWNLOAD_TIMEOUT):
            async with httpx.AsyncClient(timeout=DOWNLOAD_IDLE_TIMEOUT) as client:
                # Follow manually: HTTPX otherwise buffers redirect bodies.
                for redirect in range(21):
                    async with client.stream("GET", url, headers={
                        "User-Agent": "Mozilla/5.0 (compatible; FaceSwap/1.0)",
                        "Accept": "image/*,*/*", "Accept-Encoding": "identity",
                    }) as response:
                        if response.is_redirect and "location" in response.headers:
                            if redirect == 20:
                                raise ValueError("Image URL redirected too many times.")
                            url = str(response.url.join(response.headers["location"]))
                            continue
                        response.raise_for_status()
                        if response.headers.get("content-encoding", "identity").lower() != "identity":
                            raise ValueError("Image URL returned an unsupported content encoding.")
                        size = declared_length(response.headers)
                        if size is not None and size > MAX_DOWNLOAD_BYTES:
                            raise ValueError(f"Downloaded image exceeds the limit of {MAX_DOWNLOAD_BYTES} bytes.")
                        data = bytearray()
                        async for chunk in response.aiter_raw():
                            if len(data) + len(chunk) > MAX_DOWNLOAD_BYTES:
                                raise ValueError(f"Downloaded image exceeds the limit of {MAX_DOWNLOAD_BYTES} bytes.")
                            data.extend(chunk)
                        return bytes(data)
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise ValueError("Image URL download timed out.") from exc
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        raise ValueError("Could not download image from URL.") from exc


async def read_upload(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Uploaded image exceeds the limit of {MAX_UPLOAD_BYTES} bytes.")
    return data


def check_dimensions(width: int, height: int) -> None:
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError(f"Image exceeds the limit of {MAX_IMAGE_DIMENSION} pixels per side.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image exceeds the limit of {MAX_IMAGE_PIXELS} pixels.")


def check_webp_header(data: bytes) -> None:
    # Both Pillow and OpenCV allocate animated WebP canvases while opening.
    # Read its first RIFF chunk ourselves before entering either decoder.
    chunk = data[12:16]
    size = int.from_bytes(data[16:20], "little")
    header = data[20:30]
    if len(data) < 20 or size > len(data) - 20:
        raise ValueError("Uploaded file is not a valid WebP image.")
    if chunk == b"VP8X" and size == 10 and len(header) == 10:
        width = 1 + int.from_bytes(header[4:7], "little")
        height = 1 + int.from_bytes(header[7:10], "little")
    elif chunk == b"VP8 " and size >= 10 and len(header) == 10 and header[3:6] == b"\x9d\x01\x2a":
        width = int.from_bytes(header[6:8], "little") & 0x3FFF
        height = int.from_bytes(header[8:10], "little") & 0x3FFF
    elif chunk == b"VP8L" and size >= 5 and len(header) >= 5 and header[0] == 0x2F:
        bits = int.from_bytes(header[1:5], "little")
        width = 1 + (bits & 0x3FFF)
        height = 1 + ((bits >> 14) & 0x3FFF)
    else:
        raise ValueError("Uploaded file is not a valid WebP image.")
    if width == 0 or height == 0:
        raise ValueError("Uploaded file is not a valid WebP image.")
    check_dimensions(width, height)


def decode_image(image_bytes: bytes) -> np.ndarray:
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        check_webp_header(image_bytes)
    try:
        # Probe only header-only plugins; others (e.g. ICO) eagerly load pixels.
        with Image.open(io.BytesIO(image_bytes), formats=[
            "JPEG", "PNG", "BMP", "DIB", "TIFF", "GIF", "JPEG2000", "PPM", "SUN", "WEBP",
        ]) as header:
            check_dimensions(*header.size)
    except UnidentifiedImageError:
        pass  # Other OpenCV formats still have its native allocation guards.
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Uploaded file is invalid or exceeds image limits.") from exc
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    try:
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("Uploaded file is invalid or exceeds image limits.") from exc
    if image is None:
        raise ValueError("Uploaded file is not a valid image.")
    check_dimensions(image.shape[1], image.shape[0])
    return image


class RequestSizeLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        size = declared_length(request.headers)
        exceeded = size is not None and size > MAX_REQUEST_BYTES
        total = 0

        async def bounded_receive():
            nonlocal total, exceeded
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > MAX_REQUEST_BYTES:
                    exceeded = True
                    # Starlette closes partial multipart files for this exception.
                    raise MultiPartException("Request body is too large.")
            return message

        async def bounded_send(message):
            if not exceeded:
                await send(message)

        if not exceeded:
            try:
                await self.app(scope, bounded_receive, bounded_send)
            except MultiPartException:
                if not exceeded:
                    raise
        if exceeded:
            response = templates.TemplateResponse(
                request, "index.html",
                {"request": request, "result_url": None,
                 "error": f"Request exceeds the limit of {MAX_REQUEST_BYTES} bytes."},
                status_code=413,
            )
            await response(scope, receive, send)


def save_result_image(image_bytes: bytes) -> str:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.jpg"
    output_path = GENERATED_DIR / filename
    output_path.write_bytes(image_bytes)
    return f"/static/generated/{filename}"


app = FastAPI(title="Face Swap")
app.add_middleware(RequestSizeLimit)
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
            source_bytes = await fetch_image_from_url(source_url)
        elif source_image:
            source_bytes = await read_upload(source_image)
        else:
            raise ValueError("Provide a source image file or URL.")

        if target_url:
            target_bytes = await fetch_image_from_url(target_url)
        elif target_image:
            target_bytes = await read_upload(target_image)
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
