"""Resource boundaries without model downloads or inference: python -m unittest."""
import asyncio
import io
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
from PIL import Image

# Keep the real web stack and decoders; replace only pretrained models.
insightface = types.ModuleType("insightface")
insightface.model_zoo = types.SimpleNamespace(get_model=MagicMock())
face_app = types.ModuleType("insightface.app")
face_app.FaceAnalysis = MagicMock()
common_spec = importlib.util.spec_from_file_location(
    "insightface.app.common",
    importlib.metadata.distribution("insightface").locate_file("insightface/app/common.py"),
)
common = importlib.util.module_from_spec(common_spec)
common_spec.loader.exec_module(common)
sys.modules.update({
    "insightface": insightface, "insightface.app": face_app,
    "insightface.app.common": common,
})
import app


def image_bytes(width=8, height=8):
    output = io.BytesIO()
    Image.new("RGB", (width, height)).save(output, format="PNG")
    return output.getvalue()


class Stream(httpx.AsyncByteStream):
    def __init__(self, chunks, delay=0):
        self.chunks = chunks
        self.delay = delay
        self.closed = False
        self.read = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            await asyncio.sleep(self.delay)
            self.read += 1
            yield chunk

    async def aclose(self):
        self.closed = True


class LimitsTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, content, headers=None):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url="http://test") as client:
            return await client.post("/swap", content=content, headers=headers)

    async def download(self, stream, headers=None, status=200):
        original = httpx.AsyncClient
        transport = httpx.MockTransport(lambda request: httpx.Response(status, headers=headers, stream=stream))
        with patch.object(app.httpx, "AsyncClient", side_effect=lambda **kwargs: original(transport=transport, **kwargs)):
            return await app.fetch_image_from_url("https://example.test/image")

    async def test_download_bytes_and_headers(self):
        with patch.object(app, "MAX_DOWNLOAD_BYTES", 8):
            for headers in ({}, {"Content-Length": "bad"}, {"Content-Length": "1"}, {"Content-Length": "8"}):
                stream = Stream([b"1234", b"5678"])
                self.assertEqual(await self.download(stream, headers), b"12345678")
                self.assertTrue(stream.closed)
            for headers in ({}, {"Content-Length": "bad"}, {"Content-Length": "1"}):
                stream = Stream([b"1234", b"56789", b"never read"])
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    await self.download(stream, headers)
                self.assertEqual(stream.read, 2)
                self.assertTrue(stream.closed)
            stream = Stream([b"never read"])
            with self.assertRaisesRegex(ValueError, "exceeds"):
                await self.download(stream, {"Content-Length": "9"})
            self.assertEqual(stream.read, 0)
            self.assertTrue(stream.closed)
            stream = Stream([b"compressed"])
            with self.assertRaisesRegex(ValueError, "encoding"):
                await self.download(stream, {"Content-Encoding": "gzip"})
            self.assertEqual(stream.read, 0)
            self.assertTrue(stream.closed)

    async def test_redirect_body_is_not_buffered(self):
        streams = [Stream([b"oversized redirect body"]), Stream([b"ok"])]
        seen = []
        def handler(request):
            seen.append(request)
            if len(seen) == 1:
                return httpx.Response(302, headers={"Location": "/final"}, stream=streams[0])
            return httpx.Response(200, stream=streams[1])
        original = httpx.AsyncClient
        with patch.object(app.httpx, "AsyncClient", side_effect=lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)):
            self.assertEqual(await app.fetch_image_from_url("https://example.test/start"), b"ok")
        self.assertEqual(streams[0].read, 0)
        self.assertTrue(all(stream.closed for stream in streams))
        self.assertTrue(all(request.headers["accept-encoding"] == "identity" for request in seen))
        self.assertEqual(str(seen[1].url), "https://example.test/final")

    async def test_total_deadline_stops_trickle_and_redirects(self):
        with patch.object(app, "DOWNLOAD_TIMEOUT", 0.05):
            stream = Stream([b"a"] * 100, delay=0.01)
            with self.assertRaisesRegex(ValueError, "timed out"):
                await self.download(stream)
            self.assertLess(stream.read, 10)
            self.assertTrue(stream.closed)
            original = httpx.AsyncClient
            async def handler(request):
                await asyncio.sleep(0.02)
                return httpx.Response(302, headers={"Location": "/again"}, stream=Stream([]))
            with patch.object(app.httpx, "AsyncClient", side_effect=lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)):
                with self.assertRaisesRegex(ValueError, "timed out"):
                    await app.fetch_image_from_url("https://example.test/start")

    async def test_idle_timeout_with_real_socket(self):
        async def stalled(reader, writer):
            try:
                await reader.readuntil(b"\r\n\r\n")
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\n")
                await writer.drain()
                await reader.read()
            finally:
                writer.close()
                await writer.wait_closed()
        server = await asyncio.start_server(stalled, "127.0.0.1", 0)
        async with server:
            port = server.sockets[0].getsockname()[1]
            with patch.object(app, "DOWNLOAD_IDLE_TIMEOUT", 0.03), patch.object(app, "DOWNLOAD_TIMEOUT", 1):
                with self.assertRaisesRegex(ValueError, "timed out"):
                    await app.fetch_image_from_url(f"http://127.0.0.1:{port}/image")

    async def test_upload_read_boundary(self):
        with patch.object(app, "MAX_UPLOAD_BYTES", 8):
            for count in (8, 9):
                upload = app.UploadFile(io.BytesIO(b"x" * count))
                if count == 8:
                    self.assertEqual(await app.read_upload(upload), b"x" * 8)
                else:
                    with self.assertRaisesRegex(ValueError, "exceeds"):
                        await app.read_upload(upload)
                self.assertLessEqual(upload.file.tell(), 9)
                await upload.close()

    async def test_request_counter_and_headers(self):
        body = b"source_url=a&target_url=b"
        async def chunks(extra=b""):
            yield body[:10]
            yield body[10:]
            if extra:
                yield extra
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        with patch.object(app, "MAX_REQUEST_BYTES", len(body)), patch.object(app, "fetch_image_from_url", return_value=image_bytes()), patch.object(app.face_swap_service, "swap_faces", return_value=b"jpg") as swap, patch.object(app, "save_result_image", return_value="/result"):
            for length in (None, "bad", "9" * 5000, "1", str(len(body))):
                supplied = dict(headers)
                if length is not None:
                    supplied["Content-Length"] = length
                self.assertEqual((await self.request(chunks(), supplied)).status_code, 200)
                swap.reset_mock()
                response = await self.request(chunks(b"x"), supplied)
                self.assertEqual(response.status_code, 413)
                self.assertIn("Request exceeds", response.text)
                swap.assert_not_called()
            response = await self.request(chunks(), {**headers, "Content-Length": str(len(body) + 1)})
            self.assertEqual(response.status_code, 413)

    async def test_multipart_partial_files_closed(self):
        from starlette import formparsers
        opened = []
        original = formparsers.SpooledTemporaryFile
        def track(*args, **kwargs):
            file = original(*args, **kwargs)
            opened.append(file)
            return file
        first = b'--boundary\r\nContent-Disposition: form-data; name="source_image"; filename="a.png"\r\nContent-Type: image/png\r\n\r\nabc'
        async def chunks():
            yield first
            yield b"x" * 20
        with patch.object(app, "MAX_REQUEST_BYTES", len(first) + 10), patch.object(formparsers, "SpooledTemporaryFile", side_effect=track), patch.object(app.face_swap_service, "swap_faces") as swap:
            response = await self.request(chunks(), {"Content-Type": "multipart/form-data; boundary=boundary"})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0].closed)
        swap.assert_not_called()

    async def test_input_modes_reject_before_inference(self):
        payload = image_bytes()
        original = httpx.AsyncClient
        async with original(transport=httpx.ASGITransport(app=app.app), base_url="http://test") as client:
            with patch.object(app, "MAX_UPLOAD_BYTES", len(payload) - 1), patch.object(app.face_swap_service, "swap_faces") as swap:
                response = await client.post("/swap", files={"source_image": ("a.png", payload), "target_image": ("b.png", payload)})
                self.assertEqual(response.status_code, 400)
                self.assertIn("Uploaded image exceeds", response.text)
                swap.assert_not_called()
            stream = Stream([payload])
            with patch.object(app, "MAX_DOWNLOAD_BYTES", len(payload) - 1), patch.object(app.httpx, "AsyncClient", side_effect=lambda **kwargs: original(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream)), **kwargs)), patch.object(app.face_swap_service, "swap_faces") as swap:
                response = await client.post("/swap", data={"source_url": "https://example.test/a", "target_url": "https://example.test/b"})
                self.assertEqual(response.status_code, 400)
                self.assertIn("Downloaded image exceeds", response.text)
                swap.assert_not_called()

    def test_image_boundaries_before_decode(self):
        with patch.object(app, "MAX_IMAGE_DIMENSION", 8), patch.object(app, "MAX_IMAGE_PIXELS", 64):
            self.assertEqual(app.decode_image(image_bytes()).shape, (8, 8, 3))
            with patch.object(app.cv2, "imdecode") as decoder:
                for width, height in ((9, 8), (8, 9)):
                    with self.assertRaisesRegex(ValueError, "per side"):
                        app.decode_image(image_bytes(width, height))
                decoder.assert_not_called()
        with patch.object(app, "MAX_IMAGE_DIMENSION", 10), patch.object(app, "MAX_IMAGE_PIXELS", 63), patch.object(app.cv2, "imdecode") as decoder:
            with self.assertRaisesRegex(ValueError, "pixels"):
                app.decode_image(image_bytes())
            decoder.assert_not_called()
        with self.assertRaises(ValueError):
            app.decode_image(b"not an image")

    def test_webp_limits_before_canvas_allocation(self):
        fixtures = []
        for options in ({"lossless": False}, {"lossless": True}, {"save_all": True, "append_images": [Image.new("RGB", (8, 8), "blue")], "duration": 20}):
            output = io.BytesIO()
            Image.new("RGB", (8, 8), "red").save(output, format="WEBP", **options)
            fixtures.append(output.getvalue())
        output = io.BytesIO()
        Image.new("RGBA", (8, 8), (255, 0, 0, 128)).save(output, format="WEBP")
        fixtures.append(output.getvalue())
        self.assertEqual([data[12:16] for data in fixtures], [b"VP8 ", b"VP8L", b"VP8X", b"VP8X"])
        for data in fixtures:
            with patch.object(app, "MAX_IMAGE_DIMENSION", 8), patch.object(app, "MAX_IMAGE_PIXELS", 64):
                self.assertEqual(app.decode_image(data).shape, (8, 8, 3))
            for dimension, pixels in ((7, 64), (8, 63)):
                with patch.object(app, "MAX_IMAGE_DIMENSION", dimension), patch.object(app, "MAX_IMAGE_PIXELS", pixels), patch.object(app.Image, "open") as pillow, patch.object(app.cv2, "imdecode") as decoder:
                    with self.assertRaisesRegex(ValueError, "exceeds"):
                        app.decode_image(data)
                    pillow.assert_not_called()
                    decoder.assert_not_called()
        malformed = [fixtures[0][:24], fixtures[1][:24], fixtures[2][:29], fixtures[0][:12] + b"JUNK" + fixtures[0][16:], fixtures[0][:23] + b"bad" + fixtures[0][26:], fixtures[1][:20] + b"!" + fixtures[1][21:], fixtures[2][:16] + (9).to_bytes(4, "little") + fixtures[2][20:]]
        for data in malformed:
            with patch.object(app.Image, "open") as pillow, patch.object(app.cv2, "imdecode") as decoder:
                with self.assertRaisesRegex(ValueError, "valid WebP"):
                    app.decode_image(data)
                pillow.assert_not_called()
                decoder.assert_not_called()

    def test_header_probe_skips_eager_plugins_and_keeps_native_formats(self):
        from PIL import IcoImagePlugin
        with patch.object(IcoImagePlugin.IcoImageFile, "_open") as eager:
            with self.assertRaises(ValueError):
                app.decode_image(b"\x00\x00\x01\x00" + b"\x00" * 20)
            eager.assert_not_called()
        # OpenCV supports Radiance HDR, which Pillow does not probe.
        ok, hdr = app.cv2.imencode(".hdr", np.zeros((8, 8, 3), dtype=np.float32))
        self.assertTrue(ok)
        self.assertEqual(app.decode_image(hdr.tobytes()).shape, (8, 8, 3))

    async def test_dimensions_and_face_errors_both_input_modes(self):
        payload = image_bytes()
        original = httpx.AsyncClient
        async with original(transport=httpx.ASGITransport(app=app.app), base_url="http://test") as client:
            async def submit(mode):
                if mode == "upload":
                    return await client.post("/swap", files={"source_image": ("a.png", payload), "target_image": ("b.png", payload)})
                return await client.post("/swap", data={"source_url": "https://example.test/a", "target_url": "https://example.test/b"})
            with patch.object(app, "fetch_image_from_url", return_value=payload):
                for mode in ("upload", "url"):
                    with patch.object(app, "MAX_IMAGE_PIXELS", 63), patch.object(app.face_swap_service.face_app.det_model, "detect") as detect:
                        response = await submit(mode)
                        self.assertEqual(response.status_code, 400)
                        self.assertIn("pixels", response.text)
                        detect.assert_not_called()
                    boxes = np.zeros((app.MAX_TARGET_FACES + 1, 5))
                    with patch.object(app.face_swap_service.face_app.det_model, "detect", return_value=(boxes, None)), patch.object(app.face_swap_service, "_analyze_faces") as analyze, patch.object(app.face_swap_service.swapper, "get") as swap:
                        response = await submit(mode)
                        self.assertEqual(response.status_code, 400)
                        self.assertIn("faces", response.text)
                        analyze.assert_not_called()
                        swap.assert_not_called()

    def test_face_limit_and_first_source_detection(self):
        service = app.FaceSwapService.__new__(app.FaceSwapService)
        detector, model, swapper = MagicMock(), MagicMock(), MagicMock()
        service.face_app = types.SimpleNamespace(det_model=detector, models={"detection": detector, "recognition": model})
        service.swapper = swapper
        swapper.get.side_effect = lambda result, *args, **kwargs: result
        targets = np.array([[i, 0, i + 1, 1, 1] for i in range(2)])
        source = np.array([[1, 0, 2, 1, 1], [0, 0, 8, 8, 1]])
        landmarks = np.zeros((2, 5, 2))
        detector.detect.side_effect = [(targets, landmarks), (source, landmarks)]
        model.get.side_effect = lambda image, face: setattr(face, "embedding", np.array([3.0, 4.0]))
        with patch.object(app, "MAX_TARGET_FACES", 2):
            self.assertTrue(service.swap_faces(image_bytes(), image_bytes()))
        self.assertEqual(model.get.call_count, 3)
        self.assertEqual(swapper.get.call_count, 2)
        source_face = swapper.get.call_args.args[2]
        self.assertEqual(source_face.bbox[0], 1)
        np.testing.assert_array_equal(source_face.kps, landmarks[0])
        np.testing.assert_array_equal(source_face.normed_embedding, [0.6, 0.8])
        self.assertEqual(detector.detect.call_args_list[0].kwargs["max_num"], 3)
        self.assertEqual(detector.detect.call_args_list[1].kwargs["max_num"], 0)

    def test_configuration_and_native_decoder_in_fresh_process(self):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("FACE_SWAP_")}
        for setting, default in (("FACE_SWAP_MAX_UPLOAD_BYTES", 8), ("FACE_SWAP_DOWNLOAD_TIMEOUT_SECONDS", 1.0)):
            for invalid in ("0", "-1", "bad", "nan", "inf", "9" * 4000):
                with self.assertRaises(ValueError), patch.dict(os.environ, {setting: invalid}):
                    app.positive_setting(setting, default)
        invalid_environment = {**environment, "FACE_SWAP_MAX_TARGET_FACES": "0"}
        result = subprocess.run([sys.executable, "-c", "import app"], env=invalid_environment, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FACE_SWAP_MAX_TARGET_FACES", result.stderr)
        self.assertNotIn("insightface", result.stderr)
        code = '''import test_limits
import cv2, numpy as np
for width, height in ((9, 1), (1, 9), (8, 8)):
    ok, encoded = cv2.imencode('.png', np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    try:
        cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except cv2.error:
        pass
    else:
        raise AssertionError('native allocation guard failed')
'''
        result = subprocess.run([sys.executable, "-c", code], env={**environment, "FACE_SWAP_MAX_IMAGE_DIMENSION": "8", "FACE_SWAP_MAX_IMAGE_PIXELS": "63"}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
