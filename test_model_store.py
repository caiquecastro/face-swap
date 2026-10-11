import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import model_store


def manifest(data):
    return len(data), hashlib.sha256(data).hexdigest()


class ModelStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.members = {"det_10g.onnx": b"detector", "w600k_r50.onnx": b"recognition"}
        self.swapper = b"swapper"
        self.archive = self.bundle()
        self.models = {f"buffalo_l/{name}": manifest(data) for name, data in self.members.items()}
        self.models["inswapper_128.onnx"] = manifest(self.swapper)
        self.patch_models = patch.object(model_store, "MODELS", self.models)
        self.patch_archives = patch.object(model_store, "ARCHIVES", {
            "buffalo_l.zip": manifest(self.archive), "inswapper_128.onnx": manifest(self.swapper)
        })
        self.patch_models.start()
        self.patch_archives.start()
        self.addCleanup(self.patch_models.stop)
        self.addCleanup(self.patch_archives.stop)

    def bundle(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as bundle:
            for name, data in self.members.items():
                bundle.writestr(name, data)
        return output.getvalue()

    def downloader(self, name, target):
        target.write_bytes(self.archive if name == "buffalo_l.zip" else self.swapper)
        self.assertTrue(model_store.valid(target, model_store.ARCHIVES[name]))

    def test_setup_verify_idempotence_and_directory_setting(self):
        with patch.object(model_store, "download", side_effect=self.downloader) as download, patch.dict("os.environ", {"FACE_SWAP_MODEL_DIR": str(self.root)}):
            with self.assertRaisesRegex(RuntimeError, "Missing, incomplete, or corrupt"):
                model_store.verify()
            self.assertEqual(model_store.setup(), self.root)
            self.assertEqual(model_store.verify(), self.root)
            model_store.setup()
            self.assertEqual(download.call_count, 2)
        self.assertEqual((self.root / "buffalo_l/det_10g.onnx").read_bytes(), b"detector")

    def test_corrupt_and_missing_files_are_detected_and_good_files_preserved(self):
        with patch.object(model_store, "download", side_effect=self.downloader):
            model_store.setup(self.root)
            good = self.root / "buffalo_l/w600k_r50.onnx"
            old_stat = good.stat()
            (self.root / "buffalo_l/det_10g.onnx").write_bytes(b"bad")
            (self.root / "inswapper_128.onnx").unlink()
            with self.assertRaisesRegex(RuntimeError, "det_10g.onnx.*inswapper_128.onnx"):
                model_store.verify(self.root)
            model_store.setup(self.root)
            self.assertEqual(good.stat().st_ino, old_stat.st_ino)
            self.assertEqual(model_store.verify(self.root), self.root)

    def test_bad_archive_member_or_interrupted_download_publishes_nothing(self):
        good = self.root / "buffalo_l/w600k_r50.onnx"
        good.parent.mkdir()
        good.write_bytes(b"recognition")
        for archive in (b"broken", self.archive.replace(b"detector", b"tampered")):
            with self.subTest(archive=archive[:8]), patch.object(model_store, "download", side_effect=lambda name, target: target.write_bytes(archive)):
                with self.assertRaises((ValueError, zipfile.BadZipFile)):
                    model_store.setup(self.root)
                self.assertEqual(good.read_bytes(), b"recognition")
                self.assertFalse((self.root / "buffalo_l/det_10g.onnx").exists())
        def interrupted(name, target):
            target.write_bytes(b"partial")
            raise OSError("connection lost")
        with patch.object(model_store, "download", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "connection lost"):
                model_store.setup(self.root)
        self.assertEqual(good.read_bytes(), b"recognition")

    def test_download_rejects_archive_checksum_before_extraction(self):
        class Response(io.BytesIO):
            def geturl(self):
                return model_store.SOURCE + "buffalo_l.zip"

        with patch.object(model_store.urllib.request, "urlopen", return_value=Response(b"wrong archive")):
            with self.assertRaisesRegex(ValueError, "checksum or size mismatch"):
                model_store.setup(self.root)
        self.assertFalse((self.root / "buffalo_l").exists())

    def test_same_size_wrong_swapper_download_preserves_valid_analysis(self):
        analysis = self.root / "buffalo_l"
        analysis.mkdir()
        for name, data in self.members.items():
            (analysis / name).write_bytes(data)
        good = analysis / "det_10g.onnx"
        old_inode = good.stat().st_ino

        class Response(io.BytesIO):
            def geturl(self):
                return model_store.SOURCE + "inswapper_128.onnx"

        wrong = b"SWAPPER"
        self.assertEqual(len(wrong), len(self.swapper))
        with patch.object(model_store.urllib.request, "urlopen", return_value=Response(wrong)) as request:
            with self.assertRaisesRegex(ValueError, "checksum or size mismatch"):
                model_store.setup(self.root)
            request.assert_called_once()
        self.assertEqual(good.read_bytes(), self.members["det_10g.onnx"])
        self.assertEqual(good.stat().st_ino, old_inode)
        self.assertFalse((self.root / "inswapper_128.onnx").exists())

    def test_same_size_corrupt_installed_file_fails_runtime_verify(self):
        with patch.object(model_store, "download", side_effect=self.downloader):
            model_store.setup(self.root)
        good = self.root / "buffalo_l/det_10g.onnx"
        old_inode = good.stat().st_ino
        swapper = self.root / "inswapper_128.onnx"
        wrong = b"SWAPPER"
        self.assertEqual(len(wrong), swapper.stat().st_size)
        swapper.write_bytes(wrong)
        with self.assertRaisesRegex(RuntimeError, "inswapper_128.onnx"):
            model_store.verify(self.root)
        self.assertEqual(swapper.read_bytes(), wrong)
        self.assertEqual(good.read_bytes(), self.members["det_10g.onnx"])
        self.assertEqual(good.stat().st_ino, old_inode)

    def test_archive_contents_and_member_checksum_are_checked_before_publish(self):
        for members in ({"../escape.onnx": b"x"}, {"det_10g.onnx": b"wrong", "w600k_r50.onnx": b"recognition"}):
            self.members = members
            archive = self.bundle()
            with self.subTest(members=members), patch.dict(model_store.ARCHIVES, {"buffalo_l.zip": manifest(archive)}), patch.object(model_store, "download", side_effect=lambda name, target: target.write_bytes(archive if name == "buffalo_l.zip" else self.swapper)):
                with self.assertRaisesRegex(ValueError, "Unexpected files|checksum"):
                    model_store.setup(self.root)
                self.assertFalse((self.root / "buffalo_l/det_10g.onnx").exists())
                self.assertFalse((self.root / "inswapper_128.onnx").exists())
                self.assertFalse((self.root.parent / "escape.onnx").exists())

    def test_runtime_is_read_only_and_preserves_unrelated_loading_error(self):
        from test_limits import app

        with patch.object(model_store, "download", side_effect=self.downloader):
            model_store.setup(self.root)
        before = {name: (self.root / name).read_bytes() for name in self.models}
        with patch.object(app, "MODELS", self.models), patch.object(app, "model_dir", return_value=self.root), patch.object(app, "verify", wraps=model_store.verify), patch.object(model_store.urllib.request, "urlopen", side_effect=AssertionError("network called")):
            service = app.FaceSwapService()
            service.det_model.prepare.assert_called_once_with(-1, input_size=(640, 640), det_thresh=0.5)
            service.swapper.prepare.assert_not_called()
            with patch.object(app.insightface.model_zoo, "get_model", side_effect=RuntimeError("unrelated ONNX failure")) as load:
                with self.assertRaisesRegex(RuntimeError, "unrelated ONNX failure") as failure:
                    app.FaceSwapService()
                self.assertIsInstance(failure.exception.__cause__, RuntimeError)
                self.assertEqual(str(failure.exception.__cause__), "unrelated ONNX failure")
                self.assertIn("model_store.py setup", str(failure.exception))
                load.assert_called_once()
            (self.root / "buffalo_l/det_10g.onnx").write_bytes(b"corrupt")
            with patch.object(app.insightface.model_zoo, "get_model") as load:
                with self.assertRaisesRegex(RuntimeError, "det_10g.onnx"):
                    app.FaceSwapService()
                load.assert_not_called()
        for name, content in before.items():
            if name != "buffalo_l/det_10g.onnx":
                self.assertEqual((self.root / name).read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
