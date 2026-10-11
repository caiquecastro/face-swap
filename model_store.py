"""Explicit, verified provisioning of the InsightFace models used by this app."""

import argparse
import hashlib
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_DIR = Path(__file__).resolve().parent / "models"
SOURCE = "https://github.com/deepinsight/insightface/releases/download/model-zoo/"
# SHA-256 and byte sizes of the official InsightFace model-zoo release assets.
ARCHIVES = {
    "buffalo_l.zip": (288621354, "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"),
    "inswapper_128.onnx": (554253681, "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"),
}
# SHA-256 and byte sizes extracted from the verified buffalo_l.zip asset.
MODELS = {
    "buffalo_l/1k3d68.onnx": (143607619, "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc"),
    "buffalo_l/2d106det.onnx": (5030888, "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf"),
    "buffalo_l/det_10g.onnx": (16923827, "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"),
    "buffalo_l/genderage.onnx": (1322532, "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb"),
    "buffalo_l/w600k_r50.onnx": (174383860, "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"),
    "inswapper_128.onnx": ARCHIVES["inswapper_128.onnx"],
}


def model_dir():
    return Path(os.environ.get("FACE_SWAP_MODEL_DIR", DEFAULT_DIR)).expanduser().resolve()


def valid(path, expected):
    size, digest = expected
    try:
        if not path.is_file() or path.stat().st_size != size:
            return False
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        return actual == digest
    except OSError:
        return False


def verify(root=None):
    root = model_dir() if root is None else Path(root)
    invalid = [name for name, expected in MODELS.items() if not valid(root / name, expected)]
    if invalid:
        raise RuntimeError(
            f"Missing, incomplete, or corrupt models in {root}: {', '.join(invalid)}. "
            "Run `uv run python model_store.py setup` to provision verified models."
        )
    return root


def copy_checked(source, target, expected):
    size, digest = expected
    actual = hashlib.sha256()
    count = 0
    with target.open("wb") as output:
        while chunk := source.read(1024 * 1024):
            count += len(chunk)
            if count > size:
                raise ValueError(f"Model exceeds expected size: {target.name}")
            actual.update(chunk)
            output.write(chunk)
    if count != size or actual.hexdigest() != digest:
        raise ValueError(f"Model checksum or size mismatch: {target.name}")


def download(name, target):
    request = urllib.request.Request(SOURCE + name, headers={"User-Agent": "face-swap-model-setup"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.geturl() != SOURCE + name and not response.geturl().startswith("https://"):
            raise ValueError("Model download redirected outside HTTPS")
        copy_checked(response, target, ARCHIVES[name])


def setup(root=None):
    root = model_dir() if root is None else Path(root)
    root.mkdir(parents=True, exist_ok=True)
    missing = {name for name, expected in MODELS.items() if not valid(root / name, expected)}
    if not missing:
        return root
    # Stage on the same filesystem and validate everything before replacing any file.
    with tempfile.TemporaryDirectory(prefix=".model-setup-", dir=root) as temporary:
        stage = Path(temporary)
        if any(name.startswith("buffalo_l/") for name in missing):
            archive = stage / "buffalo_l.zip"
            download("buffalo_l.zip", archive)
            with zipfile.ZipFile(archive) as bundle:
                expected_names = {name.removeprefix("buffalo_l/") for name in MODELS if name.startswith("buffalo_l/")}
                if set(bundle.namelist()) != expected_names:
                    raise ValueError("Unexpected files in buffalo_l.zip")
                for name in missing:
                    if name.startswith("buffalo_l/"):
                        with bundle.open(name.removeprefix("buffalo_l/")) as source:
                            copy_checked(source, stage / Path(name).name, MODELS[name])
        if "inswapper_128.onnx" in missing:
            download("inswapper_128.onnx", stage / "inswapper_128.onnx")
        for name in sorted(missing):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / target.name, target)
    verify(root)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup", "verify"))
    args = parser.parse_args()
    try:
        print((setup if args.command == "setup" else verify)())
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Model {args.command} failed: {exc}\n")
