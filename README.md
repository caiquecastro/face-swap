# Face Swap

Small Python script for swapping a source face into a target image using `insightface` and the `inswapper_128.onnx` model.

## What It Uses

- `insightface` for face detection, face analysis, and swap model loading
- `opencv-python` (`cv2`) for image input/output and preview
- `inswapper_128.onnx` as the pretrained face swap model

## How It Works

1. Detect the face in the source image.
2. Detect all faces in the target image.
3. Replace each target face with the source face.
4. Save the result as `swapped_output.jpg`.

The current script reads:

- `face.jpg` as the source face
- `body.jpg` as the target image

## Project Files

- [`app.py`](/app.py) main script
- [`inswapper_128.onnx`](/inswapper_128.onnx) swap model
- image files in the project root used as sample inputs

## Setup

Create a virtual environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install insightface opencv-python onnxruntime
```

If you want GPU inference, you may need a GPU-enabled runtime depending on your environment.

## Run

```bash
python3 app.py
```

The script will:

- load the source and target images
- run face detection
- perform the swap
- write `swapped_output.jpg`
- open a preview window with the result

## GPU / CPU Note

The script currently uses:

```python
app.prepare(ctx_id=0, det_size=(640, 640))
```

`ctx_id=0` assumes a GPU-capable runtime/device is available. If you want to try CPU execution, change it to:

```python
app.prepare(ctx_id=-1, det_size=(640, 640))
```

## Changing Inputs

Edit the filenames in [`app.py`](/app.py) here:

- `img1 = cv2.imread("face.jpg")`
- `img2 = cv2.imread("body.jpg")`

You can point them to any source and target images you want to test.

## Notes

- The script uses the first detected face from the source image.
- It swaps all detected faces in the target image.
- Results depend heavily on image quality, pose, lighting, and occlusion.

## Responsible Use

Face swapping can be deceptive or harmful if used without consent. Use this project only for lawful, ethical, and clearly disclosed purposes.
