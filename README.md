# OCR Research — European License Plate Recognition

Experiments with reading European license plates from photos. The repo combines two parts:

1. **Plate detection**: a YOLOv8 model fine-tuned to find license plates in an image.
2. **Text recognition**: [EasyOCR](https://github.com/JaidedAI/EasyOCR) reads the characters, with tests of how its parameters and image preprocessing change the results.

## Repository layout

| Path | What it is |
|------|------------|
| [`easy-ocr-code.py`](easy-ocr-code.py) | Minimal script. Runs EasyOCR on every image in `images/` and saves annotated copies to `images/ocr_output/`. |
| [`easy-ocr-tuning.ipynb`](easy-ocr-tuning.ipynb) | Tests EasyOCR parameters (sensitivity thresholds, `mag_ratio`, decoder, character allowlist) on distant and close-up photos. Includes written observations and conclusions. |
| [`ocr-color-filter-comparison.ipynb`](ocr-color-filter-comparison.ipynb) | Compares 9 preprocessing filters (grayscale, CLAHE, inversion, HSV/LAB channels, Otsu/adaptive threshold, yellow/white mask) by OCR confidence on close-up plates. |
| [`yolo-training.ipynb`](yolo-training.ipynb) | Fine-tunes `yolov8n.pt` on the plate dataset, validates it, and runs inference on the test set and on `images/`. |
| [`docs/easy-ocr-tuning.html`](docs/easy-ocr-tuning.html) | HTML export of the tuning notebook with all outputs, viewable without Jupyter. |
| `images/` | Test photos: whole cars at a distance (top level) and close-ups of plates (`close-up/`). |
| `images-eu/` | [European License Plates](https://universe.roboflow.com/patrick-neicu/european-license-plates) dataset from Roboflow (CC BY 4.0), with YOLO-format labels and `train/`, `valid/` and `test/` splits. |
| `runs/detect/` | YOLO outputs: training run with weights (`train-result/eu_license_plates/weights/best.pt`), validation plots and prediction images. |
| `yolov8n.pt`, `yolo26n.pt` | Pre-trained Ultralytics base weights. |

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install easyocr opencv-python numpy matplotlib pandas pyyaml ultralytics jupyter
```

EasyOCR downloads its detection and recognition models the first time it runs. The notebooks use `gpu=False`, so a GPU is not required.

## Usage

**Quick OCR pass on your own images:** put `.jpg`, `.jpeg` or `.png` files into `images/` and run

```bash
python easy-ocr-code.py
```

The script prints the detected text and confidence for each image and writes annotated copies to `images/ocr_output/`.

**Notebooks:** start Jupyter from the repository root, because all paths are relative to it:

```bash
jupyter notebook
```

- `easy-ocr-tuning.ipynb` reads from `images/`.
- `ocr-color-filter-comparison.ipynb` reads from `images/close-up/`.
- `yolo-training.ipynb` creates `images-eu/data_fixed.yaml` with absolute paths for your machine, then trains. To skip training and only run inference, run the first cell and then the cells after **Train**. They load the committed `best.pt`.

## Findings

From `easy-ocr-tuning.ipynb` (full write-up in the notebook and in [`docs/easy-ocr-tuning.html`](docs/easy-ocr-tuning.html)):

- **Sensitivity** (`text_threshold`, `low_text`): strict settings (0.9 / 0.6) work best on distant photos because they filter out background text. Default settings (0.7 / 0.4) work best on close-ups. Loose settings (0.4 / 0.2) add noise and text that isn't there.
- **Magnification** (`mag_ratio`): has little effect on distant photos. On close-ups it matters: a value of 0.5 misses small characters, and 1.5 to 2.0 recovers them.
- **Decoder**: `greedy` and `beamsearch` gave the same results on these images.
- **Allowlist** (`A–Z 0–9 -`): raises confidence on plates in distant photos but hurts the other text in the scene. It made no difference on close-ups.
- **Conclusion:** no single configuration works best everywhere. Choose parameters based on how close the shot is and how large the text is.

Suggested starting configurations:

| Parameter | Distant photos | Close-ups |
|-----------|---------------:|----------:|
| `text_threshold` | 0.9 | 0.6 |
| `low_text` | 0.6 | 0.4 |
| `mag_ratio` | 1.0 | 1.8 |
| `link_threshold` | 0.5 | 0.5 |
| `width_ths` | 0.7 | 0.7 |
| `decoder` | `beamsearch` (beamWidth 5) | `beamsearch` (beamWidth 5) |

## YOLO detector status

The committed model is an early baseline: 10 epochs of `yolov8n` at 640 px, trained on the `train/images` split. On the validation set it reaches **mAP50 ≈ 0.58** and **mAP50-95 ≈ 0.49**, but precision is very low, so it predicts many false boxes. The full metrics are in `runs/detect/train-result/eu_license_plates/results.csv`. Training for more epochs, using a larger backbone (`yolov8s`/`yolov8m`), or adding the images in the `extra/` folders should improve it.

## Next steps

- Crop plates found by YOLO and pass only the crops to EasyOCR, using the close-up configuration.
- Improve the detector (see above).
- Add plate-format checks per country after OCR.

## Data & license

The `images-eu` dataset is *European License Plates* by patrick-neicu on Roboflow Universe, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [`images-eu/README.dataset.txt`](images-eu/README.dataset.txt).
