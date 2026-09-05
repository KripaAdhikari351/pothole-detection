# Pothole Detection   

Pothole detection pipeline: dataset preparation, YOLOv8 training, camera frame
capture, a Streamlit reporting app, and a browser-based demo.

## Structure

```
dataset/    prepare_dataset.py        build a YOLO-format dataset
training/   train_model.py            fine-tune YOLOv8 on the dataset
training/   train_on_colab.ipynb      same training, runnable on a free Colab GPU
capture/    capture_camera_frames.py  grab frames from a live camera
app/        streamlit_app.py          detection + reporting dashboard
web-demo/   index.html                self-contained browser demo
```

## Setup

```
pip install -r requirements.txt
```

## Dataset

Place raw images and YOLO-format label files in `raw_data/images` and
`raw_data/labels`, then run:

```
python dataset/prepare_dataset.py
```

This produces `pothole_dataset/` with `train/val/test` splits and a
`data.yaml`. `download_from_roboflow()` in the same file can pull a public
dataset from Roboflow Universe instead, given an API key.

## Training

```
python training/train_model.py
```

Weights are written to `runs/detect/pothole_detector/weights/best.pt`.

No local GPU? Open [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KripaAdhikari351/pothole-detection/blob/main/training/train_on_colab.ipynb) and run the cells top to bottom on Colab's free GPU. It downloads a public pothole dataset from Roboflow Universe, trains, and gives you `best.pt` to download.

## Model weights

`best.pt` is a generated artifact, not source code, so it isn't committed to
this repo (see `.gitignore`) — it's the output of the training step above,
and it's specific to whatever dataset it was trained on. `app/streamlit_app.py`
expects it at `runs/detect/pothole_detector/weights/best.pt` and will show a
clear error instead of crashing if it isn't there yet.

Once you have a trained `best.pt`:

1. Place it at `runs/detect/pothole_detector/weights/best.pt` to run
   `streamlit_app.py` locally, or
2. Attach it to a [GitHub Release](https://github.com/KripaAdhikari351/pothole-detection/releases/new)
   on this repo (Releases support large binary files, unlike the repo
   itself) so others don't have to train their own. Once uploaded, download
   it with:
   ```
   mkdir -p runs/detect/pothole_detector/weights
   curl -L -o runs/detect/pothole_detector/weights/best.pt <release_asset_url>
   ```
   replacing `<release_asset_url>` with the asset link from the release page.

## Camera capture

```
python capture/capture_camera_frames.py
```

## Streamlit app

```
streamlit run app/streamlit_app.py
```

Loads the trained weights above and runs detection from an uploaded image,
uploaded video, or a connected camera. Detections are logged to a local
SQLite database and shown on a Government Portal page.

## Web demo

`web-demo/index.html` is a self-contained, no-build browser demo: image,
video, and camera input; a heuristic pothole detector (local adaptive
thresholding + connected components) with a multi-frame tracker for stable
bounding boxes; a report-confirmation dialog with simulated GPS and a
generated complaint string; and a Government Portal report list. Open the
file directly in a browser.
