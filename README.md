# Pothole Detection

Pothole detection pipeline: dataset preparation, YOLOv8 training, camera frame
capture, a Streamlit reporting app, and a browser-based demo.

## Structure

```
dataset/    prepare_dataset.py        build a YOLO-format dataset
training/   train_model.py            fine-tune YOLOv8 on the dataset
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

## Model weights

`best.pt` is a generated artifact, not source code, so it isn't committed to
this repo (see `.gitignore`) — it's the output of the training step above,
and it's specific to whatever dataset it was trained on. `app/streamlit_app.py`
expects it at `runs/detect/pothole_detector/weights/best.pt` and will show a
clear error instead of crashing if it isn't there yet.

To get a working model without training from scratch:

- Run `training/train_model.py` yourself against a real pothole dataset, or
- Download a pretrained pothole YOLOv8 model from Roboflow Universe or
  Hugging Face Hub and point `MODEL_PATH` in `streamlit_app.py` at it, or
- Attach your own trained `best.pt` to a GitHub Release on this repo (Releases
  support large binary files, unlike the repo itself) and link it here once
  it's up.

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
