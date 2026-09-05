import random
import shutil
from pathlib import Path

RAW_IMAGES_DIR = "raw_data/images"
RAW_LABELS_DIR = "raw_data/labels"
OUTPUT_DIR = "pothole_dataset"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
CLASSES = ["pothole"]


def download_from_roboflow(api_key, workspace, project, version, out_dir=OUTPUT_DIR):
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download("yolov8", location=out_dir)
    return ds.location


def build_dataset_from_local():
    random.seed(42)
    images = sorted(
        p for p in Path(RAW_IMAGES_DIR).glob("*.*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not images:
        raise SystemExit(f"No images found in {RAW_IMAGES_DIR}")
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split, files in splits.items():
        img_out = Path(OUTPUT_DIR) / split / "images"
        lbl_out = Path(OUTPUT_DIR) / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            label_path = Path(RAW_LABELS_DIR) / (img_path.stem + ".txt")
            shutil.copy(img_path, img_out / img_path.name)
            if label_path.exists():
                shutil.copy(label_path, lbl_out / label_path.name)
            else:
                (lbl_out / (img_path.stem + ".txt")).touch()

    write_data_yaml()


def write_data_yaml():
    yaml_content = f"""path: {Path(OUTPUT_DIR).resolve()}
train: train/images
val: val/images
test: test/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    out = Path(OUTPUT_DIR) / "data.yaml"
    out.write_text(yaml_content)


if __name__ == "__main__":
    build_dataset_from_local()
