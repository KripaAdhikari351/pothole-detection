from ultralytics import YOLO 

DATA_YAML = "pothole_dataset/data.yaml"
EPOCHS = 100
IMG_SIZE = 640
BATCH = 16
MODEL_BASE = "yolov8n.pt"


def main():
    model = YOLO(MODEL_BASE)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name="pothole_detector",
        patience=20,
    )
    model.val()
    model.export(format="onnx")


if __name__ == "__main__":
    main()
