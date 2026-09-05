import io
import os
import sqlite3
import struct
import time
import wave
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "runs/detect/pothole_detector/weights/best.pt"
DB_PATH = "gov_reports.db"
CONF_THRESHOLD = 0.4
REPORT_IMG_DIR = "reported_images"

st.set_page_config(page_title="RoadWatch - Pothole Detection", page_icon="🛣️", layout="wide")


def init_db():
    os.makedirs(REPORT_IMG_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        location TEXT,
        confidence REAL,
        image_path TEXT,
        status TEXT DEFAULT 'Pending Review'
    )""")
    con.commit()
    con.close()


def log_report(location, confidence, image_bgr):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname = f"{REPORT_IMG_DIR}/pothole_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    cv2.imwrite(fname, image_bgr)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO reports (timestamp, location, confidence, image_path) VALUES (?, ?, ?, ?)",
        (ts, location, confidence, fname),
    )
    con.commit()
    con.close()


def fetch_reports():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, timestamp, location, confidence, image_path, status FROM reports ORDER BY id DESC"
    ).fetchall()
    con.close()
    return rows


def update_status(report_id, status):
    con = sqlite3.connect(DB_PATH)
    con.execute("UPDATE reports SET status=? WHERE id=?", (status, report_id))
    con.commit()
    con.close()


def make_beep_wav(freq=1200, duration_ms=350, volume=0.5, rate=44100):
    n_samples = int(rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        for i in range(n_samples):
            t = i / rate
            sample = volume * np.sin(2 * np.pi * freq * t)
            wf.writeframesraw(struct.pack("<h", int(sample * 32767)))
    buf.seek(0)
    return buf.read()


BEEP_BYTES = make_beep_wav()


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def run_detection(frame_bgr, model):
    results = model.predict(frame_bgr, conf=CONF_THRESHOLD, verbose=False)[0]
    detected = False
    best_conf = 0.0
    annotated = frame_bgr.copy()
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue
        detected = True
        best_conf = max(best_conf, conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(annotated, f"pothole {conf:.2f}", (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
    return annotated, detected, best_conf


init_db()

st.sidebar.markdown("## RoadWatch")
page = st.sidebar.radio("Navigate", ["Live Detection", "Government Portal"])

if page == "Live Detection":
    st.title("Pothole Detection System")

    if not os.path.exists(MODEL_PATH):
        st.error(f"No trained model found at {MODEL_PATH}")
        st.info("Run training/train_model.py first, or point MODEL_PATH at any YOLOv8 .pt file trained on a pothole dataset.")
        st.stop()

    model = load_model()
    location = st.sidebar.text_input("Road / location description", "MG Road, near KM 4 marker")

    st.sidebar.markdown("### Input settings")
    input_type = st.sidebar.radio("Select input type", ["Image", "Video", "Camera"])

    alert_box = st.empty()
    frame_box = st.empty()
    beep_box = st.empty()

    def handle_detection(annotated_bgr, detected, conf, location):
        frame_box.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        if detected:
            alert_box.error(f"POTHOLE DETECTED — confidence {conf:.2f}")
            beep_box.audio(BEEP_BYTES, format="audio/wav", autoplay=True)
            log_report(location, conf, annotated_bgr)
        else:
            alert_box.success("Road looks clear")

    if input_type == "Image":
        up = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if up:
            img = np.array(Image.open(up).convert("RGB"))[:, :, ::-1]
            annotated, detected, conf = run_detection(img, model)
            handle_detection(annotated, detected, conf, location)

    elif input_type == "Video":
        up = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        if up:
            tmp_path = f"_tmp_{up.name}"
            with open(tmp_path, "wb") as f:
                f.write(up.read())
            run_btn = st.sidebar.button("Process video")
            if run_btn:
                cap = cv2.VideoCapture(tmp_path)
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    annotated, detected, conf = run_detection(frame, model)
                    handle_detection(annotated, detected, conf, location)
                    time.sleep(0.03)
                cap.release()
            os.remove(tmp_path)

    elif input_type == "Camera":
        run_camera = st.sidebar.toggle("Start camera")
        cam_index = st.sidebar.number_input("Camera index", 0, 10, 0, 1)
        if run_camera:
            cap = cv2.VideoCapture(int(cam_index))
            stop_btn = st.sidebar.button("Stop")
            while run_camera and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    st.error("Could not read from camera")
                    break
                annotated, detected, conf = run_detection(frame, model)
                handle_detection(annotated, detected, conf, location)
                if stop_btn:
                    break
            cap.release()

else:
    st.title("Municipal Road Maintenance Portal")

    rows = fetch_reports()
    st.metric("Total reports filed", len(rows))

    statuses = ["Pending Review", "Crew Dispatched", "Repaired"]
    for r in rows:
        rid, ts, location, conf, img_path, status = r
        with st.container(border=True):
            cols = st.columns([1, 3, 1])
            if os.path.exists(img_path):
                cols[0].image(img_path, use_container_width=True)
            with cols[1]:
                st.markdown(f"**Report #{rid}** — {location}")
                st.caption(f"Filed {ts}  ·  confidence {conf:.2f}")
            with cols[2]:
                new_status = st.selectbox(
                    "Status", statuses, index=statuses.index(status), key=f"status_{rid}",
                )
                if new_status != status:
                    update_status(rid, new_status)
                    st.rerun()
