import os
import time

import cv2

CAMERA_INDEX = 0
SAVE_DIR = "captured_frames"
SAVE_EVERY_N_FRAMES = 15


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}")

    frame_count = 0
    saved_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1

        if frame_count % SAVE_EVERY_N_FRAMES == 0:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SAVE_DIR, f"frame_{ts}_{saved_count:05d}.jpg")
            cv2.imwrite(path, frame)
            saved_count += 1

        cv2.imshow("Camera feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
