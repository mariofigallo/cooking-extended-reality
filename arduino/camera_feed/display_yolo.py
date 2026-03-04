import serial
import numpy as np
import cv2
import struct
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────
PORT = '/dev/cu.usbserial-110'       # Windows: e.g. COM5  |  Linux/Mac: e.g. /dev/ttyUSB0
BAUD = 460800
YOLO_CONF = 0.4          # Minimum confidence threshold (0–1)
YOLO_EVERY_N = 2         # Run YOLO on every Nth frame (1 = every frame, higher = faster)
# ──────────────────────────────────────────────────────────────────────────────

MAGIC = b'\xFF\xAA\xBB\xCC'

# Colour palette — one BGR colour per class index (cycles if >20 classes)
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72),  (23, 204, 146), (134, 219, 61),
    (52, 147, 26),  (187, 212, 0),  (168, 153, 44), (255, 194, 0),
    (255, 141, 42), (255, 71, 76),  (255, 0, 127),  (236, 24, 0),
    (255, 16, 229), (133, 0, 255),  (68, 0, 255),   (0, 0, 155),
]


def colour_for(class_id: int):
    return PALETTE[class_id % len(PALETTE)]


def draw_detections(frame, results):
    """Draw bounding boxes + labels from a YOLO Results object onto frame."""
    boxes = results[0].boxes
    names = results[0].names

    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = f"{names[cls_id]} {conf:.2f}"
        colour = colour_for(cls_id)

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Label background
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 2, y1), colour, -1)

        # Label text
        cv2.putText(frame, label, (x1 + 1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def wait_for_magic(ser):
    """Scan incoming bytes until the 4-byte magic header is found."""
    buf = b''
    while True:
        buf += ser.read(1)
        if buf[-4:] == MAGIC:
            return True
        if len(buf) > 4:
            buf = buf[-4:]


def read_frame(ser):
    """Block until a full JPEG frame arrives, then return raw bytes."""
    if not wait_for_magic(ser):
        return None

    raw_len = ser.read(4)
    if len(raw_len) < 4:
        return None
    img_len = struct.unpack('<I', raw_len)[0]

    if img_len == 0 or img_len > 150_000:
        return None

    jpg = ser.read(img_len)
    if len(jpg) < img_len:
        return None

    return jpg


def main():
    print("Loading YOLOv8s model ...")
    model = YOLO("yolov8s.pt")   # downloads automatically on first run
    print("Model ready.")

    print(f"Opening serial port {PORT} at {BAUD} baud ...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=5)
    except serial.SerialException as e:
        print(f"Error: {e}")
        print("Check that the correct PORT is set at the top of this file.")
        return

    print("Connected! Waiting for frames — press Q to quit.")

    frame_count  = 0
    last_results = None   # cache last YOLO result so non-inferred frames still show boxes

    while True:
        jpg = read_frame(ser)
        if jpg is None:
            print("Bad/missing frame, retrying ...")
            continue

        img_np = np.frombuffer(jpg, dtype=np.uint8)
        frame  = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if frame is None:
            print("Could not decode JPEG, skipping ...")
            continue

        frame_count += 1

        # Run YOLO inference every N frames
        if frame_count % YOLO_EVERY_N == 0:
            last_results = model(frame, conf=YOLO_CONF, verbose=False)

        # Overlay detections if we have any
        if last_results is not None:
            frame = draw_detections(frame, last_results)

        # FPS counter overlay
        cv2.putText(frame, f"frame {frame_count}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("ESP32-cam + YOLOv8n", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting.")
            break

    ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()