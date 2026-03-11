import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import serial
import struct
import numpy as np
import cv2
from ultralytics import YOLO
from thermal_services import get_temp_for_object

# ── Configuration ────────────────────────────────────────────────────────────
PORT = '/dev/cu.usbserial-1140'
BAUD = 600000
YOLO_CONF = 0.4          # Minimum confidence threshold (0-1)
YOLO_EVERY_N = 2         # Run YOLO on every Nth camera frame (1 = every frame)
# ─────────────────────────────────────────────────────────────────────────────

CAM_MAGIC   = b'\xFF\xAA\xBB\xCC'
THERM_MAGIC = b'\xFF\xDD\xEE\x11'
THERMAL_SIZE = 32 * 24 * 4  # 768 floats x 4 bytes = 3072 bytes

# Colour palette for bounding boxes — one BGR colour per class index
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72),  (23, 204, 146), (134, 219, 61),
    (52, 147, 26),  (187, 212, 0),  (168, 153, 44), (255, 194, 0),
    (255, 141, 42), (255, 71, 76),  (255, 0, 127),  (236, 24, 0),
    (255, 16, 229), (133, 0, 255),  (68, 0, 255),   (0, 0, 155),
]


def colour_for(class_id: int):
    return PALETTE[class_id % len(PALETTE)]


def get_raw_temp_for_box(thermal_grid, x1, y1, x2, y2, cam_w=320, cam_h=240):
    """Map a bounding box from camera space to the 32x24 thermal grid and return the avg temp."""
    if thermal_grid is None:
        return None
    th, tw = thermal_grid.shape  # 24, 32

    # Scale camera coords to thermal grid coords
    tx1 = max(0, int(x1 * tw / cam_w))
    ty1 = max(0, int(y1 * th / cam_h))
    tx2 = min(tw, int(x2 * tw / cam_w))
    ty2 = min(th, int(y2 * th / cam_h))

    if tx2 <= tx1 or ty2 <= ty1:
        return None

    region = thermal_grid[ty1:ty2, tx1:tx2]
    return float(np.mean(region))


def draw_detections(frame, results_list, thermal_grid=None):
    """Draw bounding boxes + labels from one or more YOLO Results onto frame."""
    for results in results_list:
        boxes = results[0].boxes
        names = results[0].names

        for box in boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            class_name = names[cls_id]
            colour = colour_for(cls_id)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get corrected temperature for this object
            raw_temp = get_raw_temp_for_box(thermal_grid, x1, y1, x2, y2)
            if raw_temp is not None:
                result = get_temp_for_object(class_name, raw_temp)
                temp_str = f" {result['true_temp_c']}C"
            else:
                temp_str = ""

            label = f"{class_name} {conf:.2f}{temp_str}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 2, y1), colour, -1)
            cv2.putText(frame, label, (x1 + 1, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def read_next_packet(ser):
    """Scan for a magic header and return ('camera', jpg) or ('thermal', temps)."""
    buf = b''
    while True:
        byte = ser.read(1)
        if len(byte) == 0:
            continue
        buf += byte

        if buf[-4:] == CAM_MAGIC:
            raw_len = ser.read(4)
            if len(raw_len) < 4:
                buf = b''
                continue
            img_len = struct.unpack('<I', raw_len)[0]
            if img_len == 0 or img_len > 150_000:
                buf = b''
                continue
            jpg = ser.read(img_len)
            if len(jpg) == img_len:
                return ('camera', jpg)
            buf = b''

        elif buf[-4:] == THERM_MAGIC:
            raw = ser.read(THERMAL_SIZE)
            if len(raw) == THERMAL_SIZE:
                temps = np.frombuffer(raw, dtype=np.float32).reshape(24, 32)
                return ('thermal', temps)
            buf = b''

        if len(buf) > 4:
            buf = buf[-4:]


def thermal_to_colormap(temps):
    """Convert 24x32 temperature array to a 240x320 BGR heatmap image."""
    # Flip to match camera orientation (mirror horizontally + vertically)
    temps = np.fliplr(temps)

    t_min, t_max = temps.min(), temps.max()
    if t_max - t_min < 0.1:
        t_max = t_min + 0.1
    normalized = ((temps - t_min) / (t_max - t_min) * 255).astype(np.uint8)

    upscaled = cv2.resize(normalized, (320, 240), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(upscaled, cv2.COLORMAP_JET)
    return heatmap, t_min, t_max


def main():
    print("Loading models ...")
    model_base = YOLO("yolo11m.pt")                                          # 80 COCO classes
    model_custom = YOLO("../../runs/detect/pancake_pan2/weights/best.pt")     # pan + pancake
    print("Both models ready.")

    print(f"Opening {PORT} at {BAUD} baud ...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=5)
    except serial.SerialException as e:
        print(f"Error: {e}")
        return

    print("Connected! Waiting for frames — press Q to quit.")

    last_camera = np.zeros((240, 320, 3), dtype=np.uint8)
    last_thermal = np.zeros((240, 320, 3), dtype=np.uint8)
    last_thermal_grid = None  # raw 24x32 temperature array (flipped)
    t_min, t_max = 0.0, 0.0
    frame_count = 0
    last_results_list = []

    while True:
        packet = read_next_packet(ser)
        if packet is None:
            continue

        ptype, data = packet

        if ptype == 'camera':
            img_np = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frame = cv2.resize(frame, (320, 240))
            frame_count += 1

            # Run both models every N camera frames
            if frame_count % YOLO_EVERY_N == 0:
                last_results_list = [
                    model_base(frame, conf=YOLO_CONF, verbose=False),
                    model_custom(frame, conf=YOLO_CONF, verbose=False),
                ]

            if last_results_list:
                frame = draw_detections(frame, last_results_list, last_thermal_grid)

            last_camera = frame

        elif ptype == 'thermal':
            last_thermal_grid = np.fliplr(data)  # keep raw temps for temperature lookup
            last_thermal, t_min, t_max = thermal_to_colormap(data)

        # Side by side: camera+YOLO on left, thermal heatmap on right
        combined = np.hstack((last_camera, last_thermal))

        # Frame counter on camera side
        cv2.putText(combined, f"frame {frame_count}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Temperature range on thermal side
        label = f"{t_min:.1f}C - {t_max:.1f}C"
        cv2.putText(combined, label, (330, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Camera + YOLO + Thermal", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting.")
            break

    ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
