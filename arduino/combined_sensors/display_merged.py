import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import serial
import struct
import numpy as np
import cv2
from ultralytics import YOLO
from thermal_services import get_temp_for_object
from homography import HomographyCalibrator
 
# ── Configuration ────────────────────────────────────────────────────────────
PORT = '/dev/cu.usbserial-2140'
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
 
 
def get_raw_temp_for_box(thermal_grid, x1, y1, x2, y2, cam_w=240, cam_h=320):
    """Map the center 50% of a bounding box to the 32x24 thermal grid and return the avg temp."""
    if thermal_grid is None:
        return None
    th, tw = thermal_grid.shape  # 24, 32

    # Shrink box to center 50% (half width, half height)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    qw, qh = (x2 - x1) / 4, (y2 - y1) / 4  # quarter of box size = half-extent of center region
    cx1, cy1 = cx - qw, cy - qh
    cx2, cy2 = cx + qw, cy + qh

    # Scale center region to thermal grid coords
    tx1 = max(0, int(cx1 * tw / cam_w))
    ty1 = max(0, int(cy1 * th / cam_h))
    tx2 = min(tw, int(cx2 * tw / cam_w))
    ty2 = min(th, int(cy2 * th / cam_h))

    if tx2 <= tx1 or ty2 <= ty1:
        return None

    region = thermal_grid[ty1:ty2, tx1:tx2]
    return float(np.mean(region))
 
 
def draw_detections(frame, results_list, thermal_grid=None, calibrator=None):
    """Draw bounding boxes + detection list at bottom-left."""
    # Pre-warp thermal if calibrator is ready
    warped_thermal = calibrator.warp_thermal(thermal_grid) \
        if calibrator and calibrator.calibrated and thermal_grid is not None else None

    detections = []
    for results in results_list:
        boxes = results[0].boxes
        names = results[0].names

        for box in boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            class_name = names[cls_id]
            colour = colour_for(cls_id)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Use homography-warped thermal if calibrated, else naive fallback
            if warped_thermal is not None:
                raw_temp = calibrator.get_temp_for_box(warped_thermal, x1, y1, x2, y2)
            else:
                raw_temp = get_raw_temp_for_box(thermal_grid, x1, y1, x2, y2)
            if raw_temp is not None:
                result = get_temp_for_object(class_name, raw_temp)
                temp_str = f" {result['true_temp_c']}C"
            else:
                temp_str = ""
 
            label = f"{class_name} {conf:.2f}{temp_str}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            detections.append((label, colour))
 
    # Draw detection list at bottom-left
    y_offset = frame.shape[0] - 8 - len(detections) * 20
    for label, colour in detections:
        cv2.putText(frame, label, (6, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)
        cv2.putText(frame, label, (6, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 20
 
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
    """Convert 24x32 temperature array to a 320x240 BGR heatmap image (rotated 180°)."""
    # Rotate 180° by flipping both axes
    temps = np.flip(temps)
    temps = np.fliplr(temps)
 
    t_min, t_max = temps.min(), temps.max()
    if t_max - t_min < 0.1:
        t_max = t_min + 0.1
    normalized = ((temps - t_min) / (t_max - t_min) * 255).astype(np.uint8)
 
    upscaled = cv2.resize(normalized, (320, 240), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(upscaled, cv2.COLORMAP_JET)
    return heatmap, t_min, t_max
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom-only", action="store_true",
                        help="Run only the custom trained model (no base YOLO)")
    args = parser.parse_args()

    print("Loading models ...")
    model_custom = YOLO("../../runs/detect/pancake_pan2/weights/best.pt")     # pan + pancake
    model_base = None
    if not args.custom_only:
        model_base = YOLO("yolo11m.pt")                                      # 80 COCO classes
        print("Both models ready.")
    else:
        print("Custom model ready (base YOLO disabled).")
 
    print(f"Opening {PORT} at {BAUD} baud ...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=5)
    except serial.SerialException as e:
        print(f"Error: {e}")
        return
 
    print("Connected! Waiting for frames — press Q to quit.")
 
    last_camera = np.zeros((320, 240, 3), dtype=np.uint8)  # 90° CW: height/width swapped
    last_thermal = np.zeros((240, 320, 3), dtype=np.uint8)
    last_thermal_grid = None  # raw 24x32 temperature array (rotated 180°)
    t_min, t_max = 0.0, 0.0
    frame_count = 0
    last_results_list = []

    # ── Homography calibrator ───────────────────────────────────────────
    calib_path = os.path.join(os.path.dirname(__file__), '..', '..', 'homography', 'calibration.json')
    calibrator = HomographyCalibrator(thermal_shape=(24, 32), rgb_shape=(320, 240))
    if os.path.exists(calib_path):
        calibrator.load(calib_path)
        print(f"Loaded previous calibration: {calibrator.status_text}")
    else:
        print("No saved calibration — will auto-calibrate from detections.")
 
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
            # Rotate 90° clockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame_count += 1
 
            # Run model(s) every N camera frames
            if frame_count % YOLO_EVERY_N == 0:
                last_results_list = [model_custom(frame, conf=YOLO_CONF, verbose=False)]
                if model_base is not None:
                    last_results_list.append(model_base(frame, conf=YOLO_CONF, verbose=False))

            # ── Homography: collect correspondences from custom model detections ──
            if last_thermal_grid is not None and last_results_list and not calibrator.calibrated:
                custom_boxes = last_results_list[0][0].boxes
                if len(custom_boxes) > 0:
                    best_idx = int(custom_boxes.conf.argmax())
                    bbox = tuple(map(int, custom_boxes.xyxy[best_idx]))
                    if calibrator.add_correspondence(last_thermal_grid, bbox):
                        print(f"[calib] {calibrator.status_text}")

            if last_results_list:
                frame = draw_detections(frame, last_results_list, last_thermal_grid, calibrator)
 
            last_camera = frame
 
        elif ptype == 'thermal':
            # Rotate 180°: flip both axes on raw grid for temperature lookups
            last_thermal_grid = np.flip(data)
            last_thermal_grid = np.fliplr(last_thermal_grid)
            last_thermal, t_min, t_max = thermal_to_colormap(data)
 
        # Resize thermal to match rotated camera height (320) for side-by-side display
        thermal_display = cv2.resize(last_thermal, (320, 320))
 
        # Side by side: camera+YOLO on left (240x320), thermal heatmap on right (320x320)
        combined = np.hstack((last_camera, thermal_display))
 
        # Frame counter + calibration status on camera side
        cv2.putText(combined, f"frame {frame_count}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(combined, calibrator.status_text, (6, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
 
        # Temperature range on thermal side
        label = f"{t_min:.1f}C - {t_max:.1f}C"
        cv2.putText(combined, label, (330, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
 
        cv2.imshow("Camera + YOLO + Thermal", combined)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting.")
            break

    # Save calibration for next run
    if calibrator.num_points > 0:
        calibrator.save(calib_path)

    ser.close()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()