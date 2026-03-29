"""
Pancake Monitor — Web App
--------------------------
Flask + Socket.IO web interface for the thermal pancake cooking assistant.
Reuses the same detection/tracking logic from cooking_pancake.py.
"""

import sys, os, time, math, threading, base64
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import serial
import struct
import numpy as np
import cv2
from collections import deque
from ultralytics import YOLO
from flask import Flask, render_template
from flask_socketio import SocketIO

from homography import HomographyCalibrator
from thermal_services import estimate_temp, get_properties

# ── Configuration ────────────────────────────────────────────────────────────
PORT            = '/dev/cu.usbserial-210'
BAUD            = 600000

THERM_ROWS      = 24
THERM_COLS      = 32
THERM_SIZE      = THERM_ROWS * THERM_COLS * 4

CAM_W           = 240
CAM_H           = 320

PAN_TEMP_THRESH       = 70.0
PANCAKE_COLD_DELTA    = 100.0
MIN_PAN_AREA_PX       = 40
MIN_PANCAKE_AREA_PX   = 30
PAN_CIRCULARITY_MIN   = 0.35

CALIB_YOLO_CONF       = 0.4
CALIB_YOLO_EVERY_N    = 3

COOK_TABLE = {
    (100, 130): (240, 180),
    (130, 160): (180, 120),
    (160, 190): (120,  90),
    (190, 220): (80,   60),
    (220, 280): (60,   45),
}
DEFAULT_COOK_TIME = (150, 100)

FLIP_TEMP_THRESHOLD   = 75.0
DONE_TEMP_THRESHOLD   = 80.0
TEMP_RISE_RATE_FLIP   = 0.3
TEMP_HISTORY_LEN      = 30
PAN_TEMP_HISTORY_LEN  = 20
PANCAKE_DETECT_HYSTERESIS = 30

CAM_MAGIC   = b'\xFF\xAA\xBB\xCC'
THERM_MAGIC = b'\xFF\xDD\xEE\x11'


# ── Cook State Machine (same as cooking_pancake.py) ──────────────────────────

class CookState:
    IDLE     = "WAITING"
    SIDE_1   = "SIDE 1"
    FLIP_NOW = "FLIP"
    SIDE_2   = "SIDE 2"
    DONE     = "DONE"


class CookTracker:
    def __init__(self):
        self.state = CookState.IDLE
        self.side1_start = None
        self.side2_start = None
        self.flip_time = None
        self.done_time = None
        self.target_side1 = 0
        self.target_side2 = 0
        self.pancake_temp_history = deque(maxlen=TEMP_HISTORY_LEN)
        self.pan_temp_history = deque(maxlen=PAN_TEMP_HISTORY_LEN)
        self.pan_temp_at_start = None
        self.flip_acknowledged = False

    def lookup_cook_times(self, pan_temp_c):
        for (lo, hi), (s1, s2) in COOK_TABLE.items():
            if lo <= pan_temp_c < hi:
                return s1, s2
        return DEFAULT_COOK_TIME

    def pancake_detected(self, pan_temp, pancake_temp, timestamp):
        if self.state != CookState.IDLE:
            return
        self.pan_temp_at_start = pan_temp
        self.target_side1, self.target_side2 = self.lookup_cook_times(pan_temp)
        self.side1_start = timestamp
        self.state = CookState.SIDE_1
        self.pancake_temp_history.clear()

    def update(self, pancake_temp, pan_temp, timestamp):
        if pancake_temp is not None:
            self.pancake_temp_history.append((timestamp, pancake_temp))
        if pan_temp is not None:
            self.pan_temp_history.append((timestamp, pan_temp))

        if self.state == CookState.SIDE_1:
            elapsed = timestamp - self.side1_start
            progress = min(1.0, elapsed / self.target_side1) if self.target_side1 > 0 else 0
            rate = self._temp_rise_rate()
            temp_ready = (pancake_temp is not None and pancake_temp >= FLIP_TEMP_THRESHOLD)
            rate_ready = (rate is not None and rate < TEMP_RISE_RATE_FLIP and elapsed > 30)
            if progress >= 1.0 or (temp_ready and progress > 0.6) or (rate_ready and progress > 0.5):
                self.state = CookState.FLIP_NOW
                self.flip_time = timestamp
                self.flip_acknowledged = False

        elif self.state == CookState.FLIP_NOW:
            if timestamp - self.flip_time > 3.0:
                if pancake_temp is not None and len(self.pancake_temp_history) > 5:
                    recent_temps = [t for _, t in list(self.pancake_temp_history)[-5:]]
                    if self.flip_acknowledged or (max(recent_temps) - min(recent_temps) > 8):
                        self.side2_start = timestamp
                        self.state = CookState.SIDE_2
                        self.pancake_temp_history.clear()
                if timestamp - self.flip_time > 15.0:
                    self.side2_start = timestamp
                    self.state = CookState.SIDE_2
                    self.pancake_temp_history.clear()

        elif self.state == CookState.SIDE_2:
            elapsed = timestamp - self.side2_start
            progress = min(1.0, elapsed / self.target_side2) if self.target_side2 > 0 else 0
            temp_ready = (pancake_temp is not None and pancake_temp >= DONE_TEMP_THRESHOLD)
            if progress >= 1.0 or (temp_ready and progress > 0.6):
                self.state = CookState.DONE
                self.done_time = timestamp

        elif self.state == CookState.DONE:
            if timestamp - self.done_time > 10.0:
                self.reset()

    def acknowledge_flip(self):
        if self.state == CookState.FLIP_NOW:
            self.flip_acknowledged = True

    def reset(self):
        self.__init__()

    def progress(self, timestamp):
        if self.state == CookState.SIDE_1 and self.side1_start:
            elapsed = timestamp - self.side1_start
            return min(1.0, elapsed / self.target_side1) if self.target_side1 > 0 else 0
        elif self.state == CookState.SIDE_2 and self.side2_start:
            elapsed = timestamp - self.side2_start
            return min(1.0, elapsed / self.target_side2) if self.target_side2 > 0 else 0
        elif self.state in (CookState.FLIP_NOW, CookState.DONE):
            return 1.0
        return 0.0

    def smoothed_pan_temp(self):
        if not self.pan_temp_history:
            return None
        return float(np.mean([t for _, t in self.pan_temp_history]))

    def smoothed_pancake_temp(self):
        if not self.pancake_temp_history:
            return None
        temps = [t for _, t in self.pancake_temp_history]
        if len(temps) >= 5:
            weights = np.linspace(0.5, 1.0, len(temps))
            return float(np.average(temps, weights=weights))
        return float(np.mean(temps))

    def elapsed_str(self, timestamp):
        if self.state == CookState.SIDE_1 and self.side1_start:
            e = timestamp - self.side1_start
        elif self.state == CookState.SIDE_2 and self.side2_start:
            e = timestamp - self.side2_start
        elif self.state == CookState.FLIP_NOW and self.side1_start:
            e = timestamp - self.side1_start
        elif self.state == CookState.DONE and self.side1_start:
            e = (self.done_time or timestamp) - self.side1_start
        else:
            return "--:--"
        m, s = divmod(int(e), 60)
        return f"{m}:{s:02d}"

    def total_elapsed_str(self, timestamp):
        if self.side1_start is None:
            return "--:--"
        end = self.done_time or timestamp
        e = end - self.side1_start
        m, s = divmod(int(e), 60)
        return f"{m}:{s:02d}"

    def _temp_rise_rate(self):
        if len(self.pancake_temp_history) < 10:
            return None
        recent = list(self.pancake_temp_history)[-10:]
        dt = recent[-1][0] - recent[0][0]
        if dt < 1.0:
            return None
        return (recent[-1][1] - recent[0][1]) / dt


# ── Thermal detection (same as cooking_pancake.py) ───────────────────────────

def detect_pan_and_pancake(thermal_grid, warped=False):
    if thermal_grid is None:
        return None, None, None, None

    h, w = thermal_grid.shape

    if warped:
        scale = (h * w) / (THERM_ROWS * THERM_COLS)
        min_pan_area = int(MIN_PAN_AREA_PX * scale)
        min_pc_area = int(MIN_PANCAKE_AREA_PX * scale)
        ksize = max(3, int(3 * (h / THERM_ROWS)))
        if ksize % 2 == 0:
            ksize += 1
    else:
        min_pan_area = MIN_PAN_AREA_PX
        min_pc_area = MIN_PANCAKE_AREA_PX
        ksize = 3

    if warped:
        valid_mask = ~np.isnan(thermal_grid)
        thermal_safe = np.nan_to_num(thermal_grid, nan=0.0)
    else:
        valid_mask = np.ones((h, w), dtype=bool)
        thermal_safe = thermal_grid

    pan_mask_raw = ((thermal_safe >= PAN_TEMP_THRESH) & valid_mask).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    pan_mask_clean = cv2.morphologyEx(pan_mask_raw, cv2.MORPH_CLOSE, kernel)
    pan_mask_clean = cv2.morphologyEx(pan_mask_clean, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(pan_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None

    pan_contour = max(contours, key=cv2.contourArea)
    pan_area = cv2.contourArea(pan_contour)
    if pan_area < min_pan_area:
        return None, None, None, None

    perimeter = cv2.arcLength(pan_contour, True)
    if perimeter > 0:
        circularity = 4 * math.pi * pan_area / (perimeter * perimeter)
        if circularity < PAN_CIRCULARITY_MIN:
            return pan_contour, None, None, None

    pan_filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(pan_filled, [pan_contour], -1, 255, cv2.FILLED)
    pan_rim_mask = cv2.bitwise_and(pan_mask_clean, pan_filled)
    pan_pixels = thermal_safe[pan_rim_mask > 0]
    pan_temp = float(np.max(pan_pixels)) if len(pan_pixels) > 0 else None

    if pan_temp is None:
        return pan_contour, None, pan_temp, None

    cold_thresh = pan_temp - PANCAKE_COLD_DELTA
    cold_inside = np.zeros((h, w), dtype=np.uint8)
    cold_inside[(thermal_safe < cold_thresh) & (pan_filled > 0) & valid_mask] = 255

    warm_thresh = pan_temp - PANCAKE_COLD_DELTA / 2.0
    warm_pancake = np.zeros((h, w), dtype=np.uint8)
    warm_pancake[(thermal_safe < warm_thresh) & (thermal_safe > PAN_TEMP_THRESH - 30)
                 & (pan_filled > 0) & valid_mask] = 255
    cold_inside = cv2.bitwise_or(cold_inside, warm_pancake)

    cold_inside = cv2.morphologyEx(cold_inside, cv2.MORPH_CLOSE, kernel)
    cold_inside = cv2.morphologyEx(cold_inside, cv2.MORPH_OPEN, kernel)

    pancake_contours, _ = cv2.findContours(cold_inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not pancake_contours:
        return pan_contour, None, pan_temp, None

    pancake_contour = max(pancake_contours, key=cv2.contourArea)
    if cv2.contourArea(pancake_contour) < min_pc_area:
        return pan_contour, None, pan_temp, None

    pancake_filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(pancake_filled, [pancake_contour], -1, 255, cv2.FILLED)
    M = cv2.moments(pancake_contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cy = max(0, min(cy, h - 1))
        cx = max(0, min(cx, w - 1))
        pancake_temp = float(thermal_safe[cy, cx])
    else:
        pancake_pixels = thermal_safe[pancake_filled > 0]
        pancake_temp = float(np.mean(pancake_pixels)) if len(pancake_pixels) > 0 else None

    return pan_contour, pancake_contour, pan_temp, pancake_temp


# ── Serial protocol ──────────────────────────────────────────────────────────

def read_next_packet(ser):
    buf = b''
    while True:
        byte = ser.read(1)
        if len(byte) == 0:
            continue
        buf += byte
        if buf[-4:] == CAM_MAGIC:
            raw_len = ser.read(4)
            if len(raw_len) < 4:
                buf = b''; continue
            img_len = struct.unpack('<I', raw_len)[0]
            if img_len == 0 or img_len > 150_000:
                buf = b''; continue
            jpg = ser.read(img_len)
            if len(jpg) == img_len:
                return ('camera', jpg)
            buf = b''
        elif buf[-4:] == THERM_MAGIC:
            raw = ser.read(THERM_SIZE)
            if len(raw) == THERM_SIZE:
                temps = np.frombuffer(raw, dtype=np.float32).reshape(THERM_ROWS, THERM_COLS)
                return ('thermal', temps)
            buf = b''
        if len(buf) > 4:
            buf = buf[-4:]


def thermal_to_heatmap(thermal_grid, target_size=None):
    safe = np.nan_to_num(thermal_grid, nan=0.0)
    t_min, t_max = float(np.nanmin(thermal_grid)), float(np.nanmax(thermal_grid))
    if t_max - t_min < 0.1:
        t_max = t_min + 0.1
    normalized = ((safe - t_min) / (t_max - t_min) * 255).astype(np.uint8)
    if target_size is not None:
        normalized = cv2.resize(normalized, target_size, interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return heatmap, t_min, t_max


# ── Flask App ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pancake-monitor'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Shared state between capture thread and web clients
tracker = CookTracker()
state = {
    'rgb_frame': None,
    'thermal_frame': None,
    'cook_state': CookState.IDLE,
    'pan_temp': None,
    'pancake_temp': None,
    'progress': 0.0,
    'elapsed': '--:--',
    'total_elapsed': '--:--',
    'target_side1': 0,
    'target_side2': 0,
    'running': True,
}
state_lock = threading.Lock()


def encode_frame(frame):
    """Encode a BGR frame to base64 JPEG string."""
    if frame is None:
        return None
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode('utf-8')


def capture_loop():
    """Background thread: read serial, run detection, update shared state."""
    print("Loading YOLO ...")
    yolo_model = YOLO(os.path.join(os.path.dirname(__file__), '..',
                                    'runs', 'detect', 'pancake_pan2', 'weights', 'best.pt'))
    print("YOLO ready.")

    global tracker
    last_thermal_grid = None
    frame_count = 0
    pancake_present_frames = 0
    last_yolo_boxes = None

    calib_path = os.path.join(os.path.dirname(__file__), '..', 'homography', 'calibration.json')
    calibrator = HomographyCalibrator(thermal_shape=(24, 32), rgb_shape=(240, 320))

    if os.path.exists(calib_path):
        calibrator.load(calib_path)
        print(f"Loaded calibration: {calibrator.status_text}")

    print(f"Opening {PORT} at {BAUD} baud ...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=5)
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return

    print("Connected! Streaming to web ...\n")

    pan_emissivity = get_properties("pan")["emissivity"]
    pancake_emissivity = get_properties("default")["emissivity"]

    while state['running']:
        packet = read_next_packet(ser)
        if packet is None:
            continue

        ptype, data = packet
        now = time.time()

        if ptype == 'camera':
            img_np = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frame = cv2.resize(frame, (CAM_W, CAM_H))
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame_count += 1

            pan_c, pc_c, pan_t, pc_t = None, None, None, None
            use_homography = False

            if frame_count % CALIB_YOLO_EVERY_N == 0:
                yolo_results = yolo_model(frame, conf=CALIB_YOLO_CONF, verbose=False)
                yolo_boxes = yolo_results[0].boxes
                if len(yolo_boxes) > 0:
                    best_idx = int(yolo_boxes.conf.argmax())
                    last_yolo_boxes = tuple(map(int, yolo_boxes.xyxy[best_idx]))

            if last_yolo_boxes is not None and last_thermal_grid is not None:
                bx1, by1, bx2, by2 = last_yolo_boxes

                if not calibrator.calibrated:
                    if calibrator.add_correspondence(last_thermal_grid, last_yolo_boxes):
                        print(f"[calib] {calibrator.status_text}")
                    if calibrator.calibrated:
                        calibrator.save(calib_path)

                if calibrator.calibrated:
                    warped_grid = calibrator.warp_thermal(last_thermal_grid)
                    roi = warped_grid[by1:by2, bx1:bx2]
                    pan_c, pc_c, pan_t, pc_t = detect_pan_and_pancake(roi, warped=True)
                    if pan_c is not None:
                        pan_c = pan_c + np.array([bx1, by1])
                    if pc_c is not None:
                        pc_c = pc_c + np.array([bx1, by1])
                    use_homography = True
                else:
                    pan_c, pc_c, pan_t, pc_t = detect_pan_and_pancake(last_thermal_grid)

            if pan_t is not None:
                pan_t = estimate_temp(pan_t, pan_emissivity)
            if pc_t is not None:
                pc_t = estimate_temp(pc_t, pancake_emissivity)

            # State transitions + update shared state (all under lock)
            with state_lock:
                if tracker.state == CookState.IDLE:
                    if pc_c is not None and pan_t is not None and pc_t is not None:
                        pancake_present_frames += 1
                        if pancake_present_frames >= PANCAKE_DETECT_HYSTERESIS:
                            tracker.pancake_detected(pan_t, pc_t, now)
                            pancake_present_frames = 0
                    else:
                        pancake_present_frames = 0

                tracker.update(pc_t, pan_t, now)

            # Draw contours on RGB frame
            if pan_c is not None:
                cv2.drawContours(frame, [pan_c], -1, (0, 140, 255), 2, cv2.LINE_AA)
            if pc_c is not None:
                cv2.drawContours(frame, [pc_c], -1, (0, 255, 200), 2, cv2.LINE_AA)

            # Build thermal panel + crop both to thermal's valid region
            thermal_frame = None
            display_rgb = frame
            if last_thermal_grid is not None:
                if calibrator.calibrated:
                    warped = calibrator.warp_thermal(last_thermal_grid)
                    # Find bounding box of valid (non-NaN) thermal pixels
                    valid = ~np.isnan(warped)
                    rows = np.any(valid, axis=1)
                    cols = np.any(valid, axis=0)
                    if rows.any() and cols.any():
                        r0, r1 = np.where(rows)[0][[0, -1]]
                        c0, c1 = np.where(cols)[0][[0, -1]]
                        # Add small padding
                        pad = 5
                        r0 = max(0, r0 - pad)
                        c0 = max(0, c0 - pad)
                        r1 = min(warped.shape[0] - 1, r1 + pad)
                        c1 = min(warped.shape[1] - 1, c1 + pad)
                        # Crop both to the same region
                        cropped_warped = warped[r0:r1+1, c0:c1+1]
                        thermal_frame, _, _ = thermal_to_heatmap(cropped_warped)
                        display_rgb = frame[r0:r1+1, c0:c1+1].copy()
                        # Offset contours for drawing on cropped thermal
                        offset = np.array([c0, r0])
                        if pan_c is not None:
                            cv2.drawContours(thermal_frame, [pan_c - offset], -1, (255, 255, 255), 2, cv2.LINE_AA)
                        if pc_c is not None:
                            cv2.drawContours(thermal_frame, [pc_c - offset], -1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        thermal_frame, _, _ = thermal_to_heatmap(warped)
                else:
                    thermal_frame, _, _ = thermal_to_heatmap(last_thermal_grid, (320, 240))

            # Build snapshot under lock
            with state_lock:
                state['rgb_frame'] = encode_frame(display_rgb)
                state['thermal_frame'] = encode_frame(thermal_frame)
                state['cook_state'] = tracker.state
                state['pan_temp'] = round(pan_t, 1) if pan_t else None
                state['pancake_temp'] = round(pc_t, 1) if pc_t else None
                state['progress'] = round(tracker.progress(now) * 100, 1)
                state['elapsed'] = tracker.elapsed_str(now)
                state['total_elapsed'] = tracker.total_elapsed_str(now)
                state['target_side1'] = tracker.target_side1
                state['target_side2'] = tracker.target_side2
                sp = tracker.smoothed_pan_temp()
                spc = tracker.smoothed_pancake_temp()
                state['smoothed_pan'] = round(sp, 1) if sp else None
                state['smoothed_pancake'] = round(spc, 1) if spc else None

            # Emit to all connected clients
            socketio.emit('update', state)

        elif ptype == 'thermal':
            rotated = np.flip(data)
            rotated = np.fliplr(rotated)
            last_thermal_grid = rotated

    ser.close()
    print("Capture loop ended.")


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('flip')
def handle_flip():
    with state_lock:
        tracker.acknowledge_flip()
    print("[web] Flip acknowledged")


@socketio.on('reset')
def handle_reset():
    with state_lock:
        tracker.reset()
    print("[web] Tracker reset")


def main():
    print("=" * 50)
    print("  PANCAKE MONITOR — Web App")
    print("=" * 50)
    print()

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    print("Starting web server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=4000, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
