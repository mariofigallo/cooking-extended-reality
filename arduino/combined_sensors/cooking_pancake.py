"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PANCAKE MONITOR — Thermal-guided cooking assistant                        ║
║                                                                            ║
║  Uses thermal camera heuristics (no ML) to detect pan + pancake,           ║
║  track surface temperature, and signal FLIP / DONE.                        ║
║                                                                            ║
║  Hardware: ESP32-CAM + MLX90640 thermal camera on serial                   ║
║  Display:  RGB camera view with thermal overlays + cook HUD                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import serial
import struct
import numpy as np
import cv2
from collections import deque
from ultralytics import YOLO
from homography import HomographyCalibrator
from thermal_services import estimate_temp, get_properties

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — tweak these freely                                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# ── Serial / Hardware ────────────────────────────────────────────────────────
PORT            = '/dev/cu.usbserial-210'
BAUD            = 600000

# ── Thermal Camera ───────────────────────────────────────────────────────────
THERM_ROWS      = 24
THERM_COLS      = 32
THERM_SIZE      = THERM_ROWS * THERM_COLS * 4   # 768 float32s

# ── Camera ───────────────────────────────────────────────────────────────────
CAM_W           = 240       # raw camera width  (before rotation)
CAM_H           = 320       # raw camera height (before rotation)
DISPLAY_W       = 240       # display width  after 90° CW rotation
DISPLAY_H       = 320       # display height after 90° CW rotation

# ── Detection Thresholds ─────────────────────────────────────────────────────
PAN_TEMP_THRESH       = 70.0    # °C — pixels hotter than this are "pan"
PANCAKE_COLD_DELTA    = 100.0    # °C — pancake is at least this much cooler than pan rim
MIN_PAN_AREA_PX       = 40      # minimum pixel area in 32x24 grid to count as pan
MIN_PANCAKE_AREA_PX   = 30      # minimum pixel area in 32x24 grid to count as pancake
PAN_CIRCULARITY_MIN   = 0.35    # minimum circularity for pan contour (0-1)

# ── Pancake Specs (1/4 cup batter) ───────────────────────────────────────────
BATTER_VOLUME_ML      = 59.0    # 1/4 cup ≈ 59 mL
EXPECTED_DIAMETER_CM  = 10.0    # typical spread diameter for 1/4 cup

# ── Cook Model — Empirical Lookup ────────────────────────────────────────────
#    Format: (pan_temp_low, pan_temp_high) → side_1_seconds, side_2_seconds
#    These are your calibration values — update after experiments!
COOK_TABLE = {
    # (pan_temp_low, pan_temp_high): (side1_sec, side2_sec)
    (100, 130): (240, 180),     # low heat:    ~4 min + ~3 min
    (130, 160): (180, 120),     # medium-low:  ~3 min + ~2 min
    (160, 190): (120,  90),     # medium:      ~2 min + ~1.5 min
    (190, 220): (80,   60),     # medium-high: ~1:20  + ~1 min
    (220, 280): (60,   45),     # high:        ~1 min + ~45 sec
}
DEFAULT_COOK_TIME = (150, 100)  # fallback if pan temp outside all ranges

# ── Flip / Done Triggers ─────────────────────────────────────────────────────
#    In addition to time-based progress, we watch for thermal signatures:
FLIP_TEMP_THRESHOLD   = 75.0    # °C — pancake surface temp suggesting side 1 is done
DONE_TEMP_THRESHOLD   = 80.0    # °C — pancake surface temp suggesting side 2 is done
TEMP_RISE_RATE_FLIP   = 0.3     # °C/sec — if rate of rise drops below this, bubbles done

# ── Smoothing ────────────────────────────────────────────────────────────────
TEMP_HISTORY_LEN      = 30      # frames of pancake temp history for smoothing
PAN_TEMP_HISTORY_LEN  = 20      # frames of pan temp history

# ── Homography Calibration ──────────────────────────────────────────────────
CALIB_YOLO_CONF       = 0.4          # YOLO confidence for calibration detections
CALIB_YOLO_EVERY_N    = 3            # run YOLO every Nth frame during calibration

# ── Display ──────────────────────────────────────────────────────────────────
PROGRESS_BAR_W        = 200
PROGRESS_BAR_H        = 24
HUD_FONT              = cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_SMALL        = cv2.FONT_HERSHEY_PLAIN
WINDOW_NAME           = "Pancake Monitor"

# ── Protocol ─────────────────────────────────────────────────────────────────
CAM_MAGIC   = b'\xFF\xAA\xBB\xCC'
THERM_MAGIC = b'\xFF\xDD\xEE\x11'


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  COOK STATE MACHINE                                                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class CookState:
    IDLE        = "WAITING"         # pan hot, no pancake detected
    SIDE_1      = "SIDE 1"          # cooking first side
    FLIP_NOW    = ">>> FLIP <<<"    # signal to flip
    SIDE_2      = "SIDE 2"          # cooking second side
    DONE        = "▶ DONE ◀"       # remove from pan


class CookTracker:
    """Tracks the full cook lifecycle of one pancake."""

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

        self.flash_timer = 0       # for animation effects
        self.flip_acknowledged = False

    def lookup_cook_times(self, pan_temp_c):
        """Find empirical cook times for the given pan temperature."""
        for (lo, hi), (s1, s2) in COOK_TABLE.items():
            if lo <= pan_temp_c < hi:
                return s1, s2
        return DEFAULT_COOK_TIME

    def pancake_detected(self, pan_temp, pancake_temp, timestamp):
        """Called when a pancake blob first appears in the pan."""
        if self.state != CookState.IDLE:
            return

        self.pan_temp_at_start = pan_temp
        self.target_side1, self.target_side2 = self.lookup_cook_times(pan_temp)
        self.side1_start = timestamp
        self.state = CookState.SIDE_1
        self.pancake_temp_history.clear()
        print(f"\n🥞 PANCAKE DETECTED! Pan @ {pan_temp:.0f}°C")
        print(f"   Cook plan: Side 1 = {self.target_side1}s, Side 2 = {self.target_side2}s")

    def update(self, pancake_temp, pan_temp, timestamp):
        """Called every frame with current readings."""
        if pancake_temp is not None:
            self.pancake_temp_history.append((timestamp, pancake_temp))
        if pan_temp is not None:
            self.pan_temp_history.append((timestamp, pan_temp))

        self.flash_timer = timestamp

        if self.state == CookState.SIDE_1:
            elapsed = timestamp - self.side1_start
            progress = min(1.0, elapsed / self.target_side1) if self.target_side1 > 0 else 0

            # Check thermal flip signal: temp plateau or threshold
            rate = self._temp_rise_rate()
            temp_ready = (pancake_temp is not None and pancake_temp >= FLIP_TEMP_THRESHOLD)
            rate_ready = (rate is not None and rate < TEMP_RISE_RATE_FLIP and elapsed > 30)

            if progress >= 1.0 or (temp_ready and progress > 0.6) or (rate_ready and progress > 0.5):
                self.state = CookState.FLIP_NOW
                self.flip_time = timestamp
                self.flip_acknowledged = False
                print(f"\n🔄 FLIP NOW! (elapsed: {elapsed:.0f}s, surface: {pancake_temp:.1f}°C)")

        elif self.state == CookState.FLIP_NOW:
            # Stay in flip state for at least 3 seconds, then wait for temp drop (flip detected)
            if timestamp - self.flip_time > 3.0:
                if pancake_temp is not None and len(self.pancake_temp_history) > 5:
                    recent_temps = [t for _, t in list(self.pancake_temp_history)[-5:]]
                    # After flip, surface temp should drop significantly (raw batter side now up)
                    if self.flip_acknowledged or (max(recent_temps) - min(recent_temps) > 8):
                        self.side2_start = timestamp
                        self.state = CookState.SIDE_2
                        self.pancake_temp_history.clear()
                        print(f"\n🥞 SIDE 2 cooking...")

                # Auto-advance after 15s even if no temp drop detected
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
                total = timestamp - self.side1_start
                print(f"\n✅ DONE! Total cook time: {total:.0f}s")

        elif self.state == CookState.DONE:
            # After 10 seconds, reset for next pancake
            if timestamp - self.done_time > 10.0:
                self.reset()

    def acknowledge_flip(self):
        """User presses key to confirm flip happened."""
        if self.state == CookState.FLIP_NOW:
            self.flip_acknowledged = True

    def reset(self):
        """Reset for a new pancake."""
        self.__init__()

    def progress(self, timestamp):
        """Returns 0.0 – 1.0 progress for current side."""
        if self.state == CookState.SIDE_1 and self.side1_start:
            elapsed = timestamp - self.side1_start
            return min(1.0, elapsed / self.target_side1) if self.target_side1 > 0 else 0
        elif self.state == CookState.SIDE_2 and self.side2_start:
            elapsed = timestamp - self.side2_start
            return min(1.0, elapsed / self.target_side2) if self.target_side2 > 0 else 0
        elif self.state == CookState.FLIP_NOW:
            return 1.0
        elif self.state == CookState.DONE:
            return 1.0
        return 0.0

    def smoothed_pancake_temp(self):
        """Returns smoothed pancake temperature."""
        if len(self.pancake_temp_history) == 0:
            return None
        temps = [t for _, t in self.pancake_temp_history]
        # Weighted moving average (recent frames weighted more)
        if len(temps) >= 5:
            weights = np.linspace(0.5, 1.0, len(temps))
            return float(np.average(temps, weights=weights))
        return float(np.mean(temps))

    def smoothed_pan_temp(self):
        if len(self.pan_temp_history) == 0:
            return None
        return float(np.mean([t for _, t in self.pan_temp_history]))

    def _temp_rise_rate(self):
        """°C/sec over recent history."""
        if len(self.pancake_temp_history) < 10:
            return None
        recent = list(self.pancake_temp_history)[-10:]
        dt = recent[-1][0] - recent[0][0]
        if dt < 1.0:
            return None
        dtemp = recent[-1][1] - recent[0][1]
        return dtemp / dt

    def elapsed_str(self, timestamp):
        """Human-readable elapsed time for current side."""
        if self.state == CookState.SIDE_1 and self.side1_start:
            e = timestamp - self.side1_start
        elif self.state == CookState.SIDE_2 and self.side2_start:
            e = timestamp - self.side2_start
        elif self.state == CookState.FLIP_NOW and self.flip_time:
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


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  THERMAL DETECTION — pan & pancake finding via heuristics                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def detect_pan_and_pancake(thermal_grid, warped=False):
    """
    Detect pan (hot ring) and pancake (cooler blob inside) from a thermal image.

    When warped=False (default), thermal_grid is the raw 24×32 sensor output and
    contours are returned in thermal-pixel coords.

    When warped=True, thermal_grid is already warped into RGB display space
    (e.g. 320×240) by the homography calibrator.  Contours are returned directly
    in RGB-pixel coords — no further transformation needed for overlay.

    Returns:
        pan_contour, pancake_contour, pan_temp, pancake_temp, pan_mask, pancake_mask
    """
    if thermal_grid is None:
        return None, None, None, None, None, None

    h, w = thermal_grid.shape

    # Scale area thresholds: the raw thresholds are tuned for 24×32 = 768 pixels.
    # When running on a warped image the pixel count is much higher.
    if warped:
        scale = (h * w) / (THERM_ROWS * THERM_COLS)
        min_pan_area = int(MIN_PAN_AREA_PX * scale)
        min_pc_area = int(MIN_PANCAKE_AREA_PX * scale)
        # Larger kernel for the higher-res image
        ksize = max(3, int(3 * (h / THERM_ROWS)))
        if ksize % 2 == 0:
            ksize += 1
    else:
        min_pan_area = MIN_PAN_AREA_PX
        min_pc_area = MIN_PANCAKE_AREA_PX
        ksize = 3

    # Build a valid-pixel mask: for warped grids, NaN marks out-of-FOV regions
    if warped:
        valid_mask = ~np.isnan(thermal_grid)
        thermal_safe = np.nan_to_num(thermal_grid, nan=0.0)
    else:
        valid_mask = np.ones((h, w), dtype=bool)
        thermal_safe = thermal_grid

    # ── Step 1: Find pan pixels (hot threshold) ──────────────────────────
    pan_mask_raw = ((thermal_safe >= PAN_TEMP_THRESH) & valid_mask).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    pan_mask_clean = cv2.morphologyEx(pan_mask_raw, cv2.MORPH_CLOSE, kernel)
    pan_mask_clean = cv2.morphologyEx(pan_mask_clean, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(pan_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None, None, None

    pan_contour = max(contours, key=cv2.contourArea)
    pan_area = cv2.contourArea(pan_contour)
    if pan_area < min_pan_area:
        return None, None, None, None, None, None

    # Circularity check
    perimeter = cv2.arcLength(pan_contour, True)
    if perimeter > 0:
        circularity = 4 * math.pi * pan_area / (perimeter * perimeter)
        if circularity < PAN_CIRCULARITY_MIN:
            return pan_contour, None, None, None, pan_mask_clean, None

    # Pan rim temperature
    pan_filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(pan_filled, [pan_contour], -1, 255, cv2.FILLED)
    pan_rim_mask = cv2.bitwise_and(pan_mask_clean, pan_filled)
    pan_pixels = thermal_safe[pan_rim_mask > 0]
    pan_temp = float(np.max(pan_pixels)) if len(pan_pixels) > 0 else None

    # ── Step 2: Find pancake (cooler region inside pan) ──────────────────
    if pan_temp is None:
        return pan_contour, None, pan_temp, None, pan_mask_clean, None

    cold_thresh = pan_temp - PANCAKE_COLD_DELTA
    inside_pan = pan_filled.copy()

    cold_inside = np.zeros((h, w), dtype=np.uint8)
    cold_inside[(thermal_safe < cold_thresh) & (inside_pan > 0) & valid_mask] = 255

    # Also catch pancake that has warmed up during cooking but is still
    # distinctly cooler than the pan rim (at least PANCAKE_COLD_DELTA / 2)
    warm_pancake = np.zeros((h, w), dtype=np.uint8)
    warm_thresh = pan_temp - PANCAKE_COLD_DELTA / 2.0
    warm_pancake[(thermal_safe < warm_thresh) & (thermal_safe > PAN_TEMP_THRESH - 30)
                 & (inside_pan > 0) & valid_mask] = 255
    cold_inside = cv2.bitwise_or(cold_inside, warm_pancake)

    cold_inside = cv2.morphologyEx(cold_inside, cv2.MORPH_CLOSE, kernel)
    cold_inside = cv2.morphologyEx(cold_inside, cv2.MORPH_OPEN, kernel)

    pancake_contours, _ = cv2.findContours(cold_inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not pancake_contours:
        return pan_contour, None, pan_temp, None, pan_mask_clean, None

    pancake_contour = max(pancake_contours, key=cv2.contourArea)
    if cv2.contourArea(pancake_contour) < min_pc_area:
        return pan_contour, None, pan_temp, None, pan_mask_clean, None

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

    return pan_contour, pancake_contour, pan_temp, pancake_temp, pan_mask_clean, pancake_filled


def scale_contour_to_display(contour, from_w, from_h, to_w, to_h, homography=None):
    """Scale a contour from thermal grid coords to display coords.

    If a calibrated homography matrix is provided, uses cv2.perspectiveTransform
    for accurate projection. Otherwise falls back to naive linear scaling.
    """
    if contour is None:
        return None
    if homography is not None:
        pts = contour.reshape(-1, 1, 2).astype(np.float64)
        transformed = cv2.perspectiveTransform(pts, homography)
        return transformed.astype(np.int32)
    scaled = contour.astype(np.float64).copy()
    scaled[:, :, 0] = scaled[:, :, 0] * to_w / from_w
    scaled[:, :, 1] = scaled[:, :, 1] * to_h / from_h
    return scaled.astype(np.int32)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  HUD DRAWING                                                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def draw_hud(frame, tracker, pan_contour, pancake_contour, pan_temp, pancake_temp, timestamp, homography=None):
    """Draw all overlays onto the camera frame."""
    h, w = frame.shape[:2]

    # ── Contour overlays (scaled from thermal 32x24 to display) ──────────
    if pan_contour is not None:
        disp_pan = scale_contour_to_display(pan_contour, THERM_COLS, THERM_ROWS, w, h, homography)
        cv2.drawContours(frame, [disp_pan], -1, (0, 140, 255), 2, cv2.LINE_AA)

    if pancake_contour is not None:
        disp_pc = scale_contour_to_display(pancake_contour, THERM_COLS, THERM_ROWS, w, h, homography)
        cv2.drawContours(frame, [disp_pc], -1, (0, 255, 200), 2, cv2.LINE_AA)

    # ── Top bar: state + temperatures ────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # State label
    state = tracker.state
    if state == CookState.IDLE:
        state_color = (180, 180, 180)
    elif state == CookState.SIDE_1:
        state_color = (0, 200, 255)
    elif state == CookState.FLIP_NOW:
        # Flashing
        flash = int(time.time() * 4) % 2 == 0
        state_color = (0, 255, 255) if flash else (0, 100, 255)
    elif state == CookState.SIDE_2:
        state_color = (255, 200, 0)
    elif state == CookState.DONE:
        flash = int(time.time() * 3) % 2 == 0
        state_color = (0, 255, 100) if flash else (100, 255, 200)
    else:
        state_color = (255, 255, 255)

    cv2.putText(frame, state, (8, 20), HUD_FONT, 0.55, state_color, 2, cv2.LINE_AA)

    # Timer
    elapsed = tracker.total_elapsed_str(timestamp)
    cv2.putText(frame, elapsed, (w - 60, 20), HUD_FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Temperatures
    smt_pan = tracker.smoothed_pan_temp()
    smt_pc  = tracker.smoothed_pancake_temp()
    pan_str = f"Pan: {smt_pan:.0f}C" if smt_pan else "Pan: --"
    pc_str  = f"Pancake: {smt_pc:.0f}C" if smt_pc else "Pancake: --"
    cv2.putText(frame, pan_str, (8, 42), HUD_FONT_SMALL, 1.0, (0, 140, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, pc_str, (110, 42), HUD_FONT_SMALL, 1.0, (0, 255, 200), 1, cv2.LINE_AA)

    # ── Progress bar ─────────────────────────────────────────────────────
    progress = tracker.progress(timestamp)
    bar_x = (w - PROGRESS_BAR_W) // 2
    bar_y = h - 40

    # Background
    cv2.rectangle(frame, (bar_x - 2, bar_y - 2),
                  (bar_x + PROGRESS_BAR_W + 2, bar_y + PROGRESS_BAR_H + 2),
                  (40, 40, 40), cv2.FILLED)

    # Fill
    fill_w = int(progress * PROGRESS_BAR_W)
    if tracker.state == CookState.SIDE_1:
        bar_color = (0, 180, 255)       # orange
    elif tracker.state == CookState.SIDE_2:
        bar_color = (255, 180, 0)       # blue-ish
    elif tracker.state in (CookState.FLIP_NOW, CookState.DONE):
        bar_color = (0, 255, 100)       # green
    else:
        bar_color = (100, 100, 100)

    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + PROGRESS_BAR_H),
                      bar_color, cv2.FILLED)

    # Percentage text
    pct_str = f"{int(progress * 100)}%"
    (tw, _), _ = cv2.getTextSize(pct_str, HUD_FONT, 0.45, 1)
    cv2.putText(frame, pct_str, (bar_x + (PROGRESS_BAR_W - tw) // 2, bar_y + 17),
                HUD_FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Side label under bar
    if tracker.state in (CookState.SIDE_1, CookState.SIDE_2):
        side_label = f"{tracker.state} — {tracker.elapsed_str(timestamp)}"
        cv2.putText(frame, side_label, (bar_x, bar_y + PROGRESS_BAR_H + 16),
                    HUD_FONT_SMALL, 0.9, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Big FLIP / DONE overlay ──────────────────────────────────────────
    if tracker.state == CookState.FLIP_NOW:
        _draw_big_signal(frame, "FLIP", (0, 200, 255), timestamp)

    elif tracker.state == CookState.DONE:
        _draw_big_signal(frame, "DONE", (0, 255, 120), timestamp)

    return frame


def _draw_big_signal(frame, text, color, timestamp):
    """Draw a large pulsing text overlay for FLIP / DONE."""
    h, w = frame.shape[:2]

    # Pulsing scale
    pulse = 0.9 + 0.1 * math.sin(timestamp * 6)
    scale = 1.8 * pulse

    (tw, th), baseline = cv2.getTextSize(text, HUD_FONT, scale, 4)
    cx = (w - tw) // 2
    cy = (h + th) // 2 - 20

    # Semi-transparent backdrop
    overlay = frame.copy()
    pad = 30
    cv2.rectangle(overlay,
                  (cx - pad, cy - th - pad),
                  (cx + tw + pad, cy + baseline + pad),
                  (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Glow effect (thick dark outline + colored text)
    cv2.putText(frame, text, (cx, cy), HUD_FONT, scale, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(frame, text, (cx, cy), HUD_FONT, scale, color, 4, cv2.LINE_AA)

    # Instruction below
    if text == "FLIP":
        hint = "flip the pancake now! [press F]"
    else:
        hint = "remove from pan!"
    (hw, _), _ = cv2.getTextSize(hint, HUD_FONT, 0.45, 1)
    cv2.putText(frame, hint, ((w - hw) // 2, cy + 35),
                HUD_FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SERIAL PROTOCOL                                                         ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def read_next_packet(ser):
    """Scan for magic header, return ('camera', jpg) or ('thermal', temps)."""
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
            raw = ser.read(THERM_SIZE)
            if len(raw) == THERM_SIZE:
                temps = np.frombuffer(raw, dtype=np.float32).reshape(THERM_ROWS, THERM_COLS)
                return ('thermal', temps)
            buf = b''

        if len(buf) > 4:
            buf = buf[-4:]


def thermal_to_heatmap(thermal_grid, target_size=None):
    """Convert a temperature grid to a BGR heatmap image."""
    safe = np.nan_to_num(thermal_grid, nan=0.0)
    t_min, t_max = float(np.nanmin(thermal_grid)), float(np.nanmax(thermal_grid))
    if t_max - t_min < 0.1:
        t_max = t_min + 0.1
    normalized = ((safe - t_min) / (t_max - t_min) * 255).astype(np.uint8)
    if target_size is not None:
        normalized = cv2.resize(normalized, target_size, interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return heatmap, t_min, t_max


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN LOOP                                                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def main():
    print("=" * 60)
    print("  🥞 PANCAKE MONITOR")
    print("  Thermal-guided cooking assistant")
    print("=" * 60)
    print()
    print("Controls:")
    print("  Q — quit")
    print("  F — acknowledge flip (when FLIP signal shown)")
    print("  R — reset tracker (start fresh)")
    print()

    print(f"Opening {PORT} at {BAUD} baud ...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=5)
    except serial.SerialException as e:
        print(f"Error: {e}")
        return

    print("Connected! Waiting for frames ...\n")

    # ── Load YOLO (stays loaded — used for gating + calibration) ────────
    print("Loading YOLO ...")
    yolo_model = YOLO("../../runs/detect/pancake_pan2/weights/best.pt")
    print("YOLO ready.")

    tracker = CookTracker()

    # After 90° CW rotation, numpy shape is (DISPLAY_W, DISPLAY_H) = (240, 320)
    last_camera = np.zeros((DISPLAY_W, DISPLAY_H, 3), dtype=np.uint8)
    last_thermal_display = np.zeros((DISPLAY_W, DISPLAY_H, 3), dtype=np.uint8)
    last_thermal_grid = None   # raw 24x32 float array (rotated to match camera)
    frame_count = 0
    pancake_absent_frames = 0
    PANCAKE_DETECT_HYSTERESIS = 30  # ~3 seconds of continuous detection before triggering

    pancake_present_frames = 0
    last_yolo_boxes = None  # latest YOLO detections

    # ── Homography calibrator ───────────────────────────────────────────
    calib_path = os.path.join(os.path.dirname(__file__), '..', '..', 'homography', 'calibration.json')
    # After 90° CW rotation: numpy shape is (DISPLAY_W, DISPLAY_H) = (240, 320)
    calibrator = HomographyCalibrator(thermal_shape=(24, 32), rgb_shape=(DISPLAY_W, DISPLAY_H))

    if os.path.exists(calib_path):
        calibrator.load(calib_path)
        print(f"Loaded calibration: {calibrator.status_text}")
    else:
        print("No saved calibration — will auto-calibrate from YOLO detections.")

    while True:
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
            # Rotate 90° clockwise to match physical orientation
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame_count += 1

            # ── YOLO: detect pan in RGB ─────────────────────────────────────
            pan_c, pc_c, pan_t, pc_t = None, None, None, None
            use_homography = False

            if frame_count % CALIB_YOLO_EVERY_N == 0:
                yolo_results = yolo_model(frame, conf=CALIB_YOLO_CONF, verbose=False)
                yolo_boxes = yolo_results[0].boxes
                if len(yolo_boxes) > 0:
                    best_idx = int(yolo_boxes.conf.argmax())
                    last_yolo_boxes = tuple(map(int, yolo_boxes.xyxy[best_idx]))

            # ── Gate: only run thermal detection if YOLO sees a pan ──────
            if last_yolo_boxes is not None and last_thermal_grid is not None:
                bx1, by1, bx2, by2 = last_yolo_boxes

                # Homography: collect correspondences while calibrating
                if not calibrator.calibrated:
                    if calibrator.add_correspondence(last_thermal_grid, last_yolo_boxes):
                        print(f"[calib] {calibrator.status_text}")
                    if calibrator.calibrated:
                        calibrator.save(calib_path)
                        print("[calib] Calibration complete and saved.")

                # Run thermal detection inside the YOLO bbox region
                if calibrator.calibrated:
                    # Warp thermal into RGB space, crop to YOLO bbox
                    warped_grid = calibrator.warp_thermal(last_thermal_grid)
                    roi = warped_grid[by1:by2, bx1:bx2]
                    pan_c, pc_c, pan_t, pc_t, _, _ = detect_pan_and_pancake(roi, warped=True)
                    # Offset contours back to full-frame coords
                    if pan_c is not None:
                        pan_c = pan_c + np.array([bx1, by1])
                    if pc_c is not None:
                        pc_c = pc_c + np.array([bx1, by1])
                    use_homography = True
                else:
                    # Pre-calibration: detect on raw grid (less accurate)
                    pan_c, pc_c, pan_t, pc_t, _, _ = detect_pan_and_pancake(last_thermal_grid)

            # ── Emissivity correction ──────────────────────────────────
            pan_emissivity = get_properties("pan")["emissivity"]
            pancake_emissivity = get_properties("default")["emissivity"]
            if pan_t is not None:
                pan_t = estimate_temp(pan_t, pan_emissivity)
            if pc_t is not None:
                pc_t = estimate_temp(pc_t, pancake_emissivity)

            # ── State transitions ────────────────────────────────────────
            if tracker.state == CookState.IDLE:
                if pc_c is not None and pan_t is not None and pc_t is not None:
                    pancake_present_frames += 1
                    if pancake_present_frames >= PANCAKE_DETECT_HYSTERESIS:
                        tracker.pancake_detected(pan_t, pc_t, now)
                        pancake_present_frames = 0
                else:
                    pancake_present_frames = 0

            # Update tracker every frame
            tracker.update(pc_t, pan_t, now)

            # ── Draw HUD ─────────────────────────────────────────────────
            # When warped, contours are already in RGB coords — no transform needed
            H = None if use_homography else (calibrator.H if calibrator.calibrated else None)
            display = draw_hud(frame, tracker, pan_c, pc_c, pan_t, pc_t, now, H)
            last_camera = display

            # ── Build thermal panel with contour overlay ────────────────
            if last_thermal_grid is not None:
                # Use warped thermal if calibrated, else upscale raw grid
                if calibrator.calibrated:
                    warped_for_display = calibrator.warp_thermal(last_thermal_grid)
                    thermal_panel, t_min, t_max = thermal_to_heatmap(warped_for_display)
                else:
                    # cv2.resize takes (width, height); frame is (240 rows, 320 cols)
                    thermal_panel, t_min, t_max = thermal_to_heatmap(
                        last_thermal_grid, (DISPLAY_H, DISPLAY_W))

                frame_h, frame_w = display.shape[:2]  # actual numpy dims

                # Draw pan/pancake contours on the thermal panel
                if pan_c is not None:
                    if use_homography:
                        cv2.drawContours(thermal_panel, [pan_c], -1, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        disp_pan = scale_contour_to_display(pan_c, THERM_COLS, THERM_ROWS,
                                                             frame_w, frame_h, H)
                        cv2.drawContours(thermal_panel, [disp_pan], -1, (255, 255, 255), 2, cv2.LINE_AA)
                if pc_c is not None:
                    if use_homography:
                        cv2.drawContours(thermal_panel, [pc_c], -1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        disp_pc = scale_contour_to_display(pc_c, THERM_COLS, THERM_ROWS,
                                                            frame_w, frame_h, H)
                        cv2.drawContours(thermal_panel, [disp_pc], -1, (0, 255, 0), 2, cv2.LINE_AA)

                # Temperature labels on thermal panel
                cv2.putText(thermal_panel, f"{t_min:.0f}C - {t_max:.0f}C", (6, frame_h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                if pan_t is not None:
                    cv2.putText(thermal_panel, f"Pan: {pan_t:.0f}C", (6, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)
                if pc_t is not None:
                    cv2.putText(thermal_panel, f"Pancake: {pc_t:.0f}C", (6, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)

                last_thermal_display = thermal_panel
            else:
                last_thermal_display = np.zeros_like(display)

            # Side by side: camera+HUD on left, thermal+contours on right
            combined = np.hstack((display, last_thermal_display))
            cv2.imshow(WINDOW_NAME, combined)

        elif ptype == 'thermal':
            # Rotate 180° to align with camera orientation
            rotated = np.flip(data)
            rotated = np.fliplr(rotated)
            last_thermal_grid = rotated

        # ── Key handling ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting.")
            break
        elif key == ord('f'):
            tracker.acknowledge_flip()
            print("  [F] Flip acknowledged")
        elif key == ord('r'):
            tracker.reset()
            print("  [R] Tracker reset")

    ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()