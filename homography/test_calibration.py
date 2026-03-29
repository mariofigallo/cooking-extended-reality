"""
test_calibration.py
-------------------
Live visual test for the homography calibration.

Shows 4 panels:
  ┌────────────────┬────────────────┐
  │  RGB camera    │  Thermal raw   │
  │  + corr. pts   │  (upscaled)    │
  ├────────────────┼────────────────┤
  │  Warped thermal│  Blended       │
  │  in RGB space  │  RGB + thermal │
  └────────────────┴────────────────┘

- Correspondence points drawn as colored circles connected by lines
- Warped thermal overlaid on RGB so you can see alignment
- If not calibrated yet, collects correspondences live via YOLO

Controls:
  Q — quit
  S — save calibration
  C — clear calibration and re-collect
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import serial
import struct
import numpy as np
import cv2
from ultralytics import YOLO
from homography import HomographyCalibrator

# ── Configuration ────────────────────────────────────────────────────────────
PORT = '/dev/cu.usbserial-210'
BAUD = 600000
YOLO_CONF = 0.4
YOLO_EVERY_N = 2

# Camera / thermal dimensions (after rotation)
CAM_W, CAM_H = 240, 320       # raw, before rotation
DISPLAY_W, DISPLAY_H = 320, 240  # after 90° CW rotation (width, height for cv2)
THERM_ROWS, THERM_COLS = 24, 32
THERM_SIZE = THERM_ROWS * THERM_COLS * 4

BLEND_ALPHA = 0.45  # thermal overlay opacity

CAM_MAGIC   = b'\xFF\xAA\xBB\xCC'
THERM_MAGIC = b'\xFF\xDD\xEE\x11'

# Panel size — each panel is this size
PANEL_W, PANEL_H = DISPLAY_W, DISPLAY_H  # match rotated camera


# ── Serial ───────────────────────────────────────────────────────────────────

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


# ── Visualization helpers ────────────────────────────────────────────────────

def thermal_to_heatmap(temps, target_size=None):
    """Convert temperature array to a BGR heatmap."""
    t_min, t_max = np.nanmin(temps), np.nanmax(temps)
    if t_max - t_min < 0.1:
        t_max = t_min + 0.1
    # Replace NaN with min for colormap
    safe = np.nan_to_num(temps, nan=t_min)
    normalized = ((safe - t_min) / (t_max - t_min) * 255).astype(np.uint8)
    if target_size is not None:
        normalized = cv2.resize(normalized, target_size, interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    # Black out NaN regions if warped
    if np.any(np.isnan(temps)):
        nan_mask = np.isnan(temps)
        if target_size is not None:
            nan_mask = cv2.resize(nan_mask.astype(np.uint8), target_size,
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
        heatmap[nan_mask] = 0
    return heatmap, t_min, t_max


def draw_correspondence_points(rgb_panel, thermal_panel, calibrator):
    """Draw correspondence points on both panels with matching colors + lines."""
    n = calibrator.num_points
    if n == 0:
        return

    # Generate distinct colors for each point
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

        # RGB-side point
        rx, ry = calibrator.pts_rgb[i]
        cv2.circle(rgb_panel, (int(rx), int(ry)), 6, color, 2, cv2.LINE_AA)
        cv2.circle(rgb_panel, (int(rx), int(ry)), 2, color, -1, cv2.LINE_AA)
        cv2.putText(rgb_panel, str(i), (int(rx) + 8, int(ry) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # Thermal-side point (scale from 32×24 to panel size)
        tx, ty = calibrator.pts_thermal[i]
        sx = int(tx * PANEL_W / THERM_COLS)
        sy = int(ty * PANEL_H / THERM_ROWS)
        cv2.circle(thermal_panel, (sx, sy), 6, color, 2, cv2.LINE_AA)
        cv2.circle(thermal_panel, (sx, sy), 2, color, -1, cv2.LINE_AA)
        cv2.putText(thermal_panel, str(i), (sx + 8, sy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def draw_reprojection(rgb_panel, calibrator):
    """Draw where thermal points land in RGB space via H (reprojection check)."""
    if not calibrator.calibrated or calibrator.num_points == 0:
        return

    src = np.array(calibrator.pts_thermal, dtype=np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, calibrator.H).reshape(-1, 2)

    for i in range(len(projected)):
        # Actual RGB point (green circle)
        rx, ry = calibrator.pts_rgb[i]
        cv2.circle(rgb_panel, (int(rx), int(ry)), 8, (0, 255, 0), 1, cv2.LINE_AA)

        # Reprojected thermal point (red X)
        px, py = projected[i]
        px, py = int(px), int(py)
        cv2.drawMarker(rgb_panel, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 10, 2, cv2.LINE_AA)

        # Line between actual and reprojected (shows error)
        cv2.line(rgb_panel, (int(rx), int(ry)), (px, py), (0, 0, 255), 1, cv2.LINE_AA)

        # Error in pixels
        err = np.sqrt((rx - px)**2 + (ry - py)**2)
        cv2.putText(rgb_panel, f"{err:.1f}px", (px + 10, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)


def label_panel(panel, text, position=(8, 20), color=(255, 255, 255)):
    """Put a label with dark background on a panel."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = position
    cv2.rectangle(panel, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), cv2.FILLED)
    cv2.putText(panel, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  HOMOGRAPHY CALIBRATION TEST")
    print("=" * 50)
    print()
    print("Controls:")
    print("  Q       — quit")
    print("  S       — save calibration")
    print("  C       — clear and re-collect")
    print("  LEFT/RIGHT — adjust rotation by 5°")
    print("  UP/DOWN    — adjust rotation by 1°")
    print("  W/A        — scale up/down by 0.05")
    print()

    calib_path = os.path.join(os.path.dirname(__file__), 'calibration.json')
    rotation_deg = 0.0
    scale_factor = 1.0

    calibrator = HomographyCalibrator(thermal_shape=(THERM_ROWS, THERM_COLS),
                                      rgb_shape=(PANEL_H, PANEL_W),
                                      rotation_deg=rotation_deg,
                                      scale_factor=scale_factor)

    if os.path.exists(calib_path):
        calibrator.load(calib_path)
        rotation_deg = calibrator.rotation_deg
        scale_factor = calibrator.scale_factor

    print("Loading YOLO ...")
    yolo_model = YOLO(os.path.join(os.path.dirname(__file__), '..',
                                    'runs', 'detect', 'pancake_pan2', 'weights', 'best.pt'))
    print("YOLO ready.")

    print(f"Opening {PORT} at {BAUD} baud ...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=5)
    except serial.SerialException as e:
        print(f"Error: {e}")
        return

    print("Connected!\n")

    last_rgb = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    last_thermal_grid = None
    frame_count = 0
    last_yolo_bbox = None

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
            frame = cv2.resize(frame, (CAM_W, CAM_H))
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame_count += 1
            last_rgb = frame.copy()

            # Run YOLO
            if frame_count % YOLO_EVERY_N == 0:
                results = yolo_model(frame, conf=YOLO_CONF, verbose=False)
                boxes = results[0].boxes
                if len(boxes) > 0:
                    best_idx = int(boxes.conf.argmax())
                    last_yolo_bbox = tuple(map(int, boxes.xyxy[best_idx]))
                else:
                    last_yolo_bbox = None

            # Collect correspondences
            if last_yolo_bbox is not None and last_thermal_grid is not None \
               and not calibrator.calibrated:
                if calibrator.add_correspondence(last_thermal_grid, last_yolo_bbox):
                    print(f"[calib] {calibrator.status_text}")

        elif ptype == 'thermal':
            rotated = np.flip(data)
            rotated = np.fliplr(rotated)
            last_thermal_grid = rotated

        # ── Build 4-panel display ────────────────────────────────────────

        # Panel 1: RGB + correspondence points + YOLO bbox
        p1 = last_rgb.copy()
        if last_yolo_bbox is not None:
            bx1, by1, bx2, by2 = last_yolo_bbox
            cv2.rectangle(p1, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        draw_correspondence_points(p1, p1, calibrator)  # rgb points on p1
        draw_reprojection(p1, calibrator)
        label_panel(p1, "RGB + Points + Reprojection", color=(0, 255, 0))

        # Panel 2: Raw thermal heatmap (upscaled) + thermal correspondence points
        if last_thermal_grid is not None:
            p2, t_min, t_max = thermal_to_heatmap(last_thermal_grid, (PANEL_W, PANEL_H))
            label_panel(p2, f"Thermal Raw ({t_min:.0f}-{t_max:.0f}C)", color=(0, 200, 255))
            # Draw thermal-side points
            for i in range(calibrator.num_points):
                hue = int(180 * i / max(calibrator.num_points, 1))
                color_hsv = np.uint8([[[hue, 255, 255]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
                tx, ty = calibrator.pts_thermal[i]
                sx = int(tx * PANEL_W / THERM_COLS)
                sy = int(ty * PANEL_H / THERM_ROWS)
                cv2.circle(p2, (sx, sy), 6, color, 2, cv2.LINE_AA)
                cv2.putText(p2, str(i), (sx + 8, sy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        else:
            p2 = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            label_panel(p2, "Thermal — waiting ...", color=(100, 100, 100))

        # Panel 3: Warped thermal heatmap (in RGB space)
        if calibrator.calibrated and last_thermal_grid is not None:
            warped = calibrator.warp_thermal(last_thermal_grid)
            p3, w_min, w_max = thermal_to_heatmap(warped)
            label_panel(p3, f"Warped Thermal ({w_min:.0f}-{w_max:.0f}C)", color=(255, 200, 0))
        else:
            p3 = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            label_panel(p3, "Warped — not calibrated", color=(100, 100, 100))

        # Panel 4: Blended RGB + warped thermal
        if calibrator.calibrated and last_thermal_grid is not None:
            warped = calibrator.warp_thermal(last_thermal_grid)
            heatmap, _, _ = thermal_to_heatmap(warped)
            # Only blend where thermal data exists (not NaN)
            nan_mask = np.isnan(warped)
            nan_mask_3ch = np.stack([nan_mask]*3, axis=-1)
            p4 = last_rgb.copy()
            blended = cv2.addWeighted(p4, 1.0 - BLEND_ALPHA, heatmap, BLEND_ALPHA, 0)
            # Keep original RGB where thermal is NaN
            p4 = np.where(nan_mask_3ch, p4, blended)
            label_panel(p4, f"Blended (alpha={BLEND_ALPHA:.0%})", color=(255, 255, 255))
        else:
            p4 = last_rgb.copy()
            label_panel(p4, "Blend — not calibrated", color=(100, 100, 100))

        # Status bar
        status = f"{calibrator.status_text}  |  rot={rotation_deg:.0f} scale={scale_factor:.2f}"
        label_panel(p1, status, position=(8, PANEL_H - 10), color=(0, 255, 255))

        # Assemble grid
        top_row = np.hstack((p1, p2))
        bot_row = np.hstack((p3, p4))
        display = np.vstack((top_row, bot_row))

        cv2.imshow("Homography Calibration Test", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting.")
            break
        elif key == ord('s'):
            calibrator.save(calib_path)
            print(f"Saved: {calibrator.status_text}")
        elif key == ord('c'):
            calibrator = HomographyCalibrator(thermal_shape=(THERM_ROWS, THERM_COLS),
                                              rgb_shape=(PANEL_H, PANEL_W),
                                              rotation_deg=rotation_deg,
                                              scale_factor=scale_factor)
            print(f"Cleared — re-collecting (rot={rotation_deg:.0f}°, scale={scale_factor:.2f})")
        elif key == 81 or key == 2:   # LEFT arrow
            rotation_deg -= 5
            calibrator.recalibrate(rotation_deg=rotation_deg)
            print(f"rot={rotation_deg:.0f}° scale={scale_factor:.2f} (re-collecting...)")
        elif key == 83 or key == 3:   # RIGHT arrow
            rotation_deg += 5
            calibrator.recalibrate(rotation_deg=rotation_deg)
            print(f"rot={rotation_deg:.0f}° scale={scale_factor:.2f} (re-collecting...)")
        elif key == 82 or key == 0:   # UP arrow
            rotation_deg += 1
            calibrator.recalibrate(rotation_deg=rotation_deg)
            print(f"rot={rotation_deg:.0f}° scale={scale_factor:.2f} (re-collecting...)")
        elif key == 84 or key == 1:   # DOWN arrow
            rotation_deg -= 1
            calibrator.recalibrate(rotation_deg=rotation_deg)
            print(f"rot={rotation_deg:.0f}° scale={scale_factor:.2f} (re-collecting...)")
        elif key == ord('w'):         # scale up
            scale_factor += 0.05
            calibrator.recalibrate(scale_factor=scale_factor)
            print(f"rot={rotation_deg:.0f}° scale={scale_factor:.2f} (re-collecting...)")
        elif key == ord('a'):         # scale down
            scale_factor = max(0.1, scale_factor - 0.05)
            calibrator.recalibrate(scale_factor=scale_factor)
            print(f"rot={rotation_deg:.0f}° scale={scale_factor:.2f} (re-collecting...)")

    ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
