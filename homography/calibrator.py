"""
HomographyCalibrator
--------------------
Online self-calibration for aligning an MLX90640 thermal sensor (32×24)
to an ESP32-CAM RGB stream.

Strategy:
  1. Fit an ellipse to the hot pan contour in thermal space → center + size.
  2. YOLO detects the pan in RGB → bounding box → center + size.
  3. Compute a 2D affine transform: translate + rotate + scale.
     Rotation is a fixed offset (configured once for the mount), not
     derived from the ellipse angle (which is ambiguous for circular pans).
  4. Accumulate transforms across frames and median-average for stability.
"""

import json
import numpy as np
import cv2
from pathlib import Path


class HomographyCalibrator:
    MIN_FRAMES = 3

    def __init__(self, thermal_shape=(24, 32), rgb_shape=(240, 320),
                 rotation_deg=0.0, scale_factor=1.0):
        """
        Parameters
        ----------
        thermal_shape  : (rows, cols) of the thermal grid
        rgb_shape      : (rows, cols) of the RGB frame
        rotation_deg   : fixed rotation offset in degrees from thermal → RGB
                         (positive = counter-clockwise). Tune this once for
                         your physical mount.
        scale_factor   : multiplier on the computed scale (< 1.0 shrinks the
                         warped thermal, > 1.0 enlarges it). Use to compensate
                         for YOLO bbox being larger than the actual pan.
        """
        self.thermal_shape = thermal_shape
        self.rgb_shape = rgb_shape
        self.rotation_deg = rotation_deg
        self.scale_factor = scale_factor

        self._affine_samples = []
        self.H = None
        self.calibrated = False

        # For visualization
        self.pts_thermal = []
        self.pts_rgb = []

    # ── Ellipse fitting ───────────────────────────────────────────────────

    def _find_thermal_ellipse(self, thermal_grid):
        """Find the hot pan contour and fit an ellipse. Returns ellipse or None."""
        t_min, t_max = float(thermal_grid.min()), float(thermal_grid.max())
        if t_max - t_min < 2.0:
            return None

        thresh = t_min + 0.6 * (t_max - t_min)
        mask = (thermal_grid >= thresh).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 5 or len(largest) < 5:
            return None

        return cv2.fitEllipse(largest)

    # ── Affine computation ────────────────────────────────────────────────

    def _compute_affine(self, thermal_ellipse, bbox_xyxy):
        """
        Compute a 2×3 affine mapping thermal coords → RGB coords.

        Uses the thermal ellipse center/size, YOLO bbox center/size,
        and the fixed rotation_deg offset.
        """
        (tcx, tcy), (tw, th), _ = thermal_ellipse
        # Use the average radius (pan is circular)
        t_radius = max((tw + th) / 4.0, 0.1)

        x1, y1, x2, y2 = bbox_xyxy
        rcx = (x1 + x2) / 2.0
        rcy = (y1 + y2) / 2.0
        # Use average of bbox half-dims as RGB radius
        r_radius = max(((x2 - x1) + (y2 - y1)) / 4.0, 0.1)

        scale = (r_radius / t_radius) * self.scale_factor

        rad = np.deg2rad(self.rotation_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        # M = T_rgb @ S @ R @ T_thermal_inv
        T1 = np.array([[1, 0, -tcx],
                        [0, 1, -tcy],
                        [0, 0,    1]], dtype=np.float64)

        R = np.array([[cos_a, -sin_a, 0],
                       [sin_a,  cos_a, 0],
                       [    0,      0, 1]], dtype=np.float64)

        S = np.array([[scale, 0, 0],
                       [0, scale, 0],
                       [0,     0, 1]], dtype=np.float64)

        T2 = np.array([[1, 0, rcx],
                        [0, 1, rcy],
                        [0, 0,   1]], dtype=np.float64)

        M = T2 @ S @ R @ T1
        return M[:2, :]

    # ── Correspondence collection ─────────────────────────────────────────

    def add_correspondence(self, thermal_grid, bbox_xyxy):
        """Compute affine from current frame and accumulate."""
        thermal_ellipse = self._find_thermal_ellipse(thermal_grid)
        if thermal_ellipse is None:
            return False

        affine = self._compute_affine(thermal_ellipse, bbox_xyxy)
        if affine is None:
            return False

        self._affine_samples.append(affine)

        # Store centers for visualization
        (tcx, tcy), _, _ = thermal_ellipse
        x1, y1, x2, y2 = bbox_xyxy
        self.pts_thermal.append([tcx, tcy])
        self.pts_rgb.append([(x1+x2)/2.0, (y1+y2)/2.0])

        if len(self._affine_samples) >= self.MIN_FRAMES:
            self._solve()

        return True

    def _solve(self):
        stacked = np.array(self._affine_samples)
        avg_affine = np.median(stacked, axis=0)
        self.H = np.vstack([avg_affine, [0, 0, 1]])
        self.calibrated = True
        print(f"[HomographyCalibrator] Calibrated from {len(self._affine_samples)} frames "
              f"(rotation={self.rotation_deg:.1f}°, scale={self.scale_factor:.2f})")

    def recalibrate(self, rotation_deg=None, scale_factor=None):
        """Update rotation/scale and clear so it re-collects."""
        if rotation_deg is not None:
            self.rotation_deg = rotation_deg
        if scale_factor is not None:
            self.scale_factor = scale_factor
        self._affine_samples.clear()
        self.pts_thermal.clear()
        self.pts_rgb.clear()
        self.H = None
        self.calibrated = False

    # ── Warping ───────────────────────────────────────────────────────────

    def warp_thermal(self, thermal_grid):
        """Warp thermal grid into RGB space. Returns (rgb_h, rgb_w) float array with NaN borders."""
        if not self.calibrated:
            return None

        rgb_h, rgb_w = self.rgb_shape
        warped = cv2.warpPerspective(
            thermal_grid.astype(np.float32),
            self.H,
            (rgb_w, rgb_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float('nan'),
        )
        return warped

    def get_temp_for_box(self, warped_thermal, x1, y1, x2, y2):
        """Query max temperature inside a bounding box from warped thermal."""
        if warped_thermal is None:
            return None
        region = warped_thermal[y1:y2, x1:x2]
        if region.size == 0:
            return None
        val = np.nanmax(region)
        return None if np.isnan(val) else float(val)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path="homography/calibration.json"):
        data = {
            "pts_thermal": self.pts_thermal,
            "pts_rgb": self.pts_rgb,
            "thermal_shape": list(self.thermal_shape),
            "rgb_shape": list(self.rgb_shape),
            "rotation_deg": self.rotation_deg,
            "scale_factor": self.scale_factor,
            "affine_samples": [a.tolist() for a in self._affine_samples],
        }
        if self.H is not None:
            data["H"] = self.H.tolist()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[HomographyCalibrator] Saved to {path} "
              f"(rotation={self.rotation_deg:.1f}°, {len(self._affine_samples)} frames)")

    def load(self, path="homography/calibration.json"):
        with open(path) as f:
            data = json.load(f)
        self.pts_thermal = data.get("pts_thermal", [])
        self.pts_rgb = data.get("pts_rgb", [])
        self.thermal_shape = tuple(data["thermal_shape"])
        self.rgb_shape = tuple(data["rgb_shape"])
        self.rotation_deg = data.get("rotation_deg", 0.0)
        self.scale_factor = data.get("scale_factor", 1.0)
        self._affine_samples = [np.array(a) for a in data.get("affine_samples", [])]
        if "H" in data:
            self.H = np.array(data["H"], dtype=np.float64)
            self.calibrated = True
        print(f"[HomographyCalibrator] Loaded rotation={self.rotation_deg:.1f}°, "
              f"{len(self._affine_samples)} frames, calibrated={self.calibrated}")

    # ── Status ────────────────────────────────────────────────────────────

    @property
    def num_points(self):
        return len(self.pts_thermal)

    @property
    def status_text(self):
        if self.calibrated:
            return f"rot={self.rotation_deg:.0f} scale={self.scale_factor:.2f} ({len(self._affine_samples)}f)"
        return f"Collecting {len(self._affine_samples)}/{self.MIN_FRAMES} frames"
