"""
HomographyCalibrator
--------------------
Online self-calibration for aligning an MLX90640 thermal sensor (32×24)
to an ESP32-CAM RGB stream.

Strategy:
  1. Each frame, YOLO detects a hot object (e.g. pan) → bbox center in RGB space.
  2. The thermal hotspot centroid is found in thermal space.
  3. The (thermal_pt → rgb_pt) correspondence is stored if it's spatially
     diverse enough from existing points.
  4. Once ≥ MIN_POINTS diverse correspondences exist, cv2.findHomography
     with RANSAC computes H mapping thermal → RGB.
  5. After calibration, warp_thermal() projects the full thermal grid into
     RGB pixel space so temperature queries use aligned coordinates.
"""

import json
import numpy as np
import cv2
from pathlib import Path


class HomographyCalibrator:
    # Minimum correspondences before solving
    MIN_POINTS = 8
    # New point must be at least this far (in thermal pixels) from all existing points
    MIN_DIST_THERMAL = 3.0
    # And this far in RGB pixels
    MIN_DIST_RGB = 20.0

    def __init__(self, thermal_shape=(24, 32), rgb_shape=(240, 320)):
        """
        Parameters
        ----------
        thermal_shape : (rows, cols) of the thermal grid — (24, 32) for MLX90640
        rgb_shape     : (rows, cols) of the RGB frame the YOLO model runs on
        """
        self.thermal_shape = thermal_shape
        self.rgb_shape = rgb_shape

        # Accumulated correspondences: lists of [x, y] in each space
        self.pts_thermal = []   # source points (thermal pixel coords)
        self.pts_rgb = []       # destination points (RGB pixel coords)

        # 3×3 homography matrix (thermal → RGB), None until calibrated
        self.H = None
        self.calibrated = False

    # ── Correspondence collection ───────────────────────────────────────────

    def _is_diverse(self, thermal_pt, rgb_pt):
        """Return True if the new point is far enough from all existing ones."""
        if len(self.pts_thermal) == 0:
            return True
        t_arr = np.array(self.pts_thermal)
        r_arr = np.array(self.pts_rgb)
        t_dists = np.linalg.norm(t_arr - thermal_pt, axis=1)
        r_dists = np.linalg.norm(r_arr - rgb_pt, axis=1)
        return float(t_dists.min()) >= self.MIN_DIST_THERMAL and \
               float(r_dists.min()) >= self.MIN_DIST_RGB

    def thermal_hotspot_centroid(self, thermal_grid, top_k=5):
        """
        Find the centroid of the hottest region in the thermal grid.

        Uses the top_k hottest pixels and averages their coordinates to get a
        sub-pixel centroid that is more stable than a single argmax.

        Returns (x, y) in thermal pixel coordinates.
        """
        flat = thermal_grid.flatten()
        # Indices of top_k hottest pixels
        top_indices = flat.argsort()[-top_k:]
        rows, cols = np.unravel_index(top_indices, thermal_grid.shape)
        # Weight by temperature so hotter pixels pull the centroid more
        weights = flat[top_indices]
        weights = weights - weights.min() + 1e-6  # shift so all positive
        cx = float(np.average(cols, weights=weights))
        cy = float(np.average(rows, weights=weights))
        return np.array([cx, cy])

    def bbox_center(self, x1, y1, x2, y2):
        """Return center of a bounding box as [x, y]."""
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    def add_correspondence(self, thermal_grid, bbox_xyxy):
        """
        Try to add a new correspondence from the current frame.

        Parameters
        ----------
        thermal_grid : np.ndarray (24, 32) — raw temperature array (already
                       orientation-corrected to match camera view direction)
        bbox_xyxy    : tuple (x1, y1, x2, y2) — YOLO bounding box of a hot
                       object in RGB pixel coordinates

        Returns
        -------
        added : bool — True if a new point was accepted
        """
        thermal_pt = self.thermal_hotspot_centroid(thermal_grid)
        rgb_pt = self.bbox_center(*bbox_xyxy)

        if not self._is_diverse(thermal_pt, rgb_pt):
            return False

        self.pts_thermal.append(thermal_pt.tolist())
        self.pts_rgb.append(rgb_pt.tolist())

        # Re-solve whenever we have enough points
        if len(self.pts_thermal) >= self.MIN_POINTS:
            self._solve()

        return True

    # ── Homography computation ──────────────────────────────────────────────

    def _solve(self):
        src = np.array(self.pts_thermal, dtype=np.float64)
        dst = np.array(self.pts_rgb, dtype=np.float64)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is not None:
            self.H = H
            self.calibrated = True
            inliers = int(mask.sum()) if mask is not None else len(src)
            print(f"[HomographyCalibrator] Solved H with {inliers}/{len(src)} inliers "
                  f"({len(src)} total correspondences)")

    # ── Warping ─────────────────────────────────────────────────────────────

    def warp_thermal(self, thermal_grid):
        """
        Warp the 24×32 thermal grid into RGB pixel space.

        Returns an array of shape (rgb_h, rgb_w) where each pixel holds the
        interpolated temperature value at that RGB location.  Pixels outside
        the thermal FOV are filled with NaN.

        Returns None if not yet calibrated.
        """
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
        """
        Query the max temperature inside a bounding box from the warped
        thermal image.

        Parameters
        ----------
        warped_thermal : np.ndarray (rgb_h, rgb_w) from warp_thermal()
        x1, y1, x2, y2 : int — bounding box in RGB pixel coords

        Returns
        -------
        float or None — max temperature in the box region, ignoring NaN
        """
        if warped_thermal is None:
            return None
        region = warped_thermal[y1:y2, x1:x2]
        if region.size == 0:
            return None
        val = np.nanmax(region)
        return None if np.isnan(val) else float(val)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path="homography/calibration.json"):
        """Save correspondences and H to a JSON file."""
        data = {
            "pts_thermal": self.pts_thermal,
            "pts_rgb": self.pts_rgb,
            "thermal_shape": list(self.thermal_shape),
            "rgb_shape": list(self.rgb_shape),
        }
        if self.H is not None:
            data["H"] = self.H.tolist()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[HomographyCalibrator] Saved to {path} ({len(self.pts_thermal)} points)")

    def load(self, path="homography/calibration.json"):
        """Load a previous calibration from disk."""
        with open(path) as f:
            data = json.load(f)
        self.pts_thermal = data["pts_thermal"]
        self.pts_rgb = data["pts_rgb"]
        self.thermal_shape = tuple(data["thermal_shape"])
        self.rgb_shape = tuple(data["rgb_shape"])
        if "H" in data:
            self.H = np.array(data["H"], dtype=np.float64)
            self.calibrated = True
        print(f"[HomographyCalibrator] Loaded {len(self.pts_thermal)} points, "
              f"calibrated={self.calibrated}")

    # ── Status ──────────────────────────────────────────────────────────────

    @property
    def num_points(self):
        return len(self.pts_thermal)

    @property
    def status_text(self):
        if self.calibrated:
            return f"H calibrated ({self.num_points} pts)"
        return f"Collecting {self.num_points}/{self.MIN_POINTS} pts"
