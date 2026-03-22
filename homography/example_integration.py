"""
Example: how to integrate HomographyCalibrator into display_overlay.py or display_merged.py.

This file is NOT a runnable script — it shows the key changes you'd paste into
your capture loop.  Search for "# HOMOGRAPHY" to find the 4 insertion points.
"""

# ─── 1. Imports (add at top) ────────────────────────────────────────────────
# from homography import HomographyCalibrator

# ─── 2. Initialisation (before the while loop) ──────────────────────────────
#
# # For display_merged.py the RGB frame is 240×320 (rotated 90° CW)
# calibrator = HomographyCalibrator(thermal_shape=(24, 32), rgb_shape=(320, 240))
#
# # Optionally load a previous calibration:
# # calibrator.load("homography/calibration.json")

# ─── 3. Collect correspondences (inside the camera frame branch) ────────────
#
# After YOLO runs, pick the best detection of a hot object (e.g. the pan from
# your custom model) and feed it to the calibrator along with the current
# thermal grid.
#
# if last_thermal_grid is not None and last_results_list:
#     # Use detections from the custom model (index 0)
#     boxes = last_results_list[0][0].boxes
#     if len(boxes) > 0:
#         # Pick highest-confidence detection
#         best_idx = int(boxes.conf.argmax())
#         bbox = tuple(map(int, boxes.xyxy[best_idx]))
#         added = calibrator.add_correspondence(last_thermal_grid, bbox)
#         if added:
#             print(f"[calib] point accepted — {calibrator.status_text}")

# ─── 4. Use warped thermal for temperature queries ──────────────────────────
#
# Replace get_raw_temp_for_box() with calibrator-based lookup once calibrated:
#
# def get_raw_temp_for_box_calibrated(calibrator, thermal_grid, x1, y1, x2, y2):
#     if not calibrator.calibrated:
#         # Fall back to naive linear mapping until calibrated
#         return get_raw_temp_for_box(thermal_grid, x1, y1, x2, y2)
#     warped = calibrator.warp_thermal(thermal_grid)
#     return calibrator.get_temp_for_box(warped, x1, y1, x2, y2)

# ─── 5. HUD overlay (in the display section) ────────────────────────────────
#
# cv2.putText(frame, calibrator.status_text, (6, 36),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

# ─── 6. Save on quit (after the while loop) ─────────────────────────────────
#
# calibrator.save("homography/calibration.json")
