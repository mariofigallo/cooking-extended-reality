"""
Add our_pan images to the pancake_pan dataset and retrain.

These images are all "pan" (class 0). Since they have no bounding box annotations,
we use the previously trained model to auto-label them, then add to the training set.

After running this, run train_yolo.py again (it will resume from the best weights).
"""

import os
import shutil
import glob
import cv2
from ultralytics import YOLO

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUR_PAN_DIR = os.path.join(PROJECT_ROOT, "datasets", "our_pan")
OUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "our_pan_labeled")
TRAIN_IMAGES = os.path.join(OUT_DIR, "images")
TRAIN_LABELS = os.path.join(OUT_DIR, "labels")

# Use the best model from the previous training run to auto-label
BEST_MODEL = os.path.join(PROJECT_ROOT, "runs", "detect", "pancake_pan2", "weights", "best.pt")

PAN_CLASS_ID = 0  # "pan" is class 0 in our dataset


def auto_label_and_add():
    os.makedirs(TRAIN_IMAGES, exist_ok=True)
    os.makedirs(TRAIN_LABELS, exist_ok=True)
    model = YOLO(BEST_MODEL)

    images = glob.glob(os.path.join(OUR_PAN_DIR, "*.jpeg")) + \
             glob.glob(os.path.join(OUR_PAN_DIR, "*.jpg")) + \
             glob.glob(os.path.join(OUR_PAN_DIR, "*.png"))

    print(f"Found {len(images)} images in our_pan/")

    added = 0
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Skipping (can't read): {img_path}")
            continue

        h, w = img.shape[:2]

        # Run inference to get bounding boxes
        results = model(img, conf=0.3, verbose=False)
        boxes = results[0].boxes

        lines = []
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id != PAN_CLASS_ID:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{PAN_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # If model didn't detect pan, use full image as fallback (it's all pan)
        if not lines:
            lines.append(f"{PAN_CLASS_ID} 0.500000 0.500000 1.000000 1.000000")
            print(f"  No detection, using full-image box: {os.path.basename(img_path)}")

        # Copy image with clean filename
        clean_name = f"our_pan_{added:03d}"
        dst_img = os.path.join(TRAIN_IMAGES, f"{clean_name}.jpg")
        shutil.copy2(img_path, dst_img)

        # Write label
        dst_label = os.path.join(TRAIN_LABELS, f"{clean_name}.txt")
        with open(dst_label, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"  Added {clean_name} with {len(lines)} box(es)")
        added += 1

    print(f"\nAdded {added} images to training set")
    print(f"Total training images now: {len(os.listdir(TRAIN_IMAGES))}")


if __name__ == "__main__":
    auto_label_and_add()
