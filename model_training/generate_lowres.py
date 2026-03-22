"""
Augment the pancake_pan dataset with ESP32-CAM-like degradations.

The ESP32-CAM produces images that are:
- Low resolution (320x240 upscaled back)
- Heavy JPEG compression artifacts
- Noisy (especially in low light)
- Washed out / low contrast
- Slight color shift (cheap sensor)
- Motion blur from slow shutter

This script creates degraded copies of every training image,
duplicating the corresponding label file (bounding boxes stay the same
since we don't crop or spatially transform).

Run this BEFORE train_yolo.py.
"""

import os
import sys
import random
import cv2
import numpy as np

# Usage: python generate_lowres.py [dataset_path]
# dataset_path can be:
#   - a dataset with train/images + train/labels (e.g. datasets/pancake_pan)
#   - a flat dataset with images/ + labels/  (e.g. datasets/our_pan_labeled)
# Defaults to datasets/pancake_pan if no argument given.

def resolve_paths(dataset_arg=None):
    if dataset_arg is None:
        dataset_arg = os.path.join(os.path.dirname(__file__), "..", "datasets", "pancake_pan")
    dataset = os.path.abspath(dataset_arg)

    # Support both train/images and flat images/ layouts
    if os.path.isdir(os.path.join(dataset, "train", "images")):
        return (os.path.join(dataset, "train", "images"),
                os.path.join(dataset, "train", "labels"))
    elif os.path.isdir(os.path.join(dataset, "images")):
        return (os.path.join(dataset, "images"),
                os.path.join(dataset, "labels"))
    else:
        print(f"Error: can't find images/ in {dataset}")
        sys.exit(1)


def esp32_degrade(img):
    """Apply random ESP32-CAM-like degradations to an image."""
    h, w = img.shape[:2]

    # 1. Downscale to 320x240 then back up (simulates low-res sensor)
    small = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. Heavy JPEG compression (ESP32 uses quality 10-30 typically)
    quality = random.randint(8, 30)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # 3. Gaussian noise (sensor noise)
    noise_sigma = random.uniform(10, 30)
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Reduce contrast / wash out (random)
    if random.random() < 0.6:
        alpha = random.uniform(0.5, 0.8)  # contrast
        beta = random.randint(20, 60)      # brightness boost
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 5. Slight color tint (cheap CMOS sensor color cast)
    if random.random() < 0.5:
        channel = random.randint(0, 2)
        shift = random.randint(-15, 15)
        img[:, :, channel] = np.clip(
            img[:, :, channel].astype(np.int16) + shift, 0, 255
        ).astype(np.uint8)

    # 6. Motion blur (occasional)
    if random.random() < 0.3:
        ksize = random.choice([3, 5, 7])
        kernel = np.zeros((ksize, ksize))
        kernel[ksize // 2, :] = 1.0 / ksize
        img = cv2.filter2D(img, -1, kernel)

    return img


def main():
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else None
    TRAIN_IMAGES, TRAIN_LABELS = resolve_paths(dataset_arg)
    print(f"Augmenting: {TRAIN_IMAGES}")

    images = [f for f in os.listdir(TRAIN_IMAGES)
              if f.endswith((".jpg", ".jpeg", ".png")) and "_esp32aug" not in f]
    print(f"Found {len(images)} original images (skipping existing augmentations)")

    created = 0
    for fname in images:
        img_path = os.path.join(TRAIN_IMAGES, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Create 2 degraded variants per image
        for i in range(2):
            degraded = esp32_degrade(img)

            base = os.path.splitext(fname)[0]
            new_name = f"{base}_esp32aug{i}"

            cv2.imwrite(os.path.join(TRAIN_IMAGES, f"{new_name}.jpg"), degraded)

            # Copy label file (boxes unchanged since no spatial transform)
            src_label = os.path.join(TRAIN_LABELS, f"{base}.txt")
            dst_label = os.path.join(TRAIN_LABELS, f"{new_name}.txt")
            if os.path.exists(src_label):
                with open(src_label) as f:
                    label_data = f.read()
                with open(dst_label, "w") as f:
                    f.write(label_data)

            created += 1

    print(f"Created {created} augmented images")
    print(f"Total training images now: {len(os.listdir(TRAIN_IMAGES))}")


if __name__ == "__main__":
    main()
