"""
Quick fine-tune on just the our_pan images (14 photos).
Freezes the backbone and only trains the detection head,
so it learns your specific pan without forgetting everything else.
"""

from ultralytics import YOLO

# Load the best model from previous training
model = YOLO("../runs/detect/pancake_pan2/weights/best.pt")

model.train(
    data="our_pan_only.yaml",
    epochs=30,
    imgsz=640,
    batch=14,               # all 14 images in one batch
    device="mps",
    project=".",
    name="pancake_pan_finetuned",
    freeze=22,              # freeze first 22 layers (backbone), only train detection head
    lr0=0.0005,             # lower learning rate to avoid catastrophic forgetting
    patience=10,
    amp=True,
    cos_lr=True,
)
