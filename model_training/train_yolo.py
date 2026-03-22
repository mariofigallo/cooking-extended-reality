from ultralytics import YOLO

# Load from previous best weights (fine-tune further with new data)
# Change back to "yolo11m.pt" to train from scratch
model = YOLO("../runs/detect/pancake_pan2/weights/best.pt")

# Fine-tune on pancake/pan dataset — optimized for M4 Pro
model.train(
    data="../datasets/pancake_pan/data.yaml",
    epochs=100,
    imgsz=640,
    batch=-1,               # auto-batch: finds max batch size that fits in memory
    device="mps",           # Apple Silicon GPU
    project=".",             # save runs inside model_training/
    name="pancake_pan",
    pretrained=True,
    patience=15,             # early stopping if val loss stalls for 15 epochs
    workers=8,               # M4 Pro has 12-14 CPU cores, use 8 for data loading
    amp=True,                # mixed precision — faster on Apple Silicon
    cos_lr=True,             # cosine learning rate schedule — smoother convergence
    close_mosaic=10,         # disable mosaic augmentation last 10 epochs for fine-tuning
)