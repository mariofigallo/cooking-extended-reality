from ultralytics import YOLO

model = YOLO("yolov8s.pt")
for idx, name in model.names.items():
    print(f"{idx:>3}: {name}")
