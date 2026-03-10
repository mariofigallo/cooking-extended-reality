from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # downloads on first run
results = model("image.jpg")
results[0].show()  # display with annotations