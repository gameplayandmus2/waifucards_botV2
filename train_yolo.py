from ultralytics import YOLO

# Загружаем модель
model = YOLO("yolov8s.pt")   # можно: yolov8n.pt, yolov8m.pt, yolov8l.pt

# Обучаем
model.train(
    data="custom_yolo/dataset/data.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    workers=4,
    device="cpu",
    patience=30,
    optimizer="AdamW",
    lr0=0.0008,
    dropout=0.05,
    mosaic=1.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4
)

