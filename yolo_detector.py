# yolo_detector.py
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

YOLO_MODEL_PATH = Path("custom_yolo/best.pt")

# Загружаем YOLOv8 модель
yolo_model = YOLO(str(YOLO_MODEL_PATH))


def detect_card_yolo(pil_img: Image.Image) -> Image.Image | None:
    # Convert PIL → NumPy BGR
    image = np.array(pil_img.convert("RGB"))[..., ::-1]

    # Run YOLOv8 inference
    results = yolo_model(image, verbose=False)

    # Если нет боксов — ничего не нашли
    if len(results[0].boxes) == 0:
        return None

    # Берём первый бокс (как ты делал раньше)
    box = results[0].boxes[0]

    # Получаем координаты
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    cropped = image[y1:y2, x1:x2]

    # ==== Поворот по линиям (как в старом коде) ====
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is not None:
        angles = []

        for rho, theta in lines[:, 0]:
            angle_deg = np.rad2deg(theta)

            # Вертикальные/горизонтальные линии
            if angle_deg < 10 or (80 < angle_deg < 100) or angle_deg > 170:
                if angle_deg > 90:
                    angle_deg -= 180
                angles.append(angle_deg)

        if angles:
            median_angle = np.median(angles)

            if abs(median_angle) > 2:  # наклон >2°
                h, w = cropped.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
                rotated = cv2.warpAffine(
                    cropped, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

    # Если поворота нет или он маленький
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
