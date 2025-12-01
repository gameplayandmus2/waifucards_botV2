from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

yolo_model = YOLO("custom_yolo/best.pt")

def detect_all_cards_yolo(pil_img: Image.Image):
    img = np.array(pil_img.convert("RGB"))[..., ::-1]  # BGR

    results = yolo_model(img, verbose=False)
    boxes = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        boxes.append((x1, y1, x2, y2))

    return img, boxes


from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def draw_boxes_with_numbers(np_img, boxes):
    img_np = np_img.copy()

    # Яркие читаемые цвета (BGR для OpenCV)
    COLORS = [
        (0, 255, 0),     # зелёный
        (0, 128, 255),   # оранжево-голубой
        (255, 0, 0),     # синий
        (255, 0, 255),   # фиолетовый p.s. от twinZ'a нихуя не совпадвает, но хоть номер и рамка одного цвета
        (0, 255, 255),   # жёлтый
        (255, 255, 0),   # голубой
        (128, 0, 255),   # фиолетово-синий
        (0, 0, 255),     # красный
    ]

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        color = COLORS[(i - 1) % len(COLORS)]

        # Рисуем рамку OpenCV (BGR)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color[::-1], 3)

    # Конвертируем в PIL для текста (RGB)
    img_pil = Image.fromarray(img_np[..., ::-1])
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("arial.ttf", max(24, (x2-x1)//5))  # размер адаптивный

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        color = COLORS[(i - 1) % len(COLORS)]
        text = str(i)

        # Размер текста
        text_w, text_h = draw.textbbox((0,0), text, font=font)[2:]
        # Центр бокса
        box_cx = (x1 + x2) / 2
        box_cy = (y1 + y2) / 2

        # Положение текста с белым скруглённым прямоугольником
        padding = 6
        rect_x1 = box_cx - text_w/2 - padding
        rect_y1 = box_cy - text_h/2 - padding
        rect_x2 = box_cx + text_w/2 + padding
        rect_y2 = box_cy + text_h/2 + padding

        # Скругленный прямоугольник
        draw.rounded_rectangle(
            [rect_x1, rect_y1, rect_x2, rect_y2],
            radius=8,
            fill=(255,255,255)
        )

        # Текст по центру
        draw.text(
            (box_cx - text_w/2, box_cy - text_h/2),
            text,
            font=font,
            fill=color
        )

    return img_pil







