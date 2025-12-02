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
        (56, 150, 255),   # #3896ff
        (50, 219, 137),   # #32db89
        (255, 204, 0),    # #ffcc00
        (226, 70, 70),    # #e24646
        (255, 104, 38),   # #ff6826
        (67, 74, 247),    # #434af7
        (240, 99, 179),   # #f063b3
        (58, 232, 218),   # #3ae8da
        (160, 238, 25),   # #a0ee19
        (163, 177, 197),  # #a3b1c5
        (36, 36, 36),     # #242424

    ]

    img_pil = Image.fromarray(img_np[..., ::-1])
    draw = ImageDraw.Draw(img_pil)
    

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        color = COLORS[(i - 1) % len(COLORS)]
        # Конвертируем в PIL для текста (RGB)
        

        # Рисуем рамку OpenCV (BGR)
        draw.rounded_rectangle(
            [x1, y1, x2, y2],
            radius=7,
            outline=color,
            width=8
        )

    font = ImageFont.truetype("Montserrat-Bold.ttf", max(12, (x2-x1)//10))  # размер адаптивный


    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        color = COLORS[(i - 1) % len(COLORS)]
        text = str(i)        

        # Размер текста
        text_w, text_h = draw.textbbox((0,0), text, font=font)[2:]
        # Центр бокса
        box_cx = (x1 + x2) / 2
        box_cy = (y1 + y2) / 2

        # Положение текста с белым скруглённым прямоугольником
        padding = 12
        rect_x1 = box_cx - text_w/2 - padding
        rect_y1 = box_cy - text_h/2 - padding
        rect_x2 = box_cx + text_w/2 + padding
        rect_y2 = box_cy + text_h/2 + padding

        # Скругленный прямоугольник
        draw.rounded_rectangle(
            [rect_x1, rect_y1, rect_x2, rect_y2],
            radius=15,
            fill=(255,255,255)
        )

        # Текст по центру
        draw.text(
            (box_cx - text_w/2, box_cy - text_h/2 - 7),
            text,
            font=font,
            fill=color
        )

    return img_pil







