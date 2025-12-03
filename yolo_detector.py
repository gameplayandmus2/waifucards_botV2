from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

yolo_model = YOLO("custom_yolo/best.pt")

def should_show_quality_warning(filter_info):
    """
    Проверяет нужно ли выводить предупреждение о качестве распознавания.
    Возвращает True если отфильтровано 50% или более карт.
    """
    return filter_info["filtered_ratio"] >= 0.5


def _calculate_box_size(box):
    """Вычисляет нормализованный размер бокса (длинная, короткая сторона)"""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    # Нормализуем: длинная сторона первой
    return (max(w, h), min(w, h))


def _calculate_overlap_ratio(box1, box2):
    """Вычисляет процент перекрытия box1 от box2 (какой % box2 перекрывает box1)"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Пересечение
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    inter_area = (xi2 - xi1) * (yi2 - yi1)
    area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)

    return inter_area / area_box1 if area_box1 > 0 else 0.0


def _filter_cards(boxes):
    """
    Фильтрует боксы карт:
    1. [ПРЕДВАРИТЕЛЬНАЯ] Фильтрует ВСЕ карты с аномальным соотношением сторон (вытянутые более чем 1:1.8)
    2. Первые 2 оставшихся карты защищены от остальных фильтров
    3. Удаляет карты со слишком отклоняющимися размерами
    4. Удаляет карты с низким приоритетом, которые наезжают на другие на 60%+

    Возвращает: (отфильтрованные_боксы, информация_о_фильтрации)
    """
    original_count = len(boxes)

    if original_count == 0:
        return boxes, {"filtered_count": 0, "original_count": 0, "filtered_ratio": 0.0}

    # Вычислим нормализованные размеры и соотношение сторон
    sizes = [_calculate_box_size(box) for box in boxes]

    # Поддерживаем два типа карт:
    # - Обычная карта: 63х88мм → нормализованное соотношение 1.397:1
    # - Длинная карта (3 карты): 189х88мм → нормализованное соотношение 2.148:1
    NORMAL_ASPECT_MIN = 1.2    # Обычная карта: диапазон 1.2-1.6
    NORMAL_ASPECT_MAX = 1.6
    LONG_ASPECT_MIN = 2.0      # Длинная карта: диапазон 2.0-2.3
    LONG_ASPECT_MAX = 2.3

    # Этап 0 (ПРЕДВАРИТЕЛЬНАЯ): Отсеиваем ВСЕ карты с аномальным соотношением сторон
    valid_indices = set()
    for i in range(len(boxes)):
        long_side, short_side = sizes[i]
        aspect_ratio = long_side / short_side if short_side > 0 else 0

        # Карта должна соответствовать одному из двух типов
        is_normal = NORMAL_ASPECT_MIN <= aspect_ratio <= NORMAL_ASPECT_MAX
        is_long = LONG_ASPECT_MIN <= aspect_ratio <= LONG_ASPECT_MAX

        if is_normal or is_long:
            valid_indices.add(i)

    # Если осталось 2 или менее карт, возвращаем их
    if len(valid_indices) <= 2:
        filtered_count = original_count - len(valid_indices)
        filtered_ratio = filtered_count / original_count if original_count > 0 else 0.0
        return [boxes[i] for i in sorted(valid_indices)], {
            "filtered_count": filtered_count,
            "original_count": original_count,
            "filtered_ratio": filtered_ratio
        }

    # Определяем первые 2 оставшихся карты
    valid_sorted = sorted(valid_indices)
    first_idx = valid_sorted[0]
    second_idx = valid_sorted[1]

    # Эталон из первых 2 оставшихся карт: среднее значение каждой стороны
    ref_long = (sizes[first_idx][0] + sizes[second_idx][0]) / 2
    ref_short = (sizes[first_idx][1] + sizes[second_idx][1]) / 2

    # Допуск 20% от эталона
    tolerance = 0.2
    min_long = ref_long * (1 - tolerance)
    max_long = ref_long * (1 + tolerance)
    min_short = ref_short * (1 - tolerance)
    max_short = ref_short * (1 + tolerance)

    # Этап 1: Отсеиваем карты с отклоняющимися размерами (но НЕ первые 2)
    for i in list(valid_indices):
        if i in {first_idx, second_idx}:
            continue

        long_side, short_side = sizes[i]

        # Проверяем, находятся ли размеры в допустимом диапазоне
        if not (min_long <= long_side <= max_long and
                min_short <= short_side <= max_short):
            valid_indices.discard(i)

    # Этап 2: Удаляем карты с низким приоритетом, наезжающие на другие (но НЕ первые 2)
    for i in list(valid_indices):
        if i in {first_idx, second_idx}:
            continue

        # Проверяем пересечение с каждой более приоритетной картой
        for j in range(i):
            if j not in valid_indices:
                continue

            overlap_ratio = _calculate_overlap_ratio(boxes[j], boxes[i])

            # Если карта i наезжает на карту j более чем на 60%
            if overlap_ratio > 0.5:
                valid_indices.discard(i)
                break

    # Собираем статистику
    filtered_count = original_count - len(valid_indices)
    filtered_ratio = filtered_count / original_count if original_count > 0 else 0.0

    filter_info = {
        "filtered_count": filtered_count,
        "original_count": original_count,
        "filtered_ratio": filtered_ratio
    }

    # Возвращаем отфильтрованные боксы в исходном порядке
    return [boxes[i] for i in sorted(valid_indices)], filter_info


def detect_all_cards_yolo(pil_img: Image.Image):
    """
    Детектирует карты на изображении и применяет фильтрацию.

    Возвращает: (img, boxes, filter_info)
    - img: numpy массив изображения (BGR)
    - boxes: отфильтрованные боксы карт
    - filter_info: словарь с информацией о фильтрации:
        - filtered_count: количество отфильтрованных карт
        - original_count: исходное количество обнаруженных карт
        - filtered_ratio: доля отфильтрованных карт (0.0-1.0)
    """
    img = np.array(pil_img.convert("RGB"))[..., ::-1]  # BGR

    results = yolo_model(img, verbose=False)
    boxes = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        boxes.append((x1, y1, x2, y2))

    # Применяем фильтрацию
    boxes, filter_info = _filter_cards(boxes)

    return img, boxes, filter_info


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







