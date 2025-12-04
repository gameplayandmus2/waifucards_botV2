# bot_ptb.py
import os
import json
import torch
import faiss
import numpy as np
import requests
from datetime import datetime
from PIL import Image, ImageOps
from io import BytesIO

from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

import open_clip
from yolo_detector import detect_all_cards_yolo, draw_boxes_with_numbers, should_show_quality_warning
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
savePic = os.getenv("savePicture")

DATA_DIR = "data"
CARDS_JSON = os.path.join(DATA_DIR, "cards.json")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")
CARDS_IMG_DIR = os.path.join(DATA_DIR, "cards")

use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
print(f"⚡ Using device: {device}" + (" (GPU enabled in .env)" if use_gpu else " (CPU mode)"))

# ------------------------  LOAD CARDS  ------------------------
with open(CARDS_JSON, "r", encoding="utf-8") as f:
    cards = json.load(f)

card_by_id = {c["id"]: c for c in cards}

# ------------------------  LOAD FAISS  ------------------------
index = faiss.read_index(FAISS_INDEX_FILE)
with open(ID_MAP_FILE, "r", encoding="utf-8") as f:
    id_map = json.load(f)

# ------------------------  LOAD CLIP  ------------------------
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-H-14", pretrained="laion2b_s32b_b79k"
)
model.to(device)
model.eval()

# ------------------------ UTILS ------------------------
def normalize_rarity(rarity):
    """Удаляет слеш из редкости. Например MR/199 -> MR199"""
    return rarity.replace("/", "")


def get_card_price(card_id):
    """Получает информацию о цене карточки с API"""
    try:
        response = requests.get(f"https://waifucards.app/price?id={card_id}", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None


def find_top_matches(image: Image.Image, top_k=3):
    img = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)

    query = emb.cpu().numpy().astype("float32")
    query /= np.linalg.norm(query) + 1e-10

    distances, indices = index.search(query, top_k)

    results = []
    for rank, faiss_idx in enumerate(indices[0]):
        if faiss_idx < 0 or faiss_idx >= len(id_map):
            continue
        real_id = id_map[faiss_idx]
        card = card_by_id.get(real_id)
        if card is None:
            continue
        score = float(distances[0][rank])
        results.append((card, score))

    return results

async def safe_send_image(message_obj, img_path, caption=None, reply_markup=None):
    if not os.path.exists(img_path):
        await message_obj.reply_text(f"❌ Файл не найден: `{img_path}`")
        return
    try:
        with Image.open(img_path) as im:
            im.verify()
    except Exception as e:
        await message_obj.reply_text(f"❌ Повреждённый PNG:\n{e}")
        return

    try:
        with open(img_path, "rb") as f:
            await message_obj.reply_photo(photo=f, caption=caption, parse_mode="Markdown", reply_markup=reply_markup)
    except Exception as e:
        await message_obj.reply_text(f"⚠️ Не удалось отправить изображение: {e}")

# ------------------------ COMMANDS ------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "👋 Привет! Я бот для распознавания карточек!\n\n"
        "📸 Просто отправь мне фото с одной или несколькими карточками — я:\n"
        "• Найду все карты на изображении\n"
        "• Дам тебе выбрать нужную\n"
        "• Покажу информацию о ней\n\n"
        "⚠️ *Дисклеймер:* отправленные изображения могут использоваться для улучшения "
        "работы модели распознавания (для обучения и повышения точности). "
        "Отправляя фото, вы соглашаетесь с этим.\n\n"
        "• Делайте фото под прямым углом (можно включить направляющие в камере)\n"
        "• Избегайте бликов и засветов. Чем лучше читаемость тем выше шанс определения\n"
        "• Вы можете прислать фото с несколькими карточками. Бот предложит выбрать нужную\n\n"
        "👇 Вот пример того, какое фото можно мне отправить:"
    )

    await update.message.reply_text(welcome_text, parse_mode="Markdown")

    # Отправка примера фото
    example_path = "data/how_to.jpg"

    try:
        with open(example_path, "rb") as f:
            await update.message.reply_photo(photo=f)
    except FileNotFoundError:
        await update.message.reply_text("⚠️ Ошибка: пример фото не найден в папке data/")

# ------------------------ PHOTO HANDLER ------------------------
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        await update.message.reply_text("❌ Нет фото в сообщении.")
        return

    photo = update.message.photo[-1]
    bio = BytesIO()
    try:
        file_obj = await context.bot.get_file(photo.file_id)
        await file_obj.download_to_memory(out=bio)
        bio.seek(0)
        pil_img = Image.open(bio).convert("RGB")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка чтения фото: {e}")
        return

    # -----------------------------
    # 📌 ЛОКАЛЬНОЕ СОХРАНЕНИЕ ФОТО
    # -----------------------------
    if savePic == "TRUE":
        os.makedirs("data/user_uploads", exist_ok=True)

        user_id = update.message.from_user.id
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        save_path = f"data/user_uploads/{timestamp}_user{user_id}.jpg"

        try:
            pil_img.save(save_path, "JPEG")
        except Exception as e:
            print(f"Ошибка сохранения изображения: {e}")
    # -----------------------------

    # YOLO detect all cards
    np_img, boxes, filter_info = detect_all_cards_yolo(pil_img)

    if not boxes:
        await update.message.reply_text("❌ YOLO не нашёл карточек.")
        return

    # save detections to user session
    context.user_data["np_img"] = np_img
    context.user_data["boxes"] = boxes

    # Если осталась только одна карта — сразу обрабатываем молча
    if len(boxes) == 1:
        await _process_card_by_index(update, context, 0)
        return

    # Если карт больше одной — показываем превью
    preview = draw_boxes_with_numbers(np_img, boxes)

    out = BytesIO()
    preview.save(out, format="JPEG")
    out.seek(0)

    caption = "Найденные карты пронумерованы сверху. Отправь номер карты для распознавания."

    # Проверяем качество распознавания
    if should_show_quality_warning(filter_info):
        caption += (
            "\n\n⚠️ Карточки трудно распознать. "
            "Попробуйте обрезать фото оставив только нужную карточку."
        )

    await update.message.reply_photo(
        photo=InputFile(out, filename="preview.jpg"),
        caption=caption
    )


# Вспомогательная функция для обработки карты по индексу
async def _process_card_by_index(update: Update, context: ContextTypes.DEFAULT_TYPE, card_idx: int):
    """Обрабатывает карту по индексу: кроп, поиск, вывод результатов"""
    np_img = context.user_data["np_img"]
    boxes = context.user_data["boxes"]

    if card_idx < 0 or card_idx >= len(boxes):
        await update.message.reply_text("❌ Неверный номер карты.")
        return

    # crop selected card
    x1, y1, x2, y2 = map(int, boxes[card_idx])
    h, w = np_img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        await update.message.reply_text("Ошибка кропа: некорректные координаты.")
        return

    card_np = np_img[y1:y2, x1:x2]
    card_pil = Image.fromarray(card_np[..., ::-1])

    # show cropped card
    buf = BytesIO()
    card_pil.save(buf, format="PNG")
    buf.seek(0)
    await update.message.reply_photo(
        photo=InputFile(buf, filename="crop.png"),
        caption="🔍 Распознаю эту карту..."
    )

    matches = find_top_matches(card_pil, top_k=3)
    threshold = 0.75
    found = False

    for idx, (card, score) in enumerate(matches):
        if score < threshold and idx != 0:
            continue

        found = True

        # Информация о цене карточки
        price_data = get_card_price(card['id'])

        # Нормализуем редкость для отображения
        rarity_normalized = normalize_rarity(card['rarity'])
        match_percent = round(score * 100, 2)

        # Извлекаем данные из JSON
        character = card.get('character', 'Unknown')
        title = card.get('title', 'Unknown')
        series = card.get('series', 'Unknown')
        card_set = card.get('set', 'Unknown')

        # Строим caption в Markdown формате
        caption = f"📈 Совпадение: `{match_percent}%`\n\n"

        # [RARITY-NUMBER] без ссылки, в backticks
        caption += f"🃏 *{rarity_normalized}-{card['number']}*\n"

        # [CHARACTER] ([TITLE]) с отдельными ссылками
        caption += f"👤 [{character}](https://waifucards.app/cards?character={character}) "
        caption += f"([{title}](https://waifucards.app/cards?title={title}))\n"

        # [SERIES SET] оба жирные, ссылка только на сет
        caption += f"📚 *{series}* [{card_set}](https://waifucards.app/set/{card_set})\n"

        caption += "\n"

        # Цена
        if price_data and price_data.get("price") is not None:
            price = price_data.get("price")
            count = price_data.get("count")
            price_type = price_data.get("type", "median")

            if price_type == "recommended":
                caption += f"💰 Рекомендованная стоимость: `{price}₽`\n"
            else:
                caption += f"💰 Средняя цена {price}₽\n"
                if count:
                    caption += f"📊 На основании {count} лотов\n"
        else:
            caption += "Данных о стоимости карты нет.\n"

        caption += "\n"

        # Лимитная редкость если есть
        if card.get("limit_range"):
            caption += f"Лимит: `*/{card['limit_range']}`\n"

        # Создаём кнопку "Открыть на сайте"
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔗 Открыть на сайте", url=f"https://waifucards.app/cards?number={card['id']}"),
            InlineKeyboardButton("🛒 Найти в продаже", url=f"https://waifucards.app/cards?number={card['id']}&list=sell")]
        ])

        # путь с нормализованной редкостью: cards_png/set/RARITY-number.png
        img_path = os.path.join(
            "../goddess-story/static/img/cards_png",
            card['set'],
            f"{rarity_normalized}-{card['number']}.png"
        )

        # Fallback на WEBP если PNG не найден
        if not os.path.exists(img_path):
            img_path = os.path.join(
                "../goddess-story/static/img/cards",
                card['set'],
                f"{rarity_normalized}-{card['number']}.webp"
            )

        await safe_send_image(update.message, img_path, caption=caption, reply_markup=keyboard)

    await update.message.reply_text (f"Пришлите следующий номер от 1 до {len(boxes)}, если вы хотите найти другую карту с вашего фото.")

    if not found:
        await update.message.reply_text(
            "❌ Не удалось точно распознать карту. Попробуй другое фото."
        )


# ------------------------ TEXT HANDLER ------------------------
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    if "np_img" not in context.user_data or "boxes" not in context.user_data:
        await update.message.reply_text("Сначала отправь фото с карточками.")
        return

    try:
        idx = int(text) - 1
    except ValueError:
        await update.message.reply_text("❌ Введи номер карты цифрой.")
        return

    await _process_card_by_index(update, context, idx)


# ------------------------ RUN ------------------------
def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    print("🚀 Bot is running")
    app.run_polling()

if __name__ == "__main__":
    main()
