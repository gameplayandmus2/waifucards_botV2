# bot_ptb.py
import os
import json
import torch
import faiss
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO

from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

import open_clip
from yolo_detector import detect_card_yolo  # твой модуль YOLO
from dotenv import load_dotenv
load_dotenv()

# ------------------------
# Настройки и пути
# ------------------------
TOKEN = os.getenv("TELEGRAM_TOKEN")

DATA_DIR = "data"
CARDS_JSON = os.path.join(DATA_DIR, "cards.json")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")
CARDS_IMG_DIR = os.path.join(DATA_DIR, "cards")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚡ Using device: {device}")

# ------------------------
# Загрузка карточек
# ------------------------
with open(CARDS_JSON, "r", encoding="utf-8") as f:
    cards = json.load(f)

# Создаём быстрый доступ ID → объект карты
card_by_id = {c["id"]: c for c in cards}

# ------------------------
# Загрузка FAISS индекса
# ------------------------
if os.path.exists(FAISS_INDEX_FILE):
    index = faiss.read_index(FAISS_INDEX_FILE)
    print(f"✅ FAISS index loaded, {index.ntotal} vectors")
else:
    raise FileNotFoundError(f"{FAISS_INDEX_FILE} not found!")

# ------------------------
# Загрузка ID map
# ------------------------
with open(ID_MAP_FILE, "r", encoding="utf-8") as f:
    id_map = json.load(f)

# ------------------------
# Загрузка CLIP
# ------------------------
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-H-14", pretrained="laion2b_s32b_b79k"
)
model.to(device)
model.eval()

print("✅ CLIP model loaded")

# ------------------------
# Утилиты
# ------------------------
def preprocess_card(card_img: Image.Image) -> Image.Image:
    card_img = card_img.convert("RGB")
    return ImageOps.fit(card_img, (224, 224), Image.BICUBIC, centering=(0.5, 0.5))

def find_top_matches(image: Image.Image, top_k=3):
    # Препроцессинг
    img = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(img)

    # Нормализуем вектор
    query = emb.cpu().numpy().astype("float32")
    query /= (np.linalg.norm(query) + 1e-10)

    # Поиск через FAISS
    distances, indices = index.search(query, top_k)

    results = []
    for rank, faiss_idx in enumerate(indices[0]):
        if faiss_idx < 0:
            continue  # иногда FAISS отдаёт -1

        # получаем реальный id карточки
        real_id = id_map[faiss_idx]

        # берём объект карточки по её id
        card = card_by_id.get(real_id)
        if card is None:
            continue  # защита если что-то не сошлось

        score = float(distances[0][rank])
        results.append((card, score))

    return results



# ------------------------
# Хэндлеры
# ------------------------
async def safe_send_image(update, img_path, caption=None):
    if not os.path.exists(img_path):
        return await update.message.reply_text(f"❌ Файл не найден: `{img_path}`")

    # Проверка что PNG живой
    try:
        test_img = Image.open(img_path)
        test_img.verify()  # проверяет структуру файла
    except Exception as e:
        return await update.message.reply_text(
            f"❌ Повреждённое изображение (`{img_path}`):\n```\n{e}\n```",
            parse_mode="Markdown"
        )

    # Всё ок — отправляем
    with open(img_path, "rb") as f:
        return await update.message.reply_photo(
            photo=f,
            caption=caption,
            parse_mode="Markdown"
        )


# ---------- обработчик /start ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я бот, который распознаёт карточки с сайта waifucards.app.\n"
        "Присылай фото карточки, и я покажу самые похожие."
    )


# ---------- обработчик фото ----------
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        await update.message.reply_text("❌ Нет фотографии в сообщении.")
        return

    # ——— получение файла ———
    try:
        photo = update.message.photo[-1]
        bio = BytesIO()

        file_obj = await context.bot.get_file(photo.file_id)
        await file_obj.download_to_memory(out=bio)
        bio.seek(0)

        img = Image.open(bio).convert("RGB")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка чтения фото:\n```\n{e}\n```", parse_mode="Markdown")
        return

    # ——— детекция карты YOLO ———
    cropped_img = detect_card_yolo(img)

    if cropped_img is None:
        await update.message.reply_text("❌ Не найдено карточек на фото.")
        return

    # ——— отправляем вырезанную карту (в памяти) ———
    try:
        out_bio = BytesIO()
        cropped_img.save(out_bio, format="PNG")
        out_bio.seek(0)

        await update.message.reply_photo(
            photo=InputFile(out_bio, filename="card.png"),
            caption="🔍 Вот что я вырезал с фото:"
        )
    except Exception as e:
        await update.message.reply_text(f"⚠️ Не удалось отправить обрезанную карту:\n```\n{e}\n```", parse_mode="Markdown")

    # ——— поиск совпадений ———
    matches = find_top_matches(cropped_img, top_k=3)
    threshold = 0.75
    found = False

    for idx, (card, score) in enumerate(matches):
        if score < threshold:
            continue

        found = True

        price_info = (
            f"💰 Цена: `{card.get('price', {}).get('price', '–')}₽`\n"
            if "price" in card else ""
        )

        caption = (
            f"{idx + 1}⃣ *{card['title']}*\n"
            f"👾 Тайтл: `{card['series']}`\n"
            f"📦 Set: [{card['set']}](https://waifucards.app/set/{card['set']})\n"
            f"🌟 Rarity: `{card['rarity']}`\n"
            f"{price_info}"
            f"🔗 [Открыть на сайте](https://waifucards.app/cards?number={card['id']})\n"
            f"📈 Совпадение: `{round(score*100,2)}%`"
        )

        # путь: set_rarity_number.png
        img_path = os.path.join(
            CARDS_IMG_DIR,
            f"{card['set']}_{card['rarity']}_{card['number']}.png"
        )

        # ——— безопасная отправка карточки ———
        await safe_send_image(update, img_path, caption=caption)

    if not found:
        await update.message.reply_text(
            "❌ Не удалось точно распознать карту. Попробуй другое фото."
        )

# ------------------------
# Запуск бота
# ------------------------
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("✅ Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()
