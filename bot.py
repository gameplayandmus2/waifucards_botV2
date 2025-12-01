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
from yolo_detector import detect_all_cards_yolo, draw_boxes_with_numbers
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")

DATA_DIR = "data"
CARDS_JSON = os.path.join(DATA_DIR, "cards.json")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")
CARDS_IMG_DIR = os.path.join(DATA_DIR, "cards")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚡ Using device: {device}")

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

async def safe_send_image(message_obj, img_path, caption=None):
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
            await message_obj.reply_photo(photo=f, caption=caption, parse_mode="Markdown")
    except Exception as e:
        await message_obj.reply_text(f"⚠️ Не удалось отправить изображение: {e}")

# ------------------------ COMMANDS ------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Отправь фото с карточками ✨")

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

    # YOLO detect all cards
    np_img, boxes = detect_all_cards_yolo(pil_img)

    if not boxes:
        await update.message.reply_text("❌ YOLO не нашёл карточек.")
        return

    # save detections to user session
    context.user_data["np_img"] = np_img
    context.user_data["boxes"] = boxes

    # draw numbered boxes
    preview = draw_boxes_with_numbers(np_img, boxes)

    out = BytesIO()
    preview.save(out, format="JPEG")
    out.seek(0)

    await update.message.reply_photo(
        photo=InputFile(out, filename="preview.jpg"),
        caption="Найденные карты пронумерованы сверху. Отправь номер карты для распознавания."
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

    np_img = context.user_data["np_img"]
    boxes = context.user_data["boxes"]

    if idx < 0 or idx >= len(boxes):
        await update.message.reply_text("❌ Неверный номер карты.")
        return

    # crop selected card
    x1, y1, x2, y2 = map(int, boxes[idx])
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

        await safe_send_image(update.message, img_path, caption=caption)

    if not found:
        await update.message.reply_text(
            "❌ Не удалось точно распознать карту. Попробуй другое фото."
        )


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
