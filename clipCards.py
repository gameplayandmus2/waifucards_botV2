import os
import json
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import torch
import open_clip
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ------------------------
# Пути
# ------------------------
CARDS_JSON = "data/cards.json"
IMAGES_DIR = "data/cards"
FAISS_INDEX_FILE = "data/faiss.index"
ID_MAP_FILE = "data/id_map.json"
LOG_FILE = "data/missing_cards.log"

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# ------------------------
# Загрузка JSON
# ------------------------
with open(CARDS_JSON, "r", encoding="utf-8") as f:
    cards = json.load(f)

print(f"Всего карточек для обработки: {len(cards)}")

# ------------------------
# OpenCLIP
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚡ Using device: {device}")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-H-14", pretrained="laion2b_s32b_b79k"
)
model.to(device)
model.eval()

# ------------------------
# FAISS индекс (чистый)
# ------------------------
embedding_dim = model.visual.output_dim
index = faiss.IndexFlatIP(embedding_dim)
print("📌 Создан новый FAISS индекс")

# ------------------------
# Скачивание изображения с retry
# ------------------------
def download_image(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200 or len(r.content)<1000:
                continue
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return img
        except:
            time.sleep(1)
    return None

# ------------------------
# Обработка одной карточки
# ------------------------
def process_card(card):
    base_filename = f"{card['set']}_{card['rarity']}_{card['number']}"
    img_path = os.path.join(IMAGES_DIR, base_filename + ".png")

    # Используем существующую картинку, если есть
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            return "error", card["id"]

    # Скачиваем PNG
    else:
        img = download_image(card["image_url"])
        if img is None:
            # fallback WEBP
            webp_url = card["image_url"].replace("cards_png", "cards").replace(".png",".webp")
            img = download_image(webp_url)
            if img is None:
                return "missing", card["id"]
            else:
                # сохраняем как PNG
                img.save(img_path, "PNG")

        else:
            # сохраняем PNG
            img.save(img_path, "PNG")

    # Генерация эмбеддинга
    try:
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(img_tensor)
        emb /= emb.norm(dim=-1, keepdim=True)
        index.add(emb.cpu().numpy())
        return "done", card["id"]
    except Exception as e:
        return "error", card["id"]

# ------------------------
# Цвета для прогресса
# ------------------------
class bcolors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

# ------------------------
# Потоковая обработка
# ------------------------
missing_cards = []
done = 0
errors = 0

max_workers = 16
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_card, c): c for c in cards}
    for f in tqdm(as_completed(futures), total=len(futures), desc="Processing cards"):
        res = f.result()
        if res[0] == "done":
            done += 1
            tqdm.write(f"{bcolors.GREEN}✅ DONE: Card ID {res[1]}{bcolors.RESET}")
        elif res[0] == "error":
            errors += 1
            missing_cards.append(res[1])
            tqdm.write(f"{bcolors.RED}❌ ERROR: Card ID {res[1]}{bcolors.RESET}")
        elif res[0] == "missing":
            missing_cards.append(res[1])
            tqdm.write(f"{bcolors.RED}⚠️ MISSING: Card ID {res[1]}{bcolors.RESET}")

# ------------------------
# Сохраняем FAISS индекс
# ------------------------
faiss.write_index(index, FAISS_INDEX_FILE)
print(f"💾 FAISS индекс сохранён в {FAISS_INDEX_FILE}")

# ------------------------
# Сохраняем карту id -> индекс
# ------------------------
id_map = [card["id"] for card in cards if os.path.exists(os.path.join(IMAGES_DIR, f"{card['set']}_{card['rarity']}_{card['number']}.png"))]
with open(ID_MAP_FILE, "w", encoding="utf-8") as f:
    json.dump(id_map, f, ensure_ascii=False, indent=2)
print(f"💾 ID map сохранена в {ID_MAP_FILE}")

# ------------------------
# Лог недоступных карточек
# ------------------------
if missing_cards:
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        for mid in missing_cards:
            f.write(str(mid) + "\n")
    print(f"⚠️ Недоступные карточки сохранены в {LOG_FILE}")

# ------------------------
# Итог
# ------------------------
print(f"✅ Завершено. DONE: {done}, ERROR: {errors}, MISSING: {len(missing_cards)}")
