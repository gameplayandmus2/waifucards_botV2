import os
import json
import requests
from tqdm import tqdm
from PIL import Image, ImageFilter
from io import BytesIO
import torch
import open_clip
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dotenv import load_dotenv

load_dotenv()

# ------------------------
# Пути
# ------------------------
BASE_IMG_DIR = os.getenv("BASE_IMG_DIR", "../goddess-story/static/img")
FAISS_INDEX_FILE = "data/faiss.index"
ID_MAP_FILE = "data/id_map.json"
CARDS_DATA_FILE = "data/cards.json"
LOG_FILE = "data/missing_cards.log"

os.makedirs("data", exist_ok=True)

# ------------------------
# Загрузка JSON из API с пагинацией
# ------------------------
print("📡 Загружаю данные карточек из API...")
cards = []
try:
    page = 1
    total_count = None

    while True:
        print(f"  📄 Загружаю страницу {page}...")
        response = requests.get(
            f"https://waifucards.app/v2/search?items=all&page={page}",
            timeout=30
        )
        response.raise_for_status()
        api_data = response.json()

        page_data = api_data.get("data", [])
        if not page_data:
            break

        cards.extend(page_data)

        # Получаем общее количество при первом запросе
        if total_count is None:
            total_count = int(api_data.get("count", 0))
            print(f"  📊 Всего карточек на сервере: {total_count}")

        print(f"  ✅ Загружено: {len(cards)}/{total_count}")

        # Проверяем достигли ли мы конца
        if len(cards) >= total_count:
            break

        page += 1
        time.sleep(0.5)  # Небольшая задержка между запросами

    print(f"✅ Всего загружено {len(cards)} карточек из API")

    # Сохраняем JSON локально для справки
    with open(CARDS_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)
except Exception as e:
    print(f"❌ Ошибка загрузки API: {e}")
    print("🔄 Пытаюсь загрузить локальный JSON...")
    try:
        with open(CARDS_DATA_FILE, "r", encoding="utf-8") as f:
            cards = json.load(f)
    except:
        print("❌ Не удалось загрузить локальный JSON")
        exit(1)

print(f"Всего карточек для обработки: {len(cards)}")

# ------------------------
# OpenCLIP
# ------------------------
use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
cuda_available = torch.cuda.is_available()
device = "cuda" if (use_gpu and cuda_available) else "cpu"
print(f"⚡ Debug: USE_GPU={use_gpu}, CUDA available={cuda_available}, PyTorch version={torch.__version__}")
print(f"⚡ Using device: {device}" + (" (GPU enabled in .env)" if use_gpu else " (CPU mode)"))

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
# Нормализация редкости
# ------------------------
def normalize_rarity(rarity):
    """Удаляет слеш из редкости. Например MR/199 -> MR199"""
    return rarity.replace("/", "")


# ------------------------
# Поиск изображения карточки
# ------------------------
def find_card_image(card):
    """Ищет изображение карточки в локальных папках. Приоритет: PNG > WEBP"""
    rarity_normalized = normalize_rarity(card["rarity"])
    card_set = card["set"]
    number = card["number"]

    # Приоритет 1: PNG
    png_path = os.path.join(BASE_IMG_DIR, "cards_png", card_set, f"{rarity_normalized}-{number}.png")
    if os.path.exists(png_path):
        try:
            img = Image.open(png_path)
            # Конвертируем палитровые изображения с прозрачностью в RGBA, потом в RGB
            if img.mode == "P" and "transparency" in img.info:
                img = img.convert("RGBA").convert("RGB")
            else:
                img = img.convert("RGB")
            return img, png_path
        except Exception as e:
            pass

    # Приоритет 2: WEBP
    webp_path = os.path.join(BASE_IMG_DIR, "cards", card_set, f"{rarity_normalized}-{number}.webp")
    if os.path.exists(webp_path):
        try:
            img = Image.open(webp_path)
            # Конвертируем палитровые изображения с прозрачностью в RGBA, потом в RGB
            if img.mode == "P" and "transparency" in img.info:
                img = img.convert("RGBA").convert("RGB")
            else:
                img = img.convert("RGB")
            return img, webp_path
        except Exception as e:
            pass

    return None, None


# ------------------------
# Размытие изображения для NSFW
# ------------------------
def blur_image(img, blur_radius=20):
    """Размывает изображение"""
    return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))


# ------------------------
# Обработка одной карточки
# ------------------------
def process_card(card):
    """Обрабатывает одну карточку: загрузка изображения и генерация эмбеддинга"""
    try:
        # Ищем изображение
        img, img_path = find_card_image(card)
        if img is None:
            return "missing", card["id"], None

        # Размываем если NSFW
        if card.get("nsfw", False):
            img = blur_image(img)

        # Генерация эмбеддинга
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(img_tensor)
        emb /= emb.norm(dim=-1, keepdim=True)

        # Возвращаем эмбеддинг вместо добавления в индекс
        return "done", card["id"], emb.cpu().numpy()
    except Exception as e:
        return "error", card["id"], None


# ------------------------
# Цвета для прогресса
# ------------------------
class bcolors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'


# ------------------------
# Потоковая обработка (параллельная)
# ВАЖНО: Эмбеддинги собираются в словарь по ID без добавления в FAISS!
# Добавление в FAISS происходит потом в правильном порядке
# ------------------------
print("\n🔄 Начинаю параллельную обработку карточек...")
embeddings = {}  # Словарь: card_id -> embedding
missing_cards = []
done = 0
errors = 0

max_workers = 16
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Создаём словарь: future -> card_id
    futures = {executor.submit(process_card, card): card["id"] for card in cards}

    for f in tqdm(as_completed(futures), total=len(futures), desc="Processing cards"):
        card_id = futures[f]
        res = f.result()

        if res[0] == "done":
            embeddings[card_id] = res[2]  # Сохраняем эмбеддинг по ID карточки
            done += 1
        elif res[0] == "error":
            errors += 1
            missing_cards.append(card_id)
            tqdm.write(f"{bcolors.RED}❌ ERROR: Card ID {card_id}{bcolors.RESET}")
        elif res[0] == "missing":
            missing_cards.append(card_id)
            tqdm.write(f"{bcolors.YELLOW}⚠️ MISSING: Card ID {card_id}{bcolors.RESET}")

# ------------------------
# Добавляем эмбеддинги в FAISS в правильном (исходном) порядке
# Используем реальный ID карточки из JSON для сопоставления
# Это устраняет race condition и связывает эмбеддинги по ID!
# ------------------------
print("\n💾 Добавляю эмбеддинги в FAISS индекс в правильном порядке...")
processed_ids = []
for card in cards:
    if card["id"] in embeddings:
        index.add(embeddings[card["id"]])
        processed_ids.append(card["id"])

# Сохраняем FAISS индекс
faiss.write_index(index, FAISS_INDEX_FILE)
print(f"💾 FAISS индекс сохранён в {FAISS_INDEX_FILE}")

# Сохраняем id_map в том же порядке (соответствует FAISS индексу)
id_map = processed_ids
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
print(f"\n✅ Завершено. DONE: {done}, ERROR: {errors}, MISSING: {len(missing_cards)}")
