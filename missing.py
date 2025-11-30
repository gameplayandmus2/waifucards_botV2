import os
import json
import requests
import numpy as np
from tqdm import tqdm
from PIL import Image
import faiss
import torch
from transformers import CLIPModel, CLIPProcessor

CARDS_JSON = "data/cards.json"
MISSING_LOG = "data/missing_cards.log"
OUTPUT_DIR = "data/cards"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# ---- CLIP ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚡ Using device: {device}")

model_name = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# ---- URL templates ----
URL_TEMPLATES = [
    "https://waifucards.app/img/cards_png/{set}/{rarity}-{number}.png",
    "https://waifucards.app/img/cards/{set}/{rarity}-{number}.webp",
    "https://waifucards.app/img/cards_jpg/{set}/{rarity}-{number}.jpg",
]

# ---- Load cards ----
with open(CARDS_JSON, "r", encoding="utf-8") as f:
    parsed = json.load(f)
    cards_list = parsed

cards = {c["id"]: c for c in cards_list}

# ---- Missing ids ----
with open(MISSING_LOG, "r", encoding="utf-8") as f:
    missing_ids = [line.strip() for line in f if line.strip()]

# ---- Prepare FAISS index ----
EMB_DIM = 768  # CLIP ViT-L/14 output dim

index_path = "data/faiss.index"
emb_path = "data/embeddings.npy"
map_path = "data/id_map.json"

if os.path.exists(index_path):
    print("📌 Загружаю существующий индекс...")
    index = faiss.read_index(index_path)
    embeddings = np.load(emb_path)
    with open(map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)
else:
    print("📌 Создаю новый FAISS индекс...")
    index = faiss.IndexFlatL2(EMB_DIM)
    embeddings = np.empty((0, EMB_DIM), dtype=np.float32)
    id_map = []

# ---- Functions ----

def try_download(url: str, filename: str) -> bool:
    """Попытка скачать файл."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(os.path.join(OUTPUT_DIR, filename), "wb") as f:
                f.write(r.content)
            return True
    except:
        pass
    return False


def embed_image(path: str):
    """CLIP embedding → numpy array."""
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


# ---- MAIN ----

still_missing = []

print(f"🔍 Обработка пропавших: {len(missing_ids)}")

for cid in tqdm(missing_ids, desc="Processing missing cards"):
    try:
        cid_int = int(cid)
    except:
        print(f"⚠️ Некорректный id: {cid}")
        continue

    if cid_int not in cards:
        print(f"⚠️ Нет в JSON: {cid_int}")
        still_missing.append(cid)
        continue

    card = cards[cid_int]
    set_code = card["set"]
    rarity = card["rarity"]
    number = card["number"]

    base = f"{set_code}_{rarity}_{number}"

    # Пытаемся скачать
    downloaded_path = None
    for template in URL_TEMPLATES:
        url = template.format(set=set_code, rarity=rarity, number=number)
        ext = os.path.splitext(url)[1]
        filename = base + ext

        if try_download(url, filename):
            downloaded_path = os.path.join(OUTPUT_DIR, filename)
            break

    if not downloaded_path:
        still_missing.append(cid)
        continue

    # ---- embedding ----
    try:
        emb = embed_image(downloaded_path)
        index.add(emb)
        embeddings = np.vstack([embeddings, emb])
        id_map.append(cid_int)
    except Exception as e:
        print(f"🔥 Ошибка эмбеддинга {cid_int}: {e}")
        still_missing.append(cid)
        continue


# ---- SAVE ----
print("💾 Сохраняю результат...")

faiss.write_index(index, index_path)
np.save(emb_path, embeddings)
with open(map_path, "w", encoding="utf-8") as f:
    json.dump(id_map, f, ensure_ascii=False, indent=2)

# ---- FINAL ----
print("\n🎉 Готово!")
print(f"✔️ Восстановлено и проэмбеддино: {len(missing_ids) - len(still_missing)}")
print(f"❌ Всё ещё пропавших: {len(still_missing)}")

if still_missing:
    with open("missing_cards_final.log", "w", encoding="utf-8") as f:
        f.write("\n".join(still_missing))
    print("📄 missing_cards_final.log обновлён")
