# build_faiss_index.py
import os
import json
import numpy as np
import faiss
import torch
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# --- PATHS ---
DATA_DIR = "data"
CARDS_JSON = os.path.join(DATA_DIR, "cards.json")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")
CARDS_IMG_DIR = os.path.join(DATA_DIR, "cards")

# --- DEVICE ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚡ Using device: {device}")

# --- LOAD CARDS ---
with open(CARDS_JSON, "r", encoding="utf-8") as f:
    cards = json.load(f)

# --- LOAD CLIP ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP model loaded")

# --- UTILS ---
def preprocess_card(card_img: Image.Image) -> Image.Image:
    card_img = card_img.convert("RGB")
    return ImageOps.fit(card_img, (224, 224), Image.BICUBIC, centering=(0.5, 0.5))

# --- BUILD EMBEDDINGS ---
embeddings = []
id_map = []

for card in tqdm(cards, desc="Processing cards"):
    img_path = os.path.join(CARDS_IMG_DIR, card["image"])
    if not os.path.exists(img_path):
        continue
    img = Image.open(img_path).convert("RGB")
    img = preprocess_card(img)
    
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    vector = features[0].cpu().numpy()
    vector = vector / np.linalg.norm(vector)
    
    embeddings.append(vector)
    id_map.append(card["card_id"])

embeddings = np.array(embeddings).astype("float32")
np.save(EMBEDDINGS_FILE, embeddings)
with open(ID_MAP_FILE, "w", encoding="utf-8") as f:
    json.dump(id_map, f, ensure_ascii=False, indent=2)
print(f"✅ Saved embeddings ({embeddings.shape}) and ID map ({len(id_map)})")

# --- CREATE FAISS INDEX ---
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, FAISS_INDEX_FILE)
print(f"✅ FAISS index created with {index.ntotal} vectors, dim={index.d}")
