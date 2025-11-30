# clip_index.py
import numpy as np
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from pathlib import Path

CLIP_MODEL = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(CLIP_MODEL)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

device = "cpu"

model = model.to(device)

# Загружаем индекс
index_path = Path("clip/index.faiss")
ids_path = Path("clip/ids.npy")

faiss_index = faiss.read_index(str(index_path))
card_ids = np.load(ids_path, allow_pickle=True)


def embed_image(pil_img: Image.Image):
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


def search_card(pil_img: Image.Image):
    query = embed_image(pil_img)

    D, I = faiss_index.search(query, k=1)

    dist = float(D[0][0])
    idx = int(I[0][0])
    card_id = card_ids[idx]

    return card_id, dist
