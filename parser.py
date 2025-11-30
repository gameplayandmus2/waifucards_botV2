import json
import os

# Пути
INPUT_JSON = "data/search.json"  # твой оригинальный JSON
OUTPUT_JSON = "data/cards.json"

os.makedirs("data", exist_ok=True)

# Читаем исходный JSON
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(len(raw_data["data"]))

cards_out = []

for card in raw_data["data"]:
    set_code = card.get("set", "").replace(" ", "")
    rarity = card.get("rarity", "").upper()  # SSR, SR, R
    number = card.get("number", "")

    image_url = f"https://waifucards.app/img/cards_png/{set_code}/{rarity}-{number}.png"

    card_data = {
        "id": card.get("id"),
        "title": card.get("title"),
        "character": card.get("character"),
        "series": card.get("series"),
        "set": set_code,
        "rarity": rarity,
        "number": number,
        "image_url": image_url
    }

    cards_out.append(card_data)

# Сохраняем готовый JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(cards_out, f, ensure_ascii=False, indent=2)

print(f"✅ Подготовлено {len(cards_out)} карточек, сохранено в {OUTPUT_JSON}")
