import os
import json
import requests
from tqdm import tqdm

# 保存先ディレクトリ
SAVE_DIR = "../item_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# 辞書ファイル（tarkov_items.json）を読み込む
with open("../tarkov_items.json", "r", encoding="utf-8") as f:
    items = json.load(f)

# ダウンロード処理
for item in tqdm(items, desc="Downloading item images"):
    item_id = item["id"]
    image_url = f"https://assets.tarkov.dev/{item_id}-512.webp"
    save_path = os.path.join(SAVE_DIR, f"{item_id}.webp")

    # すでに保存済みならスキップ
    if os.path.exists(save_path):
        continue

    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"❌ Failed: {item['name']} ({item_id})")
    except Exception as e:
        print(f"⚠️ Error: {item['name']} ({item_id}) -> {e}")
