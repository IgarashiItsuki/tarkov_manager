import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import faiss
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

# === 設定 ===
INPUT_IMAGE = "../examples/salewa_sample.png"  # 調べたい画像（例）
INDEX_PATH = "../data/faiss_index.index"
META_PATH = "../data/item_metadata.json"

# === モデル準備 ===
model = resnet50(weights="IMAGENET1K_V1")
model.eval()
model = torch.nn.Sequential(*list(model.children())[:-1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === 入力画像の読み込み・前処理 ===
image = Image.open(INPUT_IMAGE).convert("RGB")

# 🔧 ここで中央をトリミング
w, h = image.size
margin_w, margin_h = int(w * 0.15), int(h * 0.15)
image = image.crop((margin_w, margin_h, w - margin_w, h - margin_h))

# 前処理（リサイズ & Tensor変換）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0).to(device)

# 特徴量抽出
with torch.no_grad():
    feature = model(input_tensor).squeeze().cpu().numpy()
    feature = np.expand_dims(feature, axis=0).astype("float32")

# FAISSインデックスの読み込み
index = faiss.read_index(INDEX_PATH)

# メタ情報の読み込み
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# 検索実行（上位5件）
D, I = index.search(feature, k=5)

# 結果表示
print("🔍 類似アイテム：")
for idx in I[0]:
    item = metadata[idx]
    print(f"- ID: {item['id']}")
    print(f"  Image: {item['image_path']}")
