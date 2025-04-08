# scripts/predict_combined.py

import os
import json
import torch
import numpy as np
import faiss
from PIL import Image
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.nn import functional as F

# ======================
# パスの設定
# ======================
MODEL_PATH = "../models/cnn_classifier.pth"
IMAGE_PATH = "../examples/salewa_sample.png"
TRAINING_DATA_DIR = "../training_data"
INDEX_PATH = "../data/faiss_index.index"
META_PATH = "../data/item_metadata.json"
CONFIDENCE_THRESHOLD = 0.8  # CNNでの予測確信度のしきい値

# ======================
# クラス名の取得
# ======================
dataset = ImageFolder(root=TRAINING_DATA_DIR)
classes = dataset.classes

# ======================
# モデルのロード
# ======================
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ======================
# 画像の読み込みと前処理
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# ======================
# CNNによる予測
# ======================
with torch.no_grad():
    outputs = model(input_tensor)
    probs = F.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probs, dim=1)
    predicted_class = classes[predicted_idx.item()]
    confidence = confidence.item()

# ======================
# CNN確信度が高ければ採用
# ======================
if confidence >= CONFIDENCE_THRESHOLD:
    print("✅ CNNによる予測")
    print(f"クラスID: {predicted_class}")
    print(f"確信度: {confidence:.2f}")
else:
    print("🔁 CNNの確信度が低いため、FAISSによる類似検索へ切り替え")

    # ResNet特徴抽出用モデル（最後のfc層を除く）
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    with torch.no_grad():
        features = feature_extractor(input_tensor).squeeze().numpy().astype("float32")

    # FAISSインデックス読み込み
    index = faiss.read_index(INDEX_PATH)

    # 類似検索
    D, I = index.search(np.array([features]), k=1)
    nearest_idx = I[0][0]

    # メタデータ読み込み
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    item = metadata[nearest_idx]
    print("✅ FAISSによる類似アイテム検索結果")
    print(f"アイテムID: {item['id']}")
    print(f"画像パス: {item['image_path']}")
