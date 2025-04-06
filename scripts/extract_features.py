import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

# フォルダのパス
IMAGE_DIR = "../item_images"
FEATURE_DIR = "../features"
META_PATH = "../data/item_metadata.json"

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(META_PATH), exist_ok=True)

# ResNet50モデルのロード（最後の分類層を除く）
model = resnet50(pretrained=True)
model.eval()
model = torch.nn.Sequential(*list(model.children())[:-1])  # 最後のFC層を除く

# デバイス設定（GPUがあれば使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 画像変換設定
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# メタ情報リスト
metadata = []

# 画像の読み込みと特徴量抽出
for filename in tqdm(os.listdir(IMAGE_DIR), desc="Extracting features"):
    if not filename.endswith(".webp"):
        continue

    item_id = filename.split(".")[0]
    image_path = os.path.join(IMAGE_DIR, filename)

    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model(input_tensor).squeeze().cpu().numpy()

        # 特徴量保存
        feature_path = os.path.join(FEATURE_DIR, f"{item_id}.npy")
        np.save(feature_path, feature)

        metadata.append({
            "id": item_id,
            "image_path": image_path,
            "feature_path": feature_path
        })

    except Exception as e:
        print(f"❌ Failed for {filename}: {e}")

# メタ情報保存
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("✅ 特徴量抽出完了！")
