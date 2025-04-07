import os
import torch
from torchvision import models, transforms
from PIL import Image
from torchvision.datasets import ImageFolder

# ====================
# パス設定
# ====================
MODEL_PATH = "../models/cnn_classifier.pth"
IMAGE_PATH = "../examples/afak_sample.png"
TRAINING_DATA_DIR = "../training_data"

# ====================
# クラス名（ID）一覧を取得
# ====================
dataset = ImageFolder(root=TRAINING_DATA_DIR)
classes = dataset.classes  # フォルダ名がそのままクラスIDとして取得される

# ====================
# モデル定義
# ====================
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ====================
# 入力画像の前処理
# ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# ====================
# 推論
# ====================
with torch.no_grad():
    outputs = model(input_tensor)
    predicted_idx = torch.argmax(outputs, dim=1).item()
    predicted_class = classes[predicted_idx]
    print(f"✅ 予測されたクラスID: {predicted_class}")
