import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データ前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# データ読み込み
dataset = ImageFolder(root="../training_data", transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# モデル定義（ResNet50の最後だけ変更）
model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes))
model = model.to(device)

# 損失関数と最適化
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 学習
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f}")

# モデル保存
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/cnn_classifier.pth")
print("✅ モデル学習完了 & 保存しました！")
