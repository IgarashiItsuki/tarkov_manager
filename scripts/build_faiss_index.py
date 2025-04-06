import os
import json
import numpy as np
import faiss

# メタデータと特徴量フォルダのパス
META_PATH = "../data/item_metadata.json"
INDEX_PATH = "../data/faiss_index.index"

# メタデータ読み込み
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# 特徴量を全てロード
features = []
for item in metadata:
    try:
        vec = np.load(item["feature_path"]).astype("float32")  # FAISSはfloat32のみ対応
        features.append(vec)
    except Exception as e:
        print(f"❌ Failed to load {item['feature_path']}: {e}")

features = np.stack(features)  # (N, 2048)

# FAISSインデックスの作成
d = features.shape[1]  # 特徴量の次元数（通常2048）
index = faiss.IndexFlatL2(d)
index.add(features)

# 保存
faiss.write_index(index, INDEX_PATH)

print(f"✅ インデックス作成完了！{features.shape[0]} 件の特徴量を登録しました。")
print(f"📦 保存先: {INDEX_PATH}")
