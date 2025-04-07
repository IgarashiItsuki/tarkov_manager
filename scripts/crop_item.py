from PIL import Image
import os

# 入力画像のパス（切り抜き後のアイテム画像）
INPUT_PATH = "../examples/salewa_cropped.png"

# 出力画像の保存パス
OUTPUT_PATH = "../examples/salewa_ready.png"

# 出力サイズ（ResNet50の入力に合わせて 224x224）
TARGET_SIZE = 224
ITEM_SIZE = 180  # アイテム自体のサイズ（中央に貼り付ける）

# 入力画像の読み込み
img = Image.open(INPUT_PATH).convert("RGB")

# アイテム画像をリサイズ
img_resized = img.resize((ITEM_SIZE, ITEM_SIZE))

# 黒背景の正方形画像を作成
background = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))

# 中央にリサイズ画像を貼り付け
paste_position = ((TARGET_SIZE - ITEM_SIZE) // 2, (TARGET_SIZE - ITEM_SIZE) // 2)
background.paste(img_resized, paste_position)

# 保存
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
background.save(OUTPUT_PATH)

print(f"✅ 正方形画像として保存完了: {OUTPUT_PATH}")
