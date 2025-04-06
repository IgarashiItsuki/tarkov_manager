import requests
import json
import pandas as pd

# Tarkov.dev GraphQL API
url = "https://api.tarkov.dev/graphql"

# すべてのアイテム情報を取得
query = """
{
  items {
    id
    name
    shortName
  }
}
"""

response = requests.post(url, json={"query": query})
data = response.json()
items = data["data"]["items"]

# 画像URLを付加
for item in items:
    item["image_url"] = f"https://assets.tarkov.dev/{item['id']}-512.webp"

# JSONで保存
with open("tarkov_items.json", "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

# CSVでも保存（Excelなどで見やすい）
df = pd.DataFrame(items)
df.to_csv("tarkov_items.csv", index=False, encoding="utf-8-sig")

print(f"✅ {len(items)} 件のアイテムを保存しました")
