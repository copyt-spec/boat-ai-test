# train_rank_model.py

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# ==========================
# 1. データ読み込み
# ==========================

DATA_PATH = "race_output.csv"

print("📥 データ読み込み中...")
df = pd.read_csv(DATA_PATH)

print("総レコード数:", len(df))


# ==========================
# 2. 特徴量と目的変数
# ==========================

# 今回は軽量構成
feature_cols = [
    "lane",
    "motor",
    "boat",
    "exhibit",
    "entry",
    "start"
]

X = df[feature_cols]
y = df["finish"] - 1   # 1着=0, 2着=1 ... に変換（重要）


# ==========================
# 3. スケーリング
# ==========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ スケーリング完了")


# ==========================
# 4. 学習
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)

print("🌲 モデル学習中...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("✅ 学習完了")


# ==========================
# 5. 精度確認
# ==========================

y_pred = model.predict(X_test)

print("\n📊 分類レポート")
print(classification_report(y_test, y_pred))


# ==========================
# 6. 保存
# ==========================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/rank_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n💾 保存完了")
print("models/rank_model.pkl")
print("models/scaler.pkl")
