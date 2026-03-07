import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

print("読み込み中...")
df = pd.read_csv("pairwise.csv")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("学習開始...")

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6
)

model.fit(X_train, y_train)

pred = model.predict_proba(X_valid)[:,1]
auc = roc_auc_score(y_valid, pred)

print("AUC:", round(auc, 4))

joblib.dump(model, "rank_model.pkl")

print("✅ rank_model.pkl 保存完了")
