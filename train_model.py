from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier

df = pd.read_csv("race_merged.csv")

features = [
    "lane","lane_inv","inside_bias","lane_power",
    "exhibit","power_score","grade_num",
    "power_diff","exhibit_diff","winrate_diff",
]

X = df[features]
y = df["is_win"]

# =====================
# 分割
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

X_tr, X_cal, y_tr, y_cal = train_test_split(
    X_train, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

# =====================
# モデル
# =====================
scale = (y == 0).sum() / (y == 1).sum()
print("クラス重み:", scale)

base_model = XGBClassifier(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.025,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale * 1.5,
    gamma=0.2,
    min_child_weight=3,
    eval_metric="logloss",
    random_state=42
)

# ⭐ 先に学習
base_model.fit(X_tr, y_tr)

# =====================
# ⭐ 新方式キャリブレーション
# =====================
cal_model = CalibratedClassifierCV(
    estimator=base_model,
    method="isotonic",
    cv=None   # ← ここ重要（prefitの代替）
)

cal_model.fit(X_cal, y_cal)

# =====================
# 確率取得
# =====================
proba = cal_model.predict_proba(X_test)[:,1]
pred  = cal_model.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score

# =====================
# 評価
# =====================
print("\n🔥 Calibrated XGBoost精度:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# =====================
# 確率ランキング
# =====================
result = X_test.copy()
result["win_prob"] = proba
result["actual"] = y_test.values

print("\nTOP予測確率")
print(result.sort_values("win_prob", ascending=False).head(20))

# =====================
# レース単位表示
# =====================
df_test = df.loc[X_test.index].copy()
df_test["win_prob"] = proba

for race_id, group in df_test.groupby("race"):

    print("\n========================")
    print("RACE", race_id)
    print("========================")

    g = group.sort_values("win_prob", ascending=False)

    print(g[[
        "lane",
        "racer",
        "win_prob",
        "finish"
    ]].head(6))
import joblib
joblib.dump(cal_model, "boat_model.pkl")
print("✅ モデル保存完了")

