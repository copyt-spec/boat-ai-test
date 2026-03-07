import pandas as pd
import joblib

# =========================
# モデル読み込み
# =========================
model = joblib.load("boat_model.pkl")

# =========================
# ⭐ 未来レース入力（例）
# ※ここを当日データで置き換える
# =========================
race = pd.DataFrame([
    {"lane":1,"lane_inv":6,"inside_bias":1,"lane_power":1.2,
     "exhibit":6.70,"power_score":72,"grade_num":5,
     "power_diff":10,"exhibit_diff":-0.03,"winrate_diff":0.12},

    {"lane":2,"lane_inv":5,"inside_bias":0.8,"lane_power":1.0,
     "exhibit":6.74,"power_score":60,"grade_num":4,
     "power_diff":-2,"exhibit_diff":0.02,"winrate_diff":-0.01},

    {"lane":3,"lane_inv":4,"inside_bias":0.6,"lane_power":0.9,
     "exhibit":6.80,"power_score":55,"grade_num":3,
     "power_diff":-5,"exhibit_diff":0.08,"winrate_diff":-0.05},

    {"lane":4,"lane_inv":3,"inside_bias":0.4,"lane_power":0.7,
     "exhibit":6.82,"power_score":50,"grade_num":3,
     "power_diff":-7,"exhibit_diff":0.10,"winrate_diff":-0.08},

    {"lane":5,"lane_inv":2,"inside_bias":0.3,"lane_power":0.6,
     "exhibit":6.88,"power_score":48,"grade_num":2,
     "power_diff":-10,"exhibit_diff":0.15,"winrate_diff":-0.10},

    {"lane":6,"lane_inv":1,"inside_bias":0.2,"lane_power":0.5,
     "exhibit":6.95,"power_score":40,"grade_num":2,
     "power_diff":-15,"exhibit_diff":0.22,"winrate_diff":-0.15},
])

# =========================
# 予測
# =========================
probs = model.predict_proba(race)[:,1]

race["win_prob"] = probs

# =========================
# 表示
# =========================
ranking = race.sort_values("win_prob", ascending=False)

print("\n===== 勝率ランキング =====")
print(ranking[["lane","win_prob"]])

print("\n🔥 本命")
print(ranking.iloc[0])
