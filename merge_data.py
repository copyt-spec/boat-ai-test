import pandas as pd

# 読み込み
race = pd.read_csv("race_output.csv")
racers = pd.read_csv("racers_master.csv")

# ===== 数値変換（超重要） =====
racers["win_rate"] = racers["win_rate"] / 100
racers["place_rate"] = racers["place_rate"] / 10
racers["start_timing"] = racers["start_timing"] / 100

# ===== グレード数値化 =====
grade_map = {
    "A1":4,
    "A2":3,
    "B1":2,
    "B2":1
}
racers["grade_num"] = racers["grade"].map(grade_map)

# ===== 結合 =====
merged = race.merge(
    racers,
    left_on="racer",
    right_on="id",
    how="left"
)

# ===== 追加特徴量 =====
merged["is_win"] = (merged["finish"] == 1).astype(int)

merged["power_index"] = (
      merged["win_rate"] * 0.5
    + merged["place_rate"] * 0.3
    + merged["grade_num"] * 0.2
)

# 保存
merged.to_csv("race_merged.csv", index=False)

print("✅ 結合完了")
print("レコード:", len(merged))
print("欠損確認:")
print(merged.isna().sum().sort_values(ascending=False).head())


# ===== コース特徴量 =====

merged["lane_inv"] = 7 - merged["lane"]

merged["inside_bias"] = merged["lane"].apply(
    lambda x: 1 if x <= 3 else 0
)

merged["lane_power"] = merged["power_index"] * merged["lane_inv"]

# 保存（上書き）
merged.to_csv("race_merged.csv", index=False)

print("✅ コース特徴量追加完了")

merged["power_diff"] = merged["power_index"] - merged.groupby("race")["power_index"].transform("mean")

merged["exhibit_diff"] = merged["exhibit"] - merged.groupby("race")["exhibit"].transform("mean")

merged["winrate_diff"] = merged["win_rate"] - merged.groupby("race")["win_rate"].transform("mean")

merged.to_csv("race_merged.csv", index=False)

print("🔥 相対特徴量追加完了")
