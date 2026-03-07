import pandas as pd
import numpy as np
import time
import os

FEATURES = [
    "lane","lane_inv","inside_bias","lane_power",
    "exhibit","power_score","grade_num",
    "power_diff","exhibit_diff","winrate_diff"
]

OUT_FILE = "pairwise.csv"

print("読み込み中...")
df = pd.read_csv("race_merged.csv")

# ⭐ 6行ごとにレースID振る
df["race_id"] = df.index // 6

print("レース数:", df["race_id"].nunique())

if os.path.exists(OUT_FILE):
    os.remove(OUT_FILE)

total_pairs = 0
race_count = 0

with open(OUT_FILE, "w") as f:
    f.write(",".join(FEATURES) + ",label\n")

    print("\nペア生成開始")
    t1 = time.time()

    for race, g in df.groupby("race_id"):

        if len(g) != 6:
            continue

        race_count += 1

        X = g[FEATURES].values.astype(np.float32)
        finish = g["finish"].values

        diff = X[:,None,:] - X[None,:,:]
        lab = (finish[:,None] < finish[None,:]).astype(int)

        for i in range(6):
            for j in range(6):
                if i == j:
                    continue

                row = ",".join(map(str, diff[i,j]))
                f.write(row + f",{lab[i,j]}\n")
                total_pairs += 1

        if race_count % 500 == 0:
            print("処理レース:", race_count)

print("\n✅ 完了")
print("総ペア数:", total_pairs)
