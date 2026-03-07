import pandas as pd

rows = []

with open("racers_sjis.txt", "rb") as f:
    for raw in f:

        pos = [0]

        def cut(n):
            chunk = raw[pos[0]:pos[0]+n]
            pos[0] += n
            return chunk.decode("cp932").strip()

        row = {
            "id": cut(4),
            "name": cut(16),
            "kana": cut(15),
            "branch": cut(4),
            "grade": cut(2),
            "era": cut(1),
            "birth": cut(6),
            "sex": cut(1),
            "age": cut(2),
            "height": cut(3),
            "weight": cut(2),
            "blood": cut(2),
            "win_rate": cut(4),
            "place_rate": cut(4),
            "start_timing": cut(3),
        }

        rows.append(row)

df = pd.DataFrame(rows)

print(df.head())

# ===== 数値変換 =====

df["win_rate"] = pd.to_numeric(df["win_rate"], errors="coerce") / 100
df["place_rate"] = pd.to_numeric(df["place_rate"], errors="coerce") / 10
df["start_timing"] = pd.to_numeric(df["start_timing"], errors="coerce") / 100

grade_map = {
    "A1": 4,
    "A2": 3,
    "B1": 2,
    "B2": 1
}

df["grade_score"] = df["grade"].map(grade_map)

# ===== スコア算出 =====
df["power_score"] = (
    df["grade_score"] * 2
    + df["win_rate"]
    + df["place_rate"] * 0.5
    - df["start_timing"] * 2
)

# ===== 上位表示 =====
print(df.sort_values("power_score", ascending=False)[
    ["name","grade","win_rate","place_rate","start_timing","power_score"]
].head(15))

df.to_csv("racers_master.csv", index=False)
print("✅ racers_master.csv 保存完了")


