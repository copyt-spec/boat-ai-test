import pandas as pd

df = pd.read_csv("data/datasets/trifecta_train_features.csv")

cols = [
    "combo",
    "a_lane", "b_lane", "c_lane",
    "a_racer_win_rate", "b_racer_win_rate", "c_racer_win_rate",
    "a_racer_ability_index", "b_racer_ability_index", "c_racer_ability_index",
    "ab_racer_win_rate_diff", "ac_racer_win_rate_diff", "bc_racer_win_rate_diff",
]

exists = [c for c in cols if c in df.columns]
print(df[exists].head(20).to_string())
