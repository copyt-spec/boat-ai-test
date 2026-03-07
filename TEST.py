import pandas as pd

df = pd.read_csv("data/datasets/trifecta_train_features.csv")

print(df["y_combo"].value_counts().head(20))
