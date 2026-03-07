from fastapi import FastAPI
import pandas as pd
import joblib
from trifecta_engine import trifecta_probs

app = FastAPI()

# ===== モデル読み込み（起動時1回） =====
rank_model = joblib.load("rank_model.pkl")

# trifecta_engine内でrank_modelを使っているので、
# グローバル差し替え
import trifecta_engine
trifecta_engine.rank_model = rank_model


@app.get("/")
def root():
    return {"status": "Boat AI running 🚀"}


@app.get("/predict/{race_index}")
def predict(race_index: int):

    df = pd.read_csv("race_merged.csv")

    # 6行で1レース前提
    start = race_index * 6
    race_df = df.iloc[start:start+6].copy()

    if len(race_df) != 6:
        return {"error": "invalid race index"}

    result = trifecta_probs(race_df)

    return result.head(10).to_dict(orient="records")

from fetch_today import fetch_today_race
from trifecta_engine import trifecta_probs

@app.get("/predict_today")
def predict_today():

    url = "今日の出走表URL"
    df = fetch_today_race(url)

    result = trifecta_probs(df)

    return result.head(10).to_dict(orient="records")

from fetch_macour import fetch_macour_kojima

@app.get("/today/{date_str}")
def today(date_str: str):
    """
    date_str = '20260220'
    """
    races = fetch_macour_kojima(date_str)

    # JSONに変換
    result = {}
    for k,v in races.items():
        result[k] = v.to_dict(orient="records")

    return result

