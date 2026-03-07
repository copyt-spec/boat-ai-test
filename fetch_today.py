import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

def fetch_today_race(url: str):

    res = requests.get(url)
    res.encoding = "utf-8"

    soup = BeautifulSoup(res.text, "lxml")

    rows = []

    # ★ サイト構造に合わせてセレクタ調整
    table = soup.select_one("table")
    trs = table.select("tr")[1:]  # ヘッダ除外

    for tr in trs:
        tds = tr.select("td")
        if len(tds) < 6:
            continue

        rows.append({
            "lane": int(tds[0].text.strip()),
            "name": tds[1].text.strip(),
            "exhibit": float(tds[2].text.strip())
        })

    df = pd.DataFrame(rows)

    return df
