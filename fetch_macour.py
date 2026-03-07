import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_macour_kojima(date_str: str):
    """
    date_str: 例 '20260220'
    """
    url = f"https://sp.macour.jp/kojima/racecards/{date_str}/"
    res = requests.get(url)
    res.encoding = "utf-8"
    soup = BeautifulSoup(res.text, "lxml")

    all_races = {}

    # レースごとに探す
    race_sections = soup.select(".race-card")

    for sec in race_sections:
        race_no = sec.select_one(".race-no").text.strip()

        boats = []

        boat_infos = sec.select(".boat-info")

        for bi in boat_infos:
            lane = int(bi.select_one(".lane").text.strip())
            name = bi.select_one(".name").text.strip()
            winrate = float(bi.select_one(".winrate").text.strip())
            exhibit = float(bi.select_one(".ex").text.strip())

            boats.append({
                "lane": lane,
                "name": name,
                "winrate": winrate,
                "exhibit": exhibit
            })

        all_races[race_no] = pd.DataFrame(boats)

    return all_races
