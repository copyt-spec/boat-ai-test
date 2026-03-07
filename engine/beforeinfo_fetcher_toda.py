# engine/beforeinfo_fetcher_toda.py

import re
import requests
from bs4 import BeautifulSoup
from typing import Any, Dict


_FW_TO_ASCII = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
})


class BeforeInfoRow:
    def __init__(self, exhibit_time: str = "未取得", st: str = "未取得", course: str = "未取得",
                 tilt: str = "未取得", parts: str = "未取得"):
        self.exhibit_time = exhibit_time
        self.st = st
        self.course = course
        self.tilt = tilt
        self.parts = parts


def fetch_beforeinfo_toda(race_no: int, date: str) -> Dict[Any, Any]:
    """
    戸田（jcd=02）の直前情報を boatrace.jp beforeinfo から取得。
    返却は既存テンプレに合わせて
      { 1: BeforeInfoRow(...), ..., "weather": "...", "wind_speed": "...", "wind_direction": "..." }
    の形。
    """
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?hd={date}&jcd=02&rno={race_no}"

    r = requests.get(url, timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text("\n", strip=True).translate(_FW_TO_ASCII)
    lines = [x.strip() for x in text.split("\n") if x.strip()]

    result: Dict[Any, Any] = {}

    # --- 1) 各枠の「展示タイム」「チルト」「部品」を拾う ---
    # 行はだいたい
    # 1 Image 名前 59.6kg 6.96 -0.5
    # の形で出る
    for lane in range(1, 7):
        exhibit = "未取得"
        tilt = "未取得"
        parts = "未取得"

        # lane 行を探す
        for i in range(len(lines)):
            if not (lines[i].startswith(f"{lane} ") or lines[i] == str(lane)):
                continue

            # 同じ行に数値が乗ってるケース
            row = lines[i]
            m = re.search(r"\b(\d+\.\d{2})\b\s+(-?\d+(?:\.\d+)?)\b", row)
            if m:
                exhibit = m.group(1)
                tilt = m.group(2)

            # 部品は、この後の数行に「リング×2」「キャブ」等が出ることがある
            for j in range(i + 1, min(i + 8, len(lines))):
                if lines[j] in {"1", "2", "3", "4", "5", "6"}:
                    break
                if "×" in lines[j] or "キャブ" in lines[j] or "リング" in lines[j] or "ギヤ" in lines[j]:
                    parts = lines[j]
                    break

            break

        # st/course はテキストから枠ごとの対応が取りづらい日があるので「未取得」扱いにする
        result[lane] = BeforeInfoRow(exhibit_time=exhibit, st="未取得", course="未取得", tilt=tilt, parts=parts)

    # --- 2) 水面気象情報（気温/風速/風向き） ---
    # 例:
    # 気温 16.0℃
    # 晴
    # 風速 7m
    temp = None
    weather_word = None
    wind_speed = None

    for i in range(len(lines)):
        if lines[i].startswith("気温"):
            temp = lines[i].replace("気温", "").strip()
            # 次行が天気（晴/曇/雨など）であることが多い
            if i + 1 < len(lines) and len(lines[i + 1]) <= 4:
                weather_word = lines[i + 1].strip()
        if lines[i].startswith("風速"):
            wind_speed = lines[i].replace("風速", "").strip()

    # テンプレ側は beforeinfo.weather / beforeinfo.wind_speed / beforeinfo.wind_direction を見にいくのでキーを揃える
    if temp and weather_word:
        result["weather"] = f"{temp} {weather_word}"
    elif temp:
        result["weather"] = temp

    if wind_speed:
        result["wind_speed"] = wind_speed

    # 風向きは pc beforeinfo のテキストだけだと取れない場合があるので未取得
    # （テンプレは未取得表示に落ちる）
    # result["wind_direction"] は入れない

    return result
