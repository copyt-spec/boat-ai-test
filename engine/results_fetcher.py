# engine/results_fetcher.py

import re
import requests
from bs4 import BeautifulSoup


def fetch_trifecta_result(venue_code: int, date: str, race_no: int, timeout: int = 10) -> str | None:
    """
    BOATRACE公式の結果ページから3連単結果（例: "1-2-3"）を取得する
    venue_code: 丸亀=15
    date: "YYYYMMDD"
    race_no: 1〜12

    return:
      "1-2-3" もしくは None（未確定/取得失敗）
    """
    # 結果ページ（PC）
    url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={race_no}&jcd={venue_code}&hd={date}"

    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # ページ内に "1-2-3" の並びがあることが多いので、まずはテキストから探索
    text = soup.get_text(" ", strip=True)

    # 代表的な取り方：1〜6の数字が3つ並ぶパターンを探す
    # ※ページ構造が変わる可能性があるので、ここは保守的に “見つかったら採用”
    m = re.search(r"\b([1-6])\s*[-ー]\s*([1-6])\s*[-ー]\s*([1-6])\b", text)
    if m:
        a, b, c = m.group(1), m.group(2), m.group(3)
        if len({a, b, c}) == 3:
            return f"{a}-{b}-{c}"

    # もし上で取れない場合、HTML上の枠順っぽいものを探す（追加保険）
    # 例：順位欄に 1 2 3 が並ぶケース
    # ここでは最小限で止める（重くしない）
    candidates = re.findall(r"\b[1-6]\b", text)
    # 数字が多すぎる場合は精度が落ちるので、ここではNoneにして上位側でスキップ
    return None
