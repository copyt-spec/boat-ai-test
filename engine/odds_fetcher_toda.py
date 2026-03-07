# engine/odds_fetcher.py

from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from typing import Dict, Tuple, Any, Optional


def _jcd_str(venue_code: int) -> str:
    # BOATRACEのjcdは 02 / 15 のように2桁
    return str(int(venue_code)).zfill(2)


def fetch_odds(race_no: int, date: str, venue_code: int = 15) -> Dict[str, Any]:
    """
    BOATRACE 公式サイトの三連単オッズを取得して grouped_odds 形式で返す
    grouped_odds = { "data": {first: {(second,third): odds}}, "min": float, "max": float }

    ※既存UI(index.html)前提でこの構造は絶対に変更しない
    """
    jcd = _jcd_str(venue_code)

    # PC版（あなたの既存実装と同系統）
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={int(race_no)}&jcd={jcd}&hd={date}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    res = requests.get(url, headers=headers, timeout=15)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    # --- odds3t は「1着ごとのブロック」に三連単が並ぶ構造（公式HTML変更に弱いので守りを入れる） ---
    # 既存で動いている前提に合わせ、"odds3t" のテーブル群から (first, second, third) を拾う。
    # もしあなたの既存 parser が別なら、下の parse 部分だけ差し替えればOK（返却形式は固定）。

    data: Dict[int, Dict[Tuple[int, int], str]] = {i: {} for i in range(1, 7)}
    nums: list[float] = []

    # “3連単”の表は複数テーブルで並ぶことが多いので、候補テーブルを広めに拾う
    tables = soup.find_all("table")
    if not tables:
        return {"data": data, "min": 0.0, "max": 0.0}

    # 超保守：ページ内の "odds" っぽいセルを走査し、
    # 近傍の艇番(1-6)表記から (first, second, third) を推定するのは危険なので、
    # ここでは「既存で動いていた odds3t パーサの形」を踏襲しやすいよう、
    # 典型的な class / data 属性に寄せて拾う。
    #
    # ---- 実用優先：よくある構造（1着=見出し、2-3=行頭、Odds=セル）に対応 ----
    # まず “1着” を含む見出し(th) から first を拾い、そのテーブル内を解析する。

    for tbl in tables:
        th = tbl.find("th")
        if not th:
            continue
        th_text = th.get_text(strip=True)
        # 例: "1着" "2着" など。あなたのUIは firstごとにテーブルを並べる前提なので "着" を目印にする
        if "着" not in th_text:
            continue

        # firstを決める： "1着" の先頭数字を拾う
        first: Optional[int] = None
        for ch in th_text:
            if ch.isdigit():
                first = int(ch)
                break
        if first is None or first not in range(1, 7):
            continue

        # 行を走査：先頭セルが "2-3" 形式、次セルがオッズ、のような形を想定
        rows = tbl.find_all("tr")
        for tr in rows:
            tds = tr.find_all(["td"])
            if len(tds) < 2:
                continue

            key_text = tds[0].get_text(strip=True)  # "2-3" みたいなの
            if "-" not in key_text:
                continue
            parts = key_text.split("-")
            if len(parts) != 2:
                continue

            try:
                second = int(parts[0])
                third = int(parts[1])
            except Exception:
                continue

            # 不正組み合わせ除外
            if second == first or third == first or second == third:
                continue
            if second not in range(1, 7) or third not in range(1, 7):
                continue

            odds_text = tds[1].get_text(strip=True)
            if not odds_text or odds_text == "-":
                continue

            # 数値化できるものだけ min/max 対象に
            try:
                val = float(odds_text)
                nums.append(val)
            except Exception:
                # 文字が混じる場合は表示はさせる（ただし min/max 対象外）
                val = None

            data[first][(second, third)] = odds_text

    if nums:
        mn = min(nums)
        mx = max(nums)
    else:
        mn = 0.0
        mx = 0.0

    return {"data": data, "min": float(mn), "max": float(mx)}
