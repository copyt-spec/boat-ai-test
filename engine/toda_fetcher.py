# engine/toda_fetcher.py
# 戸田（jcd=02）出走表：BOATRACE racelist を requests + BS4 で取得（table#2 / tbody分割対応）
#
# ログから確定した仕様：
# - “正しい出走表” は table#2（スコア最大）
# - 1レース分が tbody#1..#6 に分割され、各tbodyの tr#01 が選手情報
# - 枠番号が全角（１〜６）で入っていることがある -> 全角対応必須
# - 余計なtr（隊列/展示ST等）は tr#02以降にあるので無視できる

import re
import time
from typing import List, Dict, Any, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Tag

JCD_TODA = 2

UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
}

RE_RACERNO_ANY = re.compile(r"(\d{4})")
RE_GRADE = re.compile(r"\b([A-Z]\d|B\d)\b")
RE_RACER_GRADE = re.compile(r"(\d{4})\s*/\s*([A-Z]\d|B\d)")

# 全角数字 -> 半角
ZEN2HAN = str.maketrans({"１":"1","２":"2","３":"3","４":"4","５":"5","６":"6"})


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _normalize_digits(s: str) -> str:
    return (s or "").translate(ZEN2HAN)


def _format_fl(text: str) -> str:
    """
    Fは F0でも表示OK
    Lは L0は表示しない（L1以上のみ）
    """
    parts = []
    mf = re.search(r"\bF(\d+)\b", text)
    if mf:
        parts.append(f"F{mf.group(1)}")

    ml = re.search(r"\bL(\d+)\b", text)
    if ml:
        n = int(ml.group(1))
        if n >= 1:
            parts.append(f"L{n}")

    return " ".join(parts) if parts else "-"


def _extract_lane_from_text(text: str) -> Optional[int]:
    """
    行テキストの先頭付近から枠番号を拾う（全角対応済み）
    例: '１ 3860 / B1 ...' -> 1
    """
    t = _normalize_digits(_clean(text))
    # 先頭 or 先頭近くの 1..6
    m = re.search(r"(^| )([1-6])( |$)", t)
    if m:
        return int(m.group(2))
    # どうしても詰まってたら先頭文字
    if t and t[0] in "123456":
        return int(t[0])
    return None


def _pick_racer_no(text: str) -> Optional[str]:
    m = RE_RACERNO_ANY.search(text)
    return m.group(1) if m else None


def _parse_entry_line(text: str) -> Optional[Dict[str, Any]]:
    """
    1艇ぶんの情報が詰まったテキストからエントリーを作る
    """
    t = _normalize_digits(_clean(text))

    lane = _extract_lane_from_text(t)
    if lane is None:
        return None

    racer_no = _pick_racer_no(t)
    if not racer_no:
        return None

    grade = "-"
    mg = RE_RACER_GRADE.search(t)
    if mg:
        grade = mg.group(2)
    else:
        mg2 = RE_GRADE.search(t)
        if mg2:
            grade = mg2.group(1)

    # 氏名
    name = "-"
    mname = re.search(r"([ぁ-んァ-ン一-龥]{2,12})\s+([ぁ-んァ-ン一-龥]{1,12})", t)
    if mname:
        name = _clean(f"{mname.group(1)} {mname.group(2)}")

    # 支部/出身
    branch = "-"
    mbr = re.search(r"([^\s/]{1,6}/[^\s/]{1,6})", t)
    if mbr:
        branch = mbr.group(1)

    name_branch = "-"
    if name != "-" or branch != "-":
        name_branch = _clean(f"{'' if name=='-' else name} {'' if branch=='-' else branch}")

    fl = _format_fl(t)

    # 勝率/2連率（小数群から 勝率=3..10、2連率=10..100）
    nums = [float(x) for x in re.findall(r"\d+\.\d+", t)]
    win_rate = "-"
    quinella = "-"

    for v in nums:
        if 2.5 <= v <= 10.5:
            win_rate = f"{v:.2f}"
            break

    for v in nums:
        if 10.0 <= v <= 100.0:
            quinella = f"{v:.2f}"
            break

    # fallback（ST 0.19 を避けつつ）
    if quinella == "-":
        for v in nums:
            if 0.30 <= v <= 100.0:
                if win_rate != "-" and abs(v - float(win_rate)) < 1e-9:
                    continue
                quinella = f"{v:.2f}"
                break

    return {
        "lane": lane,
        "racer_no": racer_no,
        "name_branch": name_branch,
        "grade": grade,
        "fl": fl,
        "win_rate": win_rate,
        "quinella_rate": quinella,
        "motor": None,
        "boat": None,
        "exhibit": None,
        "start_timing": None,
    }


def _table_score(table: Tag) -> Tuple[int, int, int]:
    """
    テーブルの “出走表らしさ” を数値化
    - uniq_lanes: 1..6 のユニーク数（最重要）
    - lane_hits : lane検出回数（補助）
    - racer_hits: 4桁検出回数（補助）
    """
    uniq = set()
    lane_hits = 0
    racer_hits = 0

    for tr in table.find_all("tr"):
        txt = _normalize_digits(_clean(tr.get_text(" ")))
        lane = _extract_lane_from_text(txt)
        if lane is not None and 1 <= lane <= 6:
            uniq.add(lane)
            lane_hits += 1
        if RE_RACERNO_ANY.search(txt):
            racer_hits += 1

    return len(uniq), lane_hits, racer_hits


def _select_best_table(soup: BeautifulSoup) -> Optional[Tag]:
    tables = soup.find_all("table")
    if not tables:
        return None
    best = None
    best_key = (-1, -1, -1)
    for t in tables:
        key = _table_score(t)
        if key > best_key:
            best_key = key
            best = t
    return best


def fetch_toda_racelist(race_no: int, date: str) -> List[Dict[str, Any]]:
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?hd={date}&jcd={JCD_TODA:02d}&rno={race_no}"
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    table = _select_best_table(soup)
    if not table:
        return []

    # ★ログ通り：tbodyが6個に分割され、各tbodyの tr#01 が1艇分
    tbodies = table.find_all("tbody")
    best_by_lane: Dict[int, Dict[str, Any]] = {}

    if tbodies:
        for tbody in tbodies:
            trs = tbody.find_all("tr")
            if not trs:
                continue
            first_tr = trs[0]  # tr#01だけ使う
            e = _parse_entry_line(first_tr.get_text(" "))
            if not e:
                continue
            lane = int(e["lane"])
            if 1 <= lane <= 6 and lane not in best_by_lane:
                best_by_lane[lane] = e
            if len(best_by_lane) == 6:
                break
    else:
        # tbodyが無い場合の保険（通常は来ない）
        for tr in table.find_all("tr"):
            e = _parse_entry_line(tr.get_text(" "))
            if not e:
                continue
            lane = int(e["lane"])
            if lane not in best_by_lane:
                best_by_lane[lane] = e
            if len(best_by_lane) == 6:
                break

    entries: List[Dict[str, Any]] = []
    for lane in range(1, 7):
        if lane in best_by_lane:
            row = best_by_lane[lane]
            row["race_no"] = race_no
            entries.append(row)

    entries.sort(key=lambda x: x["lane"])
    return entries


def fetch_all_toda_entries_once(date: str, sleep_sec: float = 0.15) -> List[Dict[str, Any]]:
    all_entries: List[Dict[str, Any]] = []
    for rno in range(1, 13):
        try:
            all_entries.extend(fetch_toda_racelist(rno, date))
        except Exception:
            pass
        time.sleep(sleep_sec)
    return all_entries
