# engine/toda_fetcher.py
# 戸田（jcd=02）出走表：BOATRACE racelist を requests + BS4 で取得（table#2 / tbody分割対応）
#
# 改善点:
# - Session 再利用で高速化
# - 1R単位 / 12R単位のメモリキャッシュ
# - 軽いリトライ
# - 既存の解析ロジックは維持
#
# ログから確定した仕様：
# - “正しい出走表” は table#2（スコア最大）
# - 1レース分が tbody#1..#6 に分割され、各tbodyの tr#01 が選手情報
# - 枠番号が全角（１〜６）で入っていることがある -> 全角対応必須
# - 余計なtr（隊列/展示ST等）は tr#02以降にあるので無視できる

from __future__ import annotations

import re
import time
from typing import List, Dict, Any, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

JCD_TODA = 2

UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
}

RE_RACERNO_ANY = re.compile(r"(\d{4})")
RE_GRADE = re.compile(r"\b([A-Z]\d|B\d)\b")
RE_RACER_GRADE = re.compile(r"(\d{4})\s*/\s*([A-Z]\d|B\d)")

# 全角数字 -> 半角
ZEN2HAN = str.maketrans({"１": "1", "２": "2", "３": "3", "４": "4", "５": "5", "６": "6"})

# =========================
# cache
# =========================
_RACE_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_ALL_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

# 出走表は短時間キャッシュで十分
_CACHE_SECONDS_RACE = 60
_CACHE_SECONDS_ALL = 60

# 共有session
_SESSION: Optional[requests.Session] = None


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

    m = re.search(r"(^| )([1-6])( |$)", t)
    if m:
        return int(m.group(2))

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
        name_branch = _clean(f"{'' if name == '-' else name} {'' if branch == '-' else branch}")

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


def _get_cache(cache: Dict[str, Tuple[float, List[Dict[str, Any]]]], key: str, ttl: int) -> Optional[List[Dict[str, Any]]]:
    item = cache.get(key)
    if not item:
        return None

    ts, data = item
    if time.time() - ts > ttl:
        cache.pop(key, None)
        return None

    return data


def _set_cache(cache: Dict[str, Tuple[float, List[Dict[str, Any]]]], key: str, value: List[Dict[str, Any]]) -> None:
    cache[key] = (time.time(), value)


def _create_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=2,
        read=2,
        connect=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(UA)
    return session


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = _create_session()
    return _SESSION


def fetch_toda_racelist(race_no: int, date: str) -> List[Dict[str, Any]]:
    cache_key = f"toda_racelist_{date}_{race_no}"
    cached = _get_cache(_RACE_CACHE, cache_key, _CACHE_SECONDS_RACE)
    if cached is not None:
        return cached

    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?hd={date}&jcd={JCD_TODA:02d}&rno={race_no}"

    session = _get_session()
    r = session.get(url, timeout=(5, 15))
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

    _set_cache(_RACE_CACHE, cache_key, entries)
    return entries


def fetch_all_toda_entries_once(date: str, sleep_sec: float = 0.03) -> List[Dict[str, Any]]:
    """
    戸田12R分をまとめて取得
    - 12R全体もキャッシュ
    - 1Rごとの結果もキャッシュ
    """
    cache_key = f"toda_all_entries_{date}"
    cached = _get_cache(_ALL_CACHE, cache_key, _CACHE_SECONDS_ALL)
    if cached is not None:
        return cached

    all_entries: List[Dict[str, Any]] = []

    for rno in range(1, 13):
        try:
            rows = fetch_toda_racelist(rno, date)
            all_entries.extend(rows)
        except Exception:
            pass

        # 連打しすぎ回避。既存よりかなり短くして速度改善
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    _set_cache(_ALL_CACHE, cache_key, all_entries)
    return all_entries
