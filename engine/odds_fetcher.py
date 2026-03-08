# engine/odds_fetcher.py
# FIXED_ORDER は絶対に触らない（組み合わせの基準）

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


FIXED_ORDER = """1-2-3 2-1-3 3-1-2 4-1-2 5-1-2 6-1-2
1-2-4 2-1-4 3-1-4 4-1-3 5-1-3 6-1-3
1-2-5 2-1-5 3-1-5 4-1-5 5-1-4 6-1-4
1-2-6 2-1-6 3-1-6 4-1-6 5-1-6 6-1-5
1-3-2 2-3-1 3-2-1 4-2-1 5-2-1 6-2-1
1-3-4 2-3-4 3-2-4 4-2-3 5-2-3 6-2-3
1-3-5 2-3-5 3-2-5 4-2-5 5-2-4 6-2-4
1-3-6 2-3-6 3-2-6 4-2-6 5-2-6 6-2-5
1-4-2 2-4-1 3-4-1 4-3-1 5-3-1 6-3-1
1-4-3 2-4-3 3-4-2 4-3-2 5-3-2 6-3-2
1-4-5 2-4-5 3-4-5 4-3-5 5-3-4 6-3-4
1-4-6 2-4-6 3-4-6 4-3-6 5-3-6 6-3-5
1-5-2 2-5-1 3-5-1 4-5-1 5-4-1 6-4-1
1-5-3 2-5-3 3-5-2 4-5-2 5-4-2 6-4-2
1-5-4 2-5-4 3-5-4 4-5-3 5-4-3 6-4-3
1-5-6 2-5-6 3-5-6 4-5-6 5-4-6 6-4-5
1-6-2 2-6-1 3-6-1 4-6-1 5-6-1 6-5-1
1-6-3 2-6-3 3-6-2 4-6-2 5-6-2 6-5-2
1-6-4 2-6-4 3-6-4 4-6-3 5-6-3 6-5-3
1-6-5 2-6-5 3-6-5 4-6-5 5-6-4 6-5-4""".split()


# =========================
# shared session / cache
# =========================
_SESSION: Optional[requests.Session] = None
_CACHE: Dict[str, Tuple[float, Dict[str, str]]] = {}
_CACHE_SECONDS = 30  # オッズは短時間キャッシュ


def _cache_key(race_no: int, date: str, venue_code: int) -> str:
    return f"odds3t_{venue_code}_{date}_{race_no}"


def _get_cache(key: str) -> Optional[Dict[str, str]]:
    item = _CACHE.get(key)
    if not item:
        return None

    ts, data = item
    if time.time() - ts > _CACHE_SECONDS:
        _CACHE.pop(key, None)
        return None

    return data


def _set_cache(key: str, value: Dict[str, str]) -> None:
    _CACHE[key] = (time.time(), value)


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

    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.boatrace.jp/",
            "Connection": "keep-alive",
        }
    )

    return session


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = _create_session()
    return _SESSION


def _jcd2(venue_code: int) -> str:
    return str(int(venue_code)).zfill(2)


def fetch_odds(race_no, date, venue_code: int = 15) -> Dict[str, str]:
    """
    三連単120通りオッズを取得
    return:
      {
        "1-2-3": "5.4",
        ...
      }
    """
    race_no_i = int(race_no)
    key = _cache_key(race_no_i, str(date), int(venue_code))
    cached = _get_cache(key)
    if cached is not None:
        return cached

    jcd = _jcd2(venue_code)
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?jcd={jcd}&hd={date}&rno={race_no_i}"

    session = _get_session()
    response = session.get(url, timeout=(5, 15))
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    odds_cells = soup.select(".oddsPoint")

    odds_dict: Dict[str, str] = {}

    # 既存仕様維持：FIXED_ORDER と zip で対応
    for combo, cell in zip(FIXED_ORDER, odds_cells):
        odds_dict[combo] = cell.get_text(strip=True)

    # oddsセルが足りない時もキーは揃える
    if len(odds_dict) < len(FIXED_ORDER):
        for combo in FIXED_ORDER:
            odds_dict.setdefault(combo, "")

    _set_cache(key, odds_dict)
    return odds_dict
