# app/controller.py
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from engine.marugame_fetcher import fetch_all_entries_once as fetch_all_marugame_entries_once
from engine.toda_fetcher import (
    fetch_all_toda_entries_once,
    fetch_toda_racelist,
)

from engine.odds_fetcher import fetch_odds

# 既存（丸亀固定版）は絶対に壊さない
from engine.beforeinfo_fetcher import fetch_beforeinfo

# 追加（戸田/他会場用の可変版）
from engine.beforeinfo_fetcher_venue import fetch_beforeinfo_venue

# 追加（racelist から motor/boat を埋める）
from engine.racelist_enricher import enrich_entries_with_racelist


JCD_MARUGAME = 15
JCD_TODA = 2


class RaceController:
    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._ttl_seconds = 30

    # =========================
    # internal cache
    # =========================
    def _cache_get(self, key: str) -> Optional[Any]:
        item = self._cache.get(key)
        if not item:
            return None

        ts, value = item
        if time.time() - ts > self._ttl_seconds:
            self._cache.pop(key, None)
            return None

        return value

    def _cache_set(self, key: str, value: Any) -> Any:
        self._cache[key] = (time.time(), value)
        return value

    # =========================
    # marugame
    # =========================
    def get_all_entries(self, date: str) -> List[Dict[str, Any]]:
        key = f"all_marugame_{date}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        rows = fetch_all_marugame_entries_once(date)
        return self._cache_set(key, rows)

    def get_entries_marugame_race(self, date: str, race_no: int) -> List[Dict[str, Any]]:
        """
        丸亀側に1R専用fetcherが無い前提。
        既存の12R一括取得を使って、その中から1Rだけ抜く。
        ただし controller 内キャッシュで2回目以降は軽い。
        """
        key = f"race_marugame_{date}_{race_no}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        all_rows = self.get_all_entries(date)
        rows = [dict(x) for x in all_rows if int(x.get("race_no", 0)) == int(race_no)]
        return self._cache_set(key, rows)

    # =========================
    # toda
    # =========================
    def get_all_entries_toda(self, date: str) -> List[Dict[str, Any]]:
        key = f"all_toda_{date}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        rows = fetch_all_toda_entries_once(date)
        return self._cache_set(key, rows)

    def get_entries_toda_race(self, date: str, race_no: int) -> List[Dict[str, Any]]:
        """
        戸田は 1R専用fetcher を使う
        """
        key = f"race_toda_{date}_{race_no}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        rows = fetch_toda_racelist(race_no, date)
        return self._cache_set(key, rows)

    # =========================
    # odds
    # =========================
    def _group_odds(self, raw_odds: Dict[str, Any]) -> Dict[str, Any]:
        grouped = {i: {} for i in range(1, 7)}
        numeric_values = []

        for combo, odd in raw_odds.items():
            parts = str(combo).split("-")
            if len(parts) != 3:
                continue

            try:
                first = int(parts[0])
                second = int(parts[1])
                third = int(parts[2])
            except ValueError:
                continue

            grouped[first][(second, third)] = odd

            try:
                numeric_values.append(float(odd))
            except Exception:
                pass

        min_val = min(numeric_values) if numeric_values else 0.0
        max_val = max(numeric_values) if numeric_values else 0.0

        return {"data": grouped, "min": float(min_val), "max": float(max_val)}

    def get_odds_only(self, race_no: int, date: str) -> Dict[str, Any]:
        key = f"odds_marugame_{date}_{race_no}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        raw_odds = fetch_odds(race_no, date, venue_code=JCD_MARUGAME)
        return self._cache_set(key, self._group_odds(raw_odds))

    def get_odds_only_toda(self, race_no: int, date: str) -> Dict[str, Any]:
        key = f"odds_toda_{date}_{race_no}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        raw_odds = fetch_odds(race_no, date, venue_code=JCD_TODA)
        return self._cache_set(key, self._group_odds(raw_odds))

    # =========================
    # beforeinfo
    # =========================
    def get_beforeinfo_only(self, race_no: int, date: str) -> Dict[str, Any]:
        key = f"before_marugame_{date}_{race_no}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        rows = fetch_beforeinfo(race_no, date)
        return self._cache_set(key, rows)

    def get_beforeinfo_only_toda(self, race_no: int, date: str) -> Dict[str, Any]:
        key = f"before_toda_{date}_{race_no}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        rows = fetch_beforeinfo_venue(race_no, date, venue_code=JCD_TODA)
        return self._cache_set(key, rows)

    # =========================
    # racelist enrich
    # =========================
    def enrich_entries_marugame(self, entries: List[Dict[str, Any]], date: str, race_no: int) -> List[Dict[str, Any]]:
        key = f"enrich_marugame_{date}_{race_no}"
        cached = self._cache_get(key)
        if cached is not None:
            return [dict(x) for x in cached]

        rows = enrich_entries_with_racelist(
            entries,
            date=date,
            race_no=race_no,
            venue_code=JCD_MARUGAME,
        )
        self._cache_set(key, [dict(x) for x in rows])
        return rows

    def enrich_entries_toda(self, entries: List[Dict[str, Any]], date: str, race_no: int) -> List[Dict[str, Any]]:
        key = f"enrich_toda_{date}_{race_no}"
        cached = self._cache_get(key)
        if cached is not None:
            return [dict(x) for x in cached]

        rows = enrich_entries_with_racelist(
            entries,
            date=date,
            race_no=race_no,
            venue_code=JCD_TODA,
        )
        self._cache_set(key, [dict(x) for x in rows])
        return rows
