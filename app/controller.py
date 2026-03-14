# app/controller.py
from __future__ import annotations

from typing import Any, Dict, List

from engine.marugame_fetcher import fetch_all_entries_once as fetch_all_marugame_entries_once
from engine.toda_fetcher import fetch_all_toda_entries_once, fetch_toda_racelist
from engine.odds_fetcher import fetch_odds
from engine.beforeinfo_fetcher import fetch_beforeinfo
from engine.beforeinfo_fetcher_venue import fetch_beforeinfo_venue
from engine.racelist_enricher import enrich_entries_with_racelist

JCD_MARUGAME = 15
JCD_TODA = 2


class RaceController:
    # ===== 一覧用 =====
    def get_all_entries(self, date: str) -> List[Dict[str, Any]]:
        return fetch_all_marugame_entries_once(date)

    def get_all_entries_toda(self, date: str) -> List[Dict[str, Any]]:
        return fetch_all_toda_entries_once(date)

    # ===== 1R詳細用 =====
    def get_entries_race(self, date: str, race_no: int) -> List[Dict[str, Any]]:
        all_entries = fetch_all_marugame_entries_once(date)
        out: List[Dict[str, Any]] = []

        for row in all_entries:
            try:
                if int(row.get("race_no", 0)) == int(race_no):
                    out.append(dict(row))
            except Exception:
                continue

        out.sort(key=lambda x: int(x.get("lane", 0)))
        return out

    def get_entries_toda_race(self, date: str, race_no: int) -> List[Dict[str, Any]]:
        rows = fetch_toda_racelist(race_no, date)
        return [dict(x) for x in rows]

    # ===== odds grouped形式 =====
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

    # ===== odds =====
    def get_odds_only(self, race_no: int, date: str) -> Dict[str, Any]:
        raw_odds = fetch_odds(race_no, date, venue_code=JCD_MARUGAME)
        return self._group_odds(raw_odds)

    def get_odds_only_toda(self, race_no: int, date: str) -> Dict[str, Any]:
        raw_odds = fetch_odds(race_no, date, venue_code=JCD_TODA)
        return self._group_odds(raw_odds)

    # ===== beforeinfo =====
    def get_beforeinfo_only(self, race_no: int, date: str) -> Dict[str, Any]:
        return fetch_beforeinfo(race_no, date)

    def get_beforeinfo_only_toda(self, race_no: int, date: str) -> Dict[str, Any]:
        return fetch_beforeinfo_venue(race_no, date, venue_code=JCD_TODA)

    # ===== motor/boat enrich =====
    def enrich_entries_marugame(
        self,
        entries: List[Dict[str, Any]],
        date: str,
        race_no: int,
    ) -> List[Dict[str, Any]]:
        return enrich_entries_with_racelist(
            entries,
            date=date,
            race_no=race_no,
            venue_code=JCD_MARUGAME,
        )

    def enrich_entries_toda(
        self,
        entries: List[Dict[str, Any]],
        date: str,
        race_no: int,
    ) -> List[Dict[str, Any]]:
        return enrich_entries_with_racelist(
            entries,
            date=date,
            race_no=race_no,
            venue_code=JCD_TODA,
        )
