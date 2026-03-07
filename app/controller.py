# app/controller.py

from __future__ import annotations

from typing import Any, Dict, List

from engine.marugame_fetcher import fetch_all_entries_once as fetch_all_marugame_entries_once
from engine.toda_fetcher import fetch_all_toda_entries_once

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

    # ===== 丸亀：初期表示（12R出走表） =====
    def get_all_entries(self, date: str) -> List[Dict[str, Any]]:
        return fetch_all_marugame_entries_once(date)

    # ===== 戸田：初期表示（12R出走表） =====
    def get_all_entries_toda(self, date: str) -> List[Dict[str, Any]]:
        return fetch_all_toda_entries_once(date)

    # ===== odds: grouped_odds 形式に変換（UI固定のため形は絶対維持）=====
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

    # ===== 丸亀：オッズ（押下時のみ）=====
    def get_odds_only(self, race_no: int, date: str) -> Dict[str, Any]:
        raw_odds = fetch_odds(race_no, date, venue_code=JCD_MARUGAME)
        return self._group_odds(raw_odds)

    # ===== 戸田：オッズ（押下時のみ）=====
    def get_odds_only_toda(self, race_no: int, date: str) -> Dict[str, Any]:
        raw_odds = fetch_odds(race_no, date, venue_code=JCD_TODA)
        return self._group_odds(raw_odds)

    # ===== 丸亀：直前（押下時のみ）=====
    def get_beforeinfo_only(self, race_no: int, date: str) -> Dict[str, Any]:
        return fetch_beforeinfo(race_no, date)

    # ===== 戸田：直前（押下時のみ）=====
    def get_beforeinfo_only_toda(self, race_no: int, date: str) -> Dict[str, Any]:
        return fetch_beforeinfo_venue(race_no, date, venue_code=JCD_TODA)

    # ===== 押下時に racelist で motor/boat を補完 =====
    def enrich_entries_marugame(self, entries: List[Dict[str, Any]], date: str, race_no: int) -> List[Dict[str, Any]]:
        return enrich_entries_with_racelist(entries, date=date, race_no=race_no, venue_code=JCD_MARUGAME)

    def enrich_entries_toda(self, entries: List[Dict[str, Any]], date: str, race_no: int) -> List[Dict[str, Any]]:
        return enrich_entries_with_racelist(entries, date=date, race_no=race_no, venue_code=JCD_TODA)
