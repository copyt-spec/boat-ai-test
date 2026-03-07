# engine/dataset_builder.py

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from engine.marugame_fetcher import fetch_all_entries_once
from engine.beforeinfo_fetcher import fetch_beforeinfo  # もし関数名が違う場合はここだけ合わせる
from engine.feature_builder import build_features
from engine.results_fetcher import fetch_trifecta_result


VENUE_CODE_MARUGAME = 15


def _daterange(start_yyyymmdd: str, end_yyyymmdd: str) -> List[str]:
    s = datetime.strptime(start_yyyymmdd, "%Y%m%d")
    e = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    out = []
    cur = s
    while cur <= e:
        out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out


def _combo_to_class_id(combo: str) -> int:
    """
    "1-2-3" -> 0..119 のIDに変換（順列辞書順）
    """
    lanes = [1, 2, 3, 4, 5, 6]
    combos = []
    for a in lanes:
        for b in lanes:
            for c in lanes:
                if a != b and b != c and a != c:
                    combos.append(f"{a}-{b}-{c}")
    return combos.index(combo)


def _flatten_features(features6: List[Dict[str, Any]], beforeinfo: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    features6: build_features() が返す6艇分のdict
    return: 1レース分を1行にするwide形式
      例: lane1_exhibit_time, lane2_exhibit_time, ...
    """
    row: Dict[str, Any] = {}

    # 6艇分
    for f in features6:
        lane = int(f.get("lane", 0))
        if lane < 1 or lane > 6:
            continue
        prefix = f"lane{lane}_"
        for k, v in f.items():
            if k == "lane":
                continue
            row[prefix + k] = v

    # レース環境（トップ階層）
    if isinstance(beforeinfo, dict):
        # 取得できていれば数値化済みで入る想定（feature_builder側でwind_speed/temperatureを持つ）
        row["wind_speed"] = beforeinfo.get("wind_speed")
        row["wind_direction"] = beforeinfo.get("wind_direction")
        row["weather"] = beforeinfo.get("weather")

    return row


@dataclass
class BuildDatasetConfig:
    start_date: str  # "YYYYMMDD"
    end_date: str    # "YYYYMMDD"
    out_csv_path: str
    # 直前情報を含めるか（将来の学習用。過去日だと取れない日が多い可能性）
    with_beforeinfo: bool = False


def build_dataset(cfg: BuildDatasetConfig) -> None:
    """
    期間内の丸亀レースを走査して、学習用CSVを作る。
    - 出走表（entries）
    - （任意）直前情報
    - 結果（3連単）＝ラベル
    を揃えたレースのみ出力。
    """
    dates = _daterange(cfg.start_date, cfg.end_date)

    # CSVヘッダは可変なので、いったん行を貯めてからヘッダ確定でも良いが、
    # ここでは “最大限シンプル” に、都度キーを集めて最後に書く。
    rows: List[Dict[str, Any]] = []

    for date in dates:
        # 出走表（12R分一括）
        try:
            all_entries = fetch_all_entries_once(date)
        except Exception:
            continue

        # fetch_all_entries_onceがlist（全レース混在）想定
        if not isinstance(all_entries, list) or not all_entries:
            continue

        for race_no in range(1, 13):
            race_entries = [e for e in all_entries if isinstance(e, dict) and e.get("race_no") == race_no]
            if len(race_entries) != 6:
                continue

            # 結果（ラベル）
            result_combo = fetch_trifecta_result(VENUE_CODE_MARUGAME, date, race_no)
            if not result_combo:
                # 未確定 or 取得失敗
                continue

            # 直前情報（任意）
            beforeinfo = None
            if cfg.with_beforeinfo:
                try:
                    beforeinfo = fetch_beforeinfo(VENUE_CODE_MARUGAME, date, race_no)  # 関数名が違うなら合わせる
                except Exception:
                    beforeinfo = None

            # 特徴量（6艇）
            features6 = build_features(race_entries, before_info=beforeinfo)

            # wide化
            X = _flatten_features(features6, beforeinfo=beforeinfo)

            # 基本情報 + ラベル
            row = {
                "date": date,
                "race_no": race_no,
                "y_combo": result_combo,
                "y_class": _combo_to_class_id(result_combo),
            }
            row.update(X)

            rows.append(row)

    if not rows:
        # 何も取れなかった
        with open(cfg.out_csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    # ヘッダ確定（全行のキーの和集合）
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    header = sorted(all_keys)

    # 書き出し
    with open(cfg.out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
