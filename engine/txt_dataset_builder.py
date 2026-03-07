# engine/txt_dataset_builder.py

from engine.txt_race_parser import parse_startk_multi_venue_txt

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from engine.txt_race_parser import parse_startk_multi_venue_txt, RaceRecord


def _combo_to_class_id(combo: str) -> int:
    lanes = [1, 2, 3, 4, 5, 6]
    combos = []
    for a in lanes:
        for b in lanes:
            for c in lanes:
                if a != b and b != c and a != c:
                    combos.append(f"{a}-{b}-{c}")
    return combos.index(combo)


def _date_from_filename(filename: str) -> Optional[str]:
    # 例: K250301.TXT -> 20250301 （20YY仮定）
    m = re.search(r"([0-9]{2})([0-9]{2})([0-9]{2})", filename)
    if not m:
        return None
    yy = int(m.group(1))
    mm = int(m.group(2))
    dd = int(m.group(3))
    yyyy = 2000 + yy
    return f"{yyyy:04d}{mm:02d}{dd:02d}"


def _wide_row_from_record(rec: RaceRecord) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    row["date"] = rec.meta.date
    row["venue"] = rec.meta.venue
    row["race_no"] = rec.meta.race_no

    row["weather"] = rec.meta.weather
    row["wind_dir"] = rec.meta.wind_dir
    row["wind_speed_mps"] = rec.meta.wind_speed_mps
    row["wave_cm"] = rec.meta.wave_cm

    for b in rec.boats:
        p = f"lane{b.lane}_"
        row[p + "racer_no"] = b.racer_no
        row[p + "motor"] = b.motor
        row[p + "boat"] = b.boat
        row[p + "exhibit"] = b.exhibit
        row[p + "course"] = b.course
        row[p + "st"] = b.st
        row[p + "finish"] = b.finish

    row["y_combo"] = rec.y_combo
    row["y_class"] = _combo_to_class_id(rec.y_combo) if rec.y_combo else None
    row["trifecta_payout"] = rec.trifecta_payout

    return row


@dataclass
class TxtDatasetConfig:
    raw_txt_dir: str
    out_csv_path: str
    keep_unlabeled: bool = False
    verbose: bool = True


def build_dataset_from_txt(cfg: TxtDatasetConfig) -> None:
    all_rows: List[Dict[str, Any]] = []

    files = [fn for fn in sorted(os.listdir(cfg.raw_txt_dir)) if fn.lower().endswith(".txt")]
    if cfg.verbose:
        print("TXT files:", len(files))

    for fn in files:
        path = os.path.join(cfg.raw_txt_dir, fn)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        fallback_date = _date_from_filename(fn)

        # ★複数会場対応パース
        records = parse_startk_multi_venue_txt(text, fallback_date=fallback_date)

        total = 0
        labeled = 0
        kept = 0

        for rec in records:
            total += 1
            if (not cfg.keep_unlabeled) and (not rec.y_combo):
                continue
            kept += 1
            if rec.y_combo:
                labeled += 1
            all_rows.append(_wide_row_from_record(rec))

        if cfg.verbose:
            print(f"[{fn}] total_races={total} labeled={labeled} kept={kept}")

    if not all_rows:
        os.makedirs(os.path.dirname(cfg.out_csv_path), exist_ok=True)
        with open(cfg.out_csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        if cfg.verbose:
            print("NO ROWS. output empty:", cfg.out_csv_path)
        return

    keys = set()
    for r in all_rows:
        keys.update(r.keys())
    header = sorted(keys)

    os.makedirs(os.path.dirname(cfg.out_csv_path), exist_ok=True)
    with open(cfg.out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    if cfg.verbose:
        print("DONE rows:", len(all_rows), "->", cfg.out_csv_path)
