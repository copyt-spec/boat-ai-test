# scripts/build_trifecta_train_features.py
from __future__ import annotations

import os
import sys
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

# ===== プロジェクトルートを import path に追加 =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

# ★ odds_fetcher と同じ固定順（絶対触らない）
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

INPUT_PATH = "data/datasets/startk_dataset.csv"
MASTER_PATH = "data/master/racers_master.csv"
OUTPUT_PATH = "data/datasets/trifecta_train_features.csv"

# ★リーク除去
from engine.leakage_guard import drop_odds_leakage, find_odds_leak_columns


def _to_float(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "" or s == "-" or s.lower() in ("nan", "none"):
        return 0.0
    if s.startswith("."):
        s = "0" + s
    try:
        return float(s)
    except Exception:
        return 0.0


def _to_int(v: Any) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return 0


def _to_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _wind_dir_to_num(w: Any) -> float:
    s = _to_str(w)
    mapping = {
        "北": 1.0, "北東": 2.0, "東": 3.0, "南東": 4.0,
        "南": 5.0, "南西": 6.0, "西": 7.0, "北西": 8.0,
    }
    return mapping.get(s, 0.0)


def _weather_to_num(w: Any) -> float:
    s = _to_str(w)
    mapping = {
        "晴れ": 1.0,
        "くもり": 2.0, "曇り": 2.0,
        "雨": 3.0,
        "雪": 4.0,
    }
    return mapping.get(s, 0.0)


def _grade_to_score(grade: str) -> float:
    g = _to_str(grade).upper()
    mapping = {
        "A1": 4.0,
        "A2": 3.0,
        "B1": 2.0,
        "B2": 1.0,
    }
    return mapping.get(g, 0.0)


def _lane_advantage(lane: int) -> float:
    mapping = {
        1: 3.0,
        2: 2.0,
        3: 1.0,
        4: 0.0,
        5: -1.0,
        6: -2.0,
    }
    return mapping.get(int(lane), 0.0)


def _inside_bias(lane: int) -> float:
    mapping = {
        1: 1.00,
        2: 0.80,
        3: 0.55,
        4: 0.25,
        5: 0.10,
        6: 0.00,
    }
    return mapping.get(int(lane), 0.0)


def _safe_div(a: float, b: float) -> float:
    if abs(b) < 1e-12:
        return 0.0
    return a / b


def _load_racer_master_map() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(MASTER_PATH):
        raise FileNotFoundError(f"Missing: {MASTER_PATH}")

    df = pd.read_csv(MASTER_PATH, dtype=str, encoding="utf-8-sig").fillna("")

    if "racer_no" not in df.columns:
        raise ValueError("racer master must contain racer_no")

    num_cols = [
        "win_rate", "place_rate", "avg_st",
        "prev_ability_index", "ability_index",
        "age", "height", "weight",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for i in range(1, 7):
        for suffix in ("entry_count", "place_rate", "avg_st", "avg_st_rank"):
            c = f"course{i}_{suffix}"
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "grade" not in df.columns:
        df["grade"] = ""
    if "prev_grade" not in df.columns:
        df["prev_grade"] = ""

    df["grade_score"] = df["grade"].map(_grade_to_score).fillna(0.0)
    df["prev_grade_score"] = df["prev_grade"].map(_grade_to_score).fillna(0.0)

    df["racer_no"] = df["racer_no"].astype(str).str.strip().str.zfill(4)
    df = df.drop_duplicates(subset=["racer_no"], keep="last").reset_index(drop=True)

    out: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        out[str(r["racer_no"]).zfill(4)] = r.to_dict()
    return out


def _attach_racer_stats_to_row(row: Dict[str, Any], master_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(row)

    for lane in range(1, 7):
        racer_no_key = f"lane{lane}_racer_no"
        racer_no = _to_str(out.get(racer_no_key)).zfill(4)
        stats = master_map.get(racer_no, {})

        out[f"lane{lane}_racer_win_rate"] = _to_float(stats.get("win_rate"))
        out[f"lane{lane}_racer_place_rate"] = _to_float(stats.get("place_rate"))
        out[f"lane{lane}_racer_avg_st_base"] = _to_float(stats.get("avg_st"))
        out[f"lane{lane}_racer_ability_index"] = _to_float(stats.get("ability_index"))
        out[f"lane{lane}_racer_prev_ability_index"] = _to_float(stats.get("prev_ability_index"))
        out[f"lane{lane}_racer_grade_score"] = _to_float(stats.get("grade_score"))
        out[f"lane{lane}_racer_prev_grade_score"] = _to_float(stats.get("prev_grade_score"))
        out[f"lane{lane}_racer_age"] = _to_float(stats.get("age"))
        out[f"lane{lane}_racer_height"] = _to_float(stats.get("height"))
        out[f"lane{lane}_racer_weight"] = _to_float(stats.get("weight"))

        for c in range(1, 7):
            out[f"lane{lane}_racer_course{c}_entry_count"] = _to_float(stats.get(f"course{c}_entry_count"))
            out[f"lane{lane}_racer_course{c}_place_rate"] = _to_float(stats.get(f"course{c}_place_rate"))
            out[f"lane{lane}_racer_course{c}_avg_st"] = _to_float(stats.get(f"course{c}_avg_st"))
            out[f"lane{lane}_racer_course{c}_avg_st_rank"] = _to_float(stats.get(f"course{c}_avg_st_rank"))

    return out


def _lane_field(row: Dict[str, Any], lane: int, key: str) -> Any:
    return row.get(f"lane{lane}_{key}")


def _pack_lane(row: Dict[str, Any], lane: int, prefix: str) -> Dict[str, float]:
    motor = _to_float(_lane_field(row, lane, "motor"))
    boat = _to_float(_lane_field(row, lane, "boat"))
    exhibit = _to_float(_lane_field(row, lane, "exhibit"))
    st = _to_float(_lane_field(row, lane, "st"))
    course = _to_float(_lane_field(row, lane, "course"))

    racer_win_rate = _to_float(_lane_field(row, lane, "racer_win_rate"))
    racer_place_rate = _to_float(_lane_field(row, lane, "racer_place_rate"))
    racer_avg_st_base = _to_float(_lane_field(row, lane, "racer_avg_st_base"))
    racer_ability_index = _to_float(_lane_field(row, lane, "racer_ability_index"))
    racer_prev_ability_index = _to_float(_lane_field(row, lane, "racer_prev_ability_index"))
    racer_grade_score = _to_float(_lane_field(row, lane, "racer_grade_score"))
    racer_prev_grade_score = _to_float(_lane_field(row, lane, "racer_prev_grade_score"))
    racer_age = _to_float(_lane_field(row, lane, "racer_age"))
    racer_height = _to_float(_lane_field(row, lane, "racer_height"))
    racer_weight = _to_float(_lane_field(row, lane, "racer_weight"))

    racer_course_place_rate = _to_float(_lane_field(row, lane, f"racer_course{lane}_place_rate"))
    racer_course_avg_st = _to_float(_lane_field(row, lane, f"racer_course{lane}_avg_st"))
    racer_course_avg_st_rank = _to_float(_lane_field(row, lane, f"racer_course{lane}_avg_st_rank"))
    racer_course_entry_count = _to_float(_lane_field(row, lane, f"racer_course{lane}_entry_count"))

    lane_power = (
        1.8 * _lane_advantage(lane)
        + 0.55 * racer_win_rate
        + 0.08 * racer_ability_index
        + 0.07 * racer_course_place_rate
        + 0.03 * racer_grade_score
        - 6.0 * racer_avg_st_base
        - 4.5 * racer_course_avg_st
        - 0.8 * exhibit
    )

    motor_power = motor
    boat_power = boat
    start_power = -st
    exhibit_power = -exhibit

    one_head_score = (
        1.3 * _inside_bias(lane)
        + 0.45 * racer_win_rate
        + 0.08 * racer_ability_index
        + 0.06 * racer_course_place_rate
        + 0.04 * racer_grade_score
        + 0.015 * motor_power
        - 5.5 * racer_avg_st_base
        - 2.8 * st
        - 0.6 * exhibit
    )

    return {
        f"{prefix}_lane": float(lane),
        f"{prefix}_boat": boat,
        f"{prefix}_course": course,
        f"{prefix}_exhibit": exhibit,
        f"{prefix}_motor": motor,
        f"{prefix}_racer_no": _to_float(_lane_field(row, lane, "racer_no")),
        f"{prefix}_st": st,

        f"{prefix}_lane_advantage": _lane_advantage(lane),
        f"{prefix}_inside_bias": _inside_bias(lane),
        f"{prefix}_is_inner": 1.0 if lane <= 3 else 0.0,
        f"{prefix}_is_outer": 1.0 if lane >= 4 else 0.0,

        f"{prefix}_racer_win_rate": racer_win_rate,
        f"{prefix}_racer_place_rate": racer_place_rate,
        f"{prefix}_racer_avg_st_base": racer_avg_st_base,
        f"{prefix}_racer_ability_index": racer_ability_index,
        f"{prefix}_racer_prev_ability_index": racer_prev_ability_index,
        f"{prefix}_racer_grade_score": racer_grade_score,
        f"{prefix}_racer_prev_grade_score": racer_prev_grade_score,
        f"{prefix}_racer_age": racer_age,
        f"{prefix}_racer_height": racer_height,
        f"{prefix}_racer_weight": racer_weight,

        f"{prefix}_racer_course_place_rate": racer_course_place_rate,
        f"{prefix}_racer_course_avg_st": racer_course_avg_st,
        f"{prefix}_racer_course_avg_st_rank": racer_course_avg_st_rank,
        f"{prefix}_racer_course_entry_count": racer_course_entry_count,

        f"{prefix}_motor_power": motor_power,
        f"{prefix}_boat_power": boat_power,
        f"{prefix}_start_power": start_power,
        f"{prefix}_exhibit_power": exhibit_power,
        f"{prefix}_lane_power": lane_power,
        f"{prefix}_one_head_score": one_head_score,
    }


def build_one_race_120rows(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    y_combo = _to_str(row.get("y_combo"))
    if not y_combo:
        return []

    date = _to_str(row.get("date"))
    venue = _to_str(row.get("venue"))
    race_no = int(_to_float(row.get("race_no")))

    wave_cm = _to_float(row.get("wave_cm"))
    wind_speed_mps = _to_float(row.get("wind_speed_mps"))
    wind_dir_num = _wind_dir_to_num(row.get("wind_dir"))
    weather_num = _weather_to_num(row.get("weather"))

    out: List[Dict[str, Any]] = []

    for combo in FIXED_ORDER:
        a_s, b_s, c_s = combo.split("-")
        a, b, c = int(a_s), int(b_s), int(c_s)

        feat = {
            "date": date,
            "venue": venue,
            "race_no": float(race_no),
            "combo": combo,
            "wave_cm": wave_cm,
            "wind_speed_mps": wind_speed_mps,
            "wind_dir": wind_dir_num,
            "weather": weather_num,
        }

        feat.update(_pack_lane(row, a, "a"))
        feat.update(_pack_lane(row, b, "b"))
        feat.update(_pack_lane(row, c, "c"))

        # ===== 既存差分 =====
        feat.update({
            "ab_exhibit_diff": feat["a_exhibit"] - feat["b_exhibit"],
            "ac_exhibit_diff": feat["a_exhibit"] - feat["c_exhibit"],
            "bc_exhibit_diff": feat["b_exhibit"] - feat["c_exhibit"],

            "ab_st_diff": feat["a_st"] - feat["b_st"],
            "ac_st_diff": feat["a_st"] - feat["c_st"],
            "bc_st_diff": feat["b_st"] - feat["c_st"],

            "ab_course_diff": feat["a_course"] - feat["b_course"],
            "ac_course_diff": feat["a_course"] - feat["c_course"],
            "bc_course_diff": feat["b_course"] - feat["c_course"],

            "ab_motor_diff": feat["a_motor"] - feat["b_motor"],
            "ac_motor_diff": feat["a_motor"] - feat["c_motor"],
            "bc_motor_diff": feat["b_motor"] - feat["c_motor"],
        })

        # ===== 能力差分 =====
        feat.update({
            "ab_racer_win_rate_diff": feat["a_racer_win_rate"] - feat["b_racer_win_rate"],
            "ac_racer_win_rate_diff": feat["a_racer_win_rate"] - feat["c_racer_win_rate"],
            "bc_racer_win_rate_diff": feat["b_racer_win_rate"] - feat["c_racer_win_rate"],

            "ab_racer_place_rate_diff": feat["a_racer_place_rate"] - feat["b_racer_place_rate"],
            "ac_racer_place_rate_diff": feat["a_racer_place_rate"] - feat["c_racer_place_rate"],
            "bc_racer_place_rate_diff": feat["b_racer_place_rate"] - feat["c_racer_place_rate"],

            "ab_racer_avg_st_base_diff": feat["a_racer_avg_st_base"] - feat["b_racer_avg_st_base"],
            "ac_racer_avg_st_base_diff": feat["a_racer_avg_st_base"] - feat["c_racer_avg_st_base"],
            "bc_racer_avg_st_base_diff": feat["b_racer_avg_st_base"] - feat["c_racer_avg_st_base"],

            "ab_racer_ability_index_diff": feat["a_racer_ability_index"] - feat["b_racer_ability_index"],
            "ac_racer_ability_index_diff": feat["a_racer_ability_index"] - feat["c_racer_ability_index"],
            "bc_racer_ability_index_diff": feat["b_racer_ability_index"] - feat["c_racer_ability_index"],

            "ab_racer_grade_score_diff": feat["a_racer_grade_score"] - feat["b_racer_grade_score"],
            "ac_racer_grade_score_diff": feat["a_racer_grade_score"] - feat["c_racer_grade_score"],
            "bc_racer_grade_score_diff": feat["b_racer_grade_score"] - feat["c_racer_grade_score"],

            "ab_racer_course_place_rate_diff": feat["a_racer_course_place_rate"] - feat["b_racer_course_place_rate"],
            "ac_racer_course_place_rate_diff": feat["a_racer_course_place_rate"] - feat["c_racer_course_place_rate"],
            "bc_racer_course_place_rate_diff": feat["b_racer_course_place_rate"] - feat["c_racer_course_place_rate"],

            "ab_racer_course_avg_st_diff": feat["a_racer_course_avg_st"] - feat["b_racer_course_avg_st"],
            "ac_racer_course_avg_st_diff": feat["a_racer_course_avg_st"] - feat["c_racer_course_avg_st"],
            "bc_racer_course_avg_st_diff": feat["b_racer_course_avg_st"] - feat["c_racer_course_avg_st"],

            "ab_racer_course_avg_st_rank_diff": feat["a_racer_course_avg_st_rank"] - feat["b_racer_course_avg_st_rank"],
            "ac_racer_course_avg_st_rank_diff": feat["a_racer_course_avg_st_rank"] - feat["c_racer_course_avg_st_rank"],
            "bc_racer_course_avg_st_rank_diff": feat["b_racer_course_avg_st_rank"] - feat["c_racer_course_avg_st_rank"],
        })

        # ===== 競艇っぽい強化特徴 =====
        feat.update({
            "ab_lane_advantage_diff": feat["a_lane_advantage"] - feat["b_lane_advantage"],
            "ac_lane_advantage_diff": feat["a_lane_advantage"] - feat["c_lane_advantage"],
            "bc_lane_advantage_diff": feat["b_lane_advantage"] - feat["c_lane_advantage"],

            "ab_inside_bias_diff": feat["a_inside_bias"] - feat["b_inside_bias"],
            "ac_inside_bias_diff": feat["a_inside_bias"] - feat["c_inside_bias"],
            "bc_inside_bias_diff": feat["b_inside_bias"] - feat["c_inside_bias"],

            "ab_lane_power_diff": feat["a_lane_power"] - feat["b_lane_power"],
            "ac_lane_power_diff": feat["a_lane_power"] - feat["c_lane_power"],
            "bc_lane_power_diff": feat["b_lane_power"] - feat["c_lane_power"],

            "ab_one_head_score_diff": feat["a_one_head_score"] - feat["b_one_head_score"],
            "ac_one_head_score_diff": feat["a_one_head_score"] - feat["c_one_head_score"],
            "bc_one_head_score_diff": feat["b_one_head_score"] - feat["c_one_head_score"],

            "ab_start_power_diff": feat["a_start_power"] - feat["b_start_power"],
            "ac_start_power_diff": feat["a_start_power"] - feat["c_start_power"],
            "bc_start_power_diff": feat["b_start_power"] - feat["c_start_power"],

            "ab_exhibit_power_diff": feat["a_exhibit_power"] - feat["b_exhibit_power"],
            "ac_exhibit_power_diff": feat["a_exhibit_power"] - feat["c_exhibit_power"],
            "bc_exhibit_power_diff": feat["b_exhibit_power"] - feat["c_exhibit_power"],

            "ab_motor_power_diff": feat["a_motor_power"] - feat["b_motor_power"],
            "ac_motor_power_diff": feat["a_motor_power"] - feat["c_motor_power"],
            "bc_motor_power_diff": feat["b_motor_power"] - feat["c_motor_power"],

            "ab_boat_power_diff": feat["a_boat_power"] - feat["b_boat_power"],
            "ac_boat_power_diff": feat["a_boat_power"] - feat["c_boat_power"],
            "bc_boat_power_diff": feat["b_boat_power"] - feat["c_boat_power"],
        })

        # ===== 合計/平均/比率 =====
        feat.update({
            "abc_win_rate_sum": feat["a_racer_win_rate"] + feat["b_racer_win_rate"] + feat["c_racer_win_rate"],
            "abc_ability_sum": feat["a_racer_ability_index"] + feat["b_racer_ability_index"] + feat["c_racer_ability_index"],
            "abc_lane_power_sum": feat["a_lane_power"] + feat["b_lane_power"] + feat["c_lane_power"],
            "abc_one_head_score_sum": feat["a_one_head_score"] + feat["b_one_head_score"] + feat["c_one_head_score"],
            "abc_motor_sum": feat["a_motor"] + feat["b_motor"] + feat["c_motor"],
            "abc_exhibit_sum": feat["a_exhibit"] + feat["b_exhibit"] + feat["c_exhibit"],
            "abc_st_sum": feat["a_st"] + feat["b_st"] + feat["c_st"],

            "a_share_of_ability": _safe_div(feat["a_racer_ability_index"], feat["a_racer_ability_index"] + feat["b_racer_ability_index"] + feat["c_racer_ability_index"]),
            "b_share_of_ability": _safe_div(feat["b_racer_ability_index"], feat["a_racer_ability_index"] + feat["b_racer_ability_index"] + feat["c_racer_ability_index"]),
            "c_share_of_ability": _safe_div(feat["c_racer_ability_index"], feat["a_racer_ability_index"] + feat["b_racer_ability_index"] + feat["c_racer_ability_index"]),
        })

        # ===== 順位関係っぽいフラグ =====
        feat.update({
            "a_head_is_inner": 1.0 if feat["a_lane"] <= 2 else 0.0,
            "a_head_is_center": 1.0 if feat["a_lane"] in (3, 4) else 0.0,
            "a_head_is_outer": 1.0 if feat["a_lane"] >= 5 else 0.0,

            "a_head_best_winrate": 1.0 if feat["a_racer_win_rate"] >= max(feat["b_racer_win_rate"], feat["c_racer_win_rate"]) else 0.0,
            "a_head_best_ability": 1.0 if feat["a_racer_ability_index"] >= max(feat["b_racer_ability_index"], feat["c_racer_ability_index"]) else 0.0,
            "a_head_best_motor": 1.0 if feat["a_motor"] >= max(feat["b_motor"], feat["c_motor"]) else 0.0,

            "a_head_best_st": 1.0 if feat["a_st"] <= min(feat["b_st"], feat["c_st"]) else 0.0,
            "a_head_best_exhibit": 1.0 if feat["a_exhibit"] <= min(feat["b_exhibit"], feat["c_exhibit"]) else 0.0,

            "b_second_inner_than_c": 1.0 if feat["b_lane"] < feat["c_lane"] else 0.0,
            "b_second_better_winrate_than_c": 1.0 if feat["b_racer_win_rate"] > feat["c_racer_win_rate"] else 0.0,
            "b_second_better_st_than_c": 1.0 if feat["b_st"] < feat["c_st"] else 0.0,
        })

        feat["y_combo"] = y_combo
        out.append(feat)

    return out


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing: {INPUT_PATH}")

    if not os.path.exists(MASTER_PATH):
        raise FileNotFoundError(f"Missing: {MASTER_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    master_map = _load_racer_master_map()

    first = True
    written = 0

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f_out:
        writer: Optional[csv.DictWriter] = None

        for chunk in pd.read_csv(INPUT_PATH, chunksize=2000):
            chunk = chunk.fillna("")

            for _, r in chunk.iterrows():
                base_row = r.to_dict()
                base_row = _attach_racer_stats_to_row(base_row, master_map)

                rows = build_one_race_120rows(base_row)
                if not rows:
                    continue

                df = pd.DataFrame(rows)
                df = drop_odds_leakage(df, verbose=False, context="build_trifecta_train_features")

                if "y_combo" not in df.columns or "combo" not in df.columns:
                    raise RuntimeError("Leakage guard removed required columns. Check leakage_guard.py patterns.")

                rows2 = df.to_dict(orient="records")

                if first:
                    bad = find_odds_leak_columns(df.columns)
                    if bad:
                        print("[WARN] leak-like columns still exist in OUTPUT features:", bad)
                    writer = csv.DictWriter(f_out, fieldnames=list(rows2[0].keys()))
                    writer.writeheader()
                    first = False

                assert writer is not None
                writer.writerows(rows2)
                written += len(rows2)

            print("built_rows:", written)

    print("DONE:", OUTPUT_PATH, "rows=", written)


if __name__ == "__main__":
    main()
