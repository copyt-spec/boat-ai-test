# engine/trifecta_feature_builder.py

from __future__ import annotations

from typing import Any, Dict, List, Optional
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


# =========================
# leakage guard (odds ban)
# =========================
try:
    from engine.leakage_guard import drop_odds_leakage, find_odds_leak_columns  # type: ignore
except Exception:
    drop_odds_leakage = None  # type: ignore
    find_odds_leak_columns = None  # type: ignore


def _to_float(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for k in ("value", "val", "score", "num", "x", "y"):
            if k in v:
                return _to_float(v.get(k))
        return 0.0
    if isinstance(v, (list, tuple)):
        return _to_float(v[0]) if len(v) > 0 else 0.0

    s = str(v).strip()
    if s == "" or s == "-" or s.lower() in ("nan", "none"):
        return 0.0
    if s.startswith("."):
        s = "0" + s
    try:
        return float(s)
    except Exception:
        if len(s) >= 2 and s[0].upper() in ("F", "L"):
            try:
                return float(s[1:])
            except Exception:
                return 0.0
        return 0.0


def _wind_dir_to_num(s: Any) -> float:
    t = str(s).strip()
    mapping = {
        "北": 1.0, "北東": 2.0, "東": 3.0, "南東": 4.0,
        "南": 5.0, "南西": 6.0, "西": 7.0, "北西": 8.0,
    }
    return mapping.get(t, 0.0)


def _weather_to_num(s: Any) -> float:
    t = str(s).strip()
    mapping = {"晴れ": 1.0, "くもり": 2.0, "曇り": 2.0, "雨": 3.0, "雪": 4.0}
    return mapping.get(t, 0.0)


def _bi_get(bi: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in bi:
            return bi.get(k)
    return None


def _bi_lane_dict(v: Any) -> Dict[str, Any]:
    if isinstance(v, dict):
        return v
    if v is not None:
        return {
            "exhibit_time": getattr(v, "exhibit_time", None),
            "st": getattr(v, "st", None),
            "course": getattr(v, "course", None),
        }
    return {}


def build_trifecta_features(
    race_entries: List[Dict[str, Any]],
    before_info: Optional[Dict[str, Any]] = None,
    venue: str = "",
    race_no: int = 0,
    date: str = "",
) -> pd.DataFrame:
    """
    1レース6艇 -> 120行（combo候補）
    学習と推論で列を合わせる前提

    今回追加:
    - racer_stats_loader 由来の能力列
    - コース適性列
    - 差分特徴
    """
    bi = before_info or {}

    lane_map: Dict[int, Dict[str, Any]] = {}
    for e in race_entries:
        lane = int(_to_float(e.get("lane")))
        if 1 <= lane <= 6:
            lane_map[lane] = e

    def _bi_lane(lane: int) -> Dict[str, Any]:
        v = bi.get(lane)
        d = _bi_lane_dict(v)

        if "exhibit_time" not in d:
            for k in ("exhibit", "ex_time", "exhibitTime"):
                if k in d:
                    d["exhibit_time"] = d.get(k)
                    break

        if "st" not in d:
            for k in ("start_timing", "start", "st_time"):
                if k in d:
                    d["st"] = d.get(k)
                    break

        if "course" not in d:
            for k in ("course_no", "cource", "courseNo"):
                if k in d:
                    d["course"] = d.get(k)
                    break

        return d

    wave_cm = _to_float(_bi_get(bi, "wave_cm", "wave", "waveCm"))
    wind_speed_mps = _to_float(_bi_get(bi, "wind_speed_mps", "wind_speed", "wind", "windSpeed", "wind_speed_ms"))
    wind_dir = _wind_dir_to_num(_bi_get(bi, "wind_dir", "wind_direction", "windDirection", "wind_dir_name"))
    weather = _weather_to_num(_bi_get(bi, "weather", "tenki", "weather_name"))

    rows: List[Dict[str, Any]] = []

    for combo in FIXED_ORDER:
        a_s, b_s, c_s = combo.split("-")
        a, b, c = int(a_s), int(b_s), int(c_s)

        ea = lane_map.get(a, {})
        eb = lane_map.get(b, {})
        ec = lane_map.get(c, {})

        ba = _bi_lane(a)
        bb = _bi_lane(b)
        bc = _bi_lane(c)

        feat = {
            "date": date,
            "venue": venue,
            "race_no": float(race_no),

            "combo": combo,

            "wave_cm": wave_cm,
            "weather": weather,
            "wind_dir": wind_dir,
            "wind_speed_mps": wind_speed_mps,

            # ===== A =====
            "a_lane": float(a),
            "a_boat": _to_float(ea.get("boat")),
            "a_course": _to_float(ba.get("course")),
            "a_exhibit": _to_float(ba.get("exhibit_time")),
            "a_motor": _to_float(ea.get("motor")),
            "a_racer_no": _to_float(ea.get("racer_no")),
            "a_st": _to_float(ba.get("st")),

            "a_racer_win_rate": _to_float(ea.get("racer_win_rate")),
            "a_racer_place_rate": _to_float(ea.get("racer_place_rate")),
            "a_racer_avg_st_base": _to_float(ea.get("racer_avg_st_base")),
            "a_racer_ability_index": _to_float(ea.get("racer_ability_index")),
            "a_racer_prev_ability_index": _to_float(ea.get("racer_prev_ability_index")),
            "a_racer_grade_score": _to_float(ea.get("racer_grade_score")),
            "a_racer_prev_grade_score": _to_float(ea.get("racer_prev_grade_score")),
            "a_racer_age": _to_float(ea.get("racer_age")),
            "a_racer_height": _to_float(ea.get("racer_height")),
            "a_racer_weight": _to_float(ea.get("racer_weight")),

            # ===== B =====
            "b_lane": float(b),
            "b_boat": _to_float(eb.get("boat")),
            "b_course": _to_float(bb.get("course")),
            "b_exhibit": _to_float(bb.get("exhibit_time")),
            "b_motor": _to_float(eb.get("motor")),
            "b_racer_no": _to_float(eb.get("racer_no")),
            "b_st": _to_float(bb.get("st")),

            "b_racer_win_rate": _to_float(eb.get("racer_win_rate")),
            "b_racer_place_rate": _to_float(eb.get("racer_place_rate")),
            "b_racer_avg_st_base": _to_float(eb.get("racer_avg_st_base")),
            "b_racer_ability_index": _to_float(eb.get("racer_ability_index")),
            "b_racer_prev_ability_index": _to_float(eb.get("racer_prev_ability_index")),
            "b_racer_grade_score": _to_float(eb.get("racer_grade_score")),
            "b_racer_prev_grade_score": _to_float(eb.get("racer_prev_grade_score")),
            "b_racer_age": _to_float(eb.get("racer_age")),
            "b_racer_height": _to_float(eb.get("racer_height")),
            "b_racer_weight": _to_float(eb.get("racer_weight")),

            # ===== C =====
            "c_lane": float(c),
            "c_boat": _to_float(ec.get("boat")),
            "c_course": _to_float(bc.get("course")),
            "c_exhibit": _to_float(bc.get("exhibit_time")),
            "c_motor": _to_float(ec.get("motor")),
            "c_racer_no": _to_float(ec.get("racer_no")),
            "c_st": _to_float(bc.get("st")),

            "c_racer_win_rate": _to_float(ec.get("racer_win_rate")),
            "c_racer_place_rate": _to_float(ec.get("racer_place_rate")),
            "c_racer_avg_st_base": _to_float(ec.get("racer_avg_st_base")),
            "c_racer_ability_index": _to_float(ec.get("racer_ability_index")),
            "c_racer_prev_ability_index": _to_float(ec.get("racer_prev_ability_index")),
            "c_racer_grade_score": _to_float(ec.get("racer_grade_score")),
            "c_racer_prev_grade_score": _to_float(ec.get("racer_prev_grade_score")),
            "c_racer_age": _to_float(ec.get("racer_age")),
            "c_racer_height": _to_float(ec.get("racer_height")),
            "c_racer_weight": _to_float(ec.get("racer_weight")),
        }

        # ===== コース適性（その艇が今回の枠で走る前提）=====
        feat["a_racer_course_place_rate"] = _to_float(ea.get(f"racer_course{a}_place_rate"))
        feat["a_racer_course_avg_st"] = _to_float(ea.get(f"racer_course{a}_avg_st"))
        feat["a_racer_course_avg_st_rank"] = _to_float(ea.get(f"racer_course{a}_avg_st_rank"))
        feat["a_racer_course_entry_count"] = _to_float(ea.get(f"racer_course{a}_entry_count"))

        feat["b_racer_course_place_rate"] = _to_float(eb.get(f"racer_course{b}_place_rate"))
        feat["b_racer_course_avg_st"] = _to_float(eb.get(f"racer_course{b}_avg_st"))
        feat["b_racer_course_avg_st_rank"] = _to_float(eb.get(f"racer_course{b}_avg_st_rank"))
        feat["b_racer_course_entry_count"] = _to_float(eb.get(f"racer_course{b}_entry_count"))

        feat["c_racer_course_place_rate"] = _to_float(ec.get(f"racer_course{c}_place_rate"))
        feat["c_racer_course_avg_st"] = _to_float(ec.get(f"racer_course{c}_avg_st"))
        feat["c_racer_course_avg_st_rank"] = _to_float(ec.get(f"racer_course{c}_avg_st_rank"))
        feat["c_racer_course_entry_count"] = _to_float(ec.get(f"racer_course{c}_entry_count"))

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

        rows.append(feat)

    df = pd.DataFrame(rows)

    if drop_odds_leakage is not None:
        df = drop_odds_leakage(df, verbose=False, context="trifecta_feature_builder")

    return df
