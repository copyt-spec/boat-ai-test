# scripts/build_trifecta_train_features.py
from __future__ import annotations

import os
import csv
from typing import Any, Dict, List, Optional, Tuple

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
    # ".12" みたいな ST も float() でOK
    try:
        return float(s)
    except Exception:
        return 0.0


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


def _lane_field(row: Dict[str, Any], lane: int, key: str) -> Any:
    return row.get(f"lane{lane}_{key}")


def _race_lane_values(row: Dict[str, Any], key: str) -> Dict[int, float]:
    """lane1_key..lane6_key を lane->float にまとめる"""
    out: Dict[int, float] = {}
    for ln in range(1, 7):
        out[ln] = _to_float(_lane_field(row, ln, key))
    return out


def _rank_asc(values: Dict[int, float]) -> Dict[int, float]:
    """
    小さいほど良い（ST, exhibit_time 等）想定の順位
    rank: 1..6（同値は安定に並ぶ）
    """
    items = sorted(values.items(), key=lambda kv: (kv[1], kv[0]))
    out: Dict[int, float] = {}
    r = 1
    for ln, _v in items:
        out[ln] = float(r)
        r += 1
    return out


def _rank_desc(values: Dict[int, float]) -> Dict[int, float]:
    """大きいほど良い（motor, boat など）想定の順位"""
    items = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
    out: Dict[int, float] = {}
    r = 1
    for ln, _v in items:
        out[ln] = float(r)
        r += 1
    return out


def _mean_std(values: Dict[int, float]) -> Tuple[float, float]:
    arr = [float(v) for v in values.values()]
    if not arr:
        return 0.0, 0.0
    m = sum(arr) / len(arr)
    var = sum((x - m) ** 2 for x in arr) / max(1, len(arr))
    sd = var ** 0.5
    return float(m), float(sd)


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

    # ===== レース内の事前情報（相対）を作る =====
    exhibit_by_lane = _race_lane_values(row, "exhibit")
    st_by_lane = _race_lane_values(row, "st")
    course_by_lane = _race_lane_values(row, "course")
    motor_by_lane = _race_lane_values(row, "motor")
    boat_by_lane = _race_lane_values(row, "boat")

    # 小さいほど良い
    exhibit_rank = _rank_asc(exhibit_by_lane)
    st_rank = _rank_asc(st_by_lane)

    # 大きいほど良い（※あなたのデータの motor/boat が “強いほど大きい” 前提）
    motor_rank = _rank_desc(motor_by_lane)
    boat_rank = _rank_desc(boat_by_lane)

    ex_mean, ex_std = _mean_std(exhibit_by_lane)
    st_mean, st_std = _mean_std(st_by_lane)
    mo_mean, mo_std = _mean_std(motor_by_lane)
    bo_mean, bo_std = _mean_std(boat_by_lane)
    co_mean, co_std = _mean_std(course_by_lane)

    def _z(v: float, m: float, sd: float) -> float:
        if sd <= 1e-9:
            return 0.0
        return (v - m) / sd

    # レース全体の荒れやすさっぽい指標
    race_level = {
        "exhibit_mean": ex_mean,
        "exhibit_std": ex_std,
        "st_mean": st_mean,
        "st_std": st_std,
        "motor_mean": mo_mean,
        "motor_std": mo_std,
        "boat_mean": bo_mean,
        "boat_std": bo_std,
        "course_mean": co_mean,
        "course_std": co_std,
    }

    out: List[Dict[str, Any]] = []

    for combo in FIXED_ORDER:
        a_s, b_s, c_s = combo.split("-")
        a, b, c = int(a_s), int(b_s), int(c_s)

        def pack(prefix: str, lane: int) -> Dict[str, Any]:
            boat_v = _to_float(_lane_field(row, lane, "boat"))
            course_v = _to_float(_lane_field(row, lane, "course"))
            exhibit_v = _to_float(_lane_field(row, lane, "exhibit"))
            motor_v = _to_float(_lane_field(row, lane, "motor"))
            racer_no_v = _to_float(_lane_field(row, lane, "racer_no"))
            st_v = _to_float(_lane_field(row, lane, "st"))

            # 枠との差（進入変化を強く効かせる）
            course_delta = course_v - float(lane)
            course_abs_delta = abs(course_delta)

            return {
                f"{prefix}_lane": float(lane),
                f"{prefix}_boat": boat_v,
                f"{prefix}_course": course_v,
                f"{prefix}_exhibit": exhibit_v,
                f"{prefix}_motor": motor_v,
                f"{prefix}_racer_no": racer_no_v,
                f"{prefix}_st": st_v,

                # ===== 追加：事前情報を“相対化”して効かせる =====
                f"{prefix}_exhibit_rank": exhibit_rank.get(lane, 0.0),
                f"{prefix}_st_rank": st_rank.get(lane, 0.0),
                f"{prefix}_motor_rank": motor_rank.get(lane, 0.0),
                f"{prefix}_boat_rank": boat_rank.get(lane, 0.0),

                f"{prefix}_exhibit_diff_mean": exhibit_v - ex_mean,
                f"{prefix}_st_diff_mean": st_v - st_mean,
                f"{prefix}_motor_diff_mean": motor_v - mo_mean,
                f"{prefix}_boat_diff_mean": boat_v - bo_mean,
                f"{prefix}_course_diff_mean": course_v - co_mean,

                f"{prefix}_exhibit_z": _z(exhibit_v, ex_mean, ex_std),
                f"{prefix}_st_z": _z(st_v, st_mean, st_std),
                f"{prefix}_motor_z": _z(motor_v, mo_mean, mo_std),
                f"{prefix}_boat_z": _z(boat_v, bo_mean, bo_std),
                f"{prefix}_course_z": _z(course_v, co_mean, co_std),

                f"{prefix}_course_delta": course_delta,
                f"{prefix}_course_abs_delta": course_abs_delta,
                f"{prefix}_is_course_change": 1.0 if course_abs_delta >= 0.5 else 0.0,
            }

        feat = {
            "date": date,
            "venue": venue,
            "race_no": float(race_no),
            "combo": combo,

            # 気象
            "wave_cm": wave_cm,
            "wind_speed_mps": wind_speed_mps,
            "wind_dir": wind_dir_num,
            "weather": weather_num,
        }

        # レース全体の指標（ここも効く）
        feat.update(race_level)

        feat.update(pack("a", a))
        feat.update(pack("b", b))
        feat.update(pack("c", c))

        # 既存の差分（維持）
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

        # ★追加：相対(rank/z)の差分（事前情報をより効かせる）
        feat.update({
            "ab_exhibit_rank_diff": feat["a_exhibit_rank"] - feat["b_exhibit_rank"],
            "ac_exhibit_rank_diff": feat["a_exhibit_rank"] - feat["c_exhibit_rank"],
            "bc_exhibit_rank_diff": feat["b_exhibit_rank"] - feat["c_exhibit_rank"],

            "ab_st_rank_diff": feat["a_st_rank"] - feat["b_st_rank"],
            "ac_st_rank_diff": feat["a_st_rank"] - feat["c_st_rank"],
            "bc_st_rank_diff": feat["b_st_rank"] - feat["c_st_rank"],

            "ab_course_abs_delta_diff": feat["a_course_abs_delta"] - feat["b_course_abs_delta"],
            "ac_course_abs_delta_diff": feat["a_course_abs_delta"] - feat["c_course_abs_delta"],
            "bc_course_abs_delta_diff": feat["b_course_abs_delta"] - feat["c_course_abs_delta"],
        })

        feat["y_combo"] = y_combo
        out.append(feat)

    return out


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing: {INPUT_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    first = True
    written = 0

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f_out:
        writer: Optional[csv.DictWriter] = None

        for chunk in pd.read_csv(INPUT_PATH, chunksize=2000):
            chunk = chunk.fillna("")
            for _, r in chunk.iterrows():
                rows = build_one_race_120rows(r.to_dict())
                if not rows:
                    continue

                df = pd.DataFrame(rows)

                # ★リーク除去（オッズっぽい列が紛れ込んでも消す）
                df = drop_odds_leakage(df, verbose=False, context="build_trifecta_train_features")

                # y_combo / combo は絶対必要
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
