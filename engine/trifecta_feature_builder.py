# engine/trifecta_feature_builder.py

from __future__ import annotations

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


def _rank_asc(values: Dict[int, float]) -> Dict[int, float]:
    # 小さいほど良い（展示/ST）
    items = sorted(values.items(), key=lambda kv: (kv[1], kv[0]))
    out: Dict[int, float] = {}
    r = 1
    for ln, _v in items:
        out[ln] = float(r)
        r += 1
    return out


def _rank_desc(values: Dict[int, float]) -> Dict[int, float]:
    # 大きいほど良い（motor/boat）
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


def _z(v: float, m: float, sd: float) -> float:
    if sd <= 1e-9:
        return 0.0
    return (v - m) / sd


def build_trifecta_features(
    race_entries: List[Dict[str, Any]],
    before_info: Optional[Dict[str, Any]] = None,
    venue: str = "",
    race_no: int = 0,
    date: str = "",
) -> pd.DataFrame:
    """
    1レース6艇 -> 120行（combo候補）
    ★学習側 build_trifecta_train_features.py と列名/意味を一致させる（重要）
    - 展示/ST/進入の “相対” 特徴量を追加（rank / mean差 / z / course_delta）
    - オッズ由来列は混入しても drop（リーク防止）
    """
    bi = before_info or {}

    # race_entries は laneキーがある想定（motor/boat/racer_no はここから）
    lane_entry: Dict[int, Dict[str, Any]] = {}
    for e in race_entries:
        lane = int(_to_float(e.get("lane")))
        if lane <= 0:
            continue
        lane_entry[lane] = e

    # beforeinfo: lane -> dict/obj（キー揺れ吸収込み）
    def _bi_lane(lane: int) -> Dict[str, Any]:
        v = bi.get(lane)
        d = _bi_lane_dict(v)

        # exhibit_time
        if "exhibit_time" not in d:
            for k in ("exhibit", "ex_time", "exhibitTime"):
                if k in d:
                    d["exhibit_time"] = d.get(k)
                    break

        # st
        if "st" not in d:
            for k in ("start_timing", "start", "st_time"):
                if k in d:
                    d["st"] = d.get(k)
                    break

        # course
        if "course" not in d:
            for k in ("course_no", "cource", "courseNo"):
                if k in d:
                    d["course"] = d.get(k)
                    break

        return d

    # ===== 環境（beforeinfo由来。なければ0）=====
    wave_cm = _to_float(_bi_get(bi, "wave_cm", "wave", "waveCm"))
    wind_speed_mps = _to_float(_bi_get(bi, "wind_speed_mps", "wind_speed", "wind", "windSpeed", "wind_speed_ms"))
    wind_dir = _wind_dir_to_num(_bi_get(bi, "wind_dir", "wind_direction", "windDirection", "wind_dir_name"))
    weather = _weather_to_num(_bi_get(bi, "weather", "tenki", "weather_name"))

    # ===== レース内の lane1..6 の数値を作る（相対化のため）=====
    exhibit_by_lane: Dict[int, float] = {}
    st_by_lane: Dict[int, float] = {}
    course_by_lane: Dict[int, float] = {}
    motor_by_lane: Dict[int, float] = {}
    boat_by_lane: Dict[int, float] = {}

    for ln in range(1, 7):
        e = lane_entry.get(ln, {})
        b = _bi_lane(ln)

        boat_by_lane[ln] = _to_float(e.get("boat"))
        motor_by_lane[ln] = _to_float(e.get("motor"))
        exhibit_by_lane[ln] = _to_float(b.get("exhibit_time"))
        st_by_lane[ln] = _to_float(b.get("st"))
        course_by_lane[ln] = _to_float(b.get("course"))

    exhibit_rank = _rank_asc(exhibit_by_lane)
    st_rank = _rank_asc(st_by_lane)
    motor_rank = _rank_desc(motor_by_lane)
    boat_rank = _rank_desc(boat_by_lane)

    ex_mean, ex_std = _mean_std(exhibit_by_lane)
    st_mean, st_std = _mean_std(st_by_lane)
    mo_mean, mo_std = _mean_std(motor_by_lane)
    bo_mean, bo_std = _mean_std(boat_by_lane)
    co_mean, co_std = _mean_std(course_by_lane)

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

    def _pack(prefix: str, lane: int) -> Dict[str, Any]:
        e = lane_entry.get(lane, {})
        b = _bi_lane(lane)

        boat_v = _to_float(e.get("boat"))
        motor_v = _to_float(e.get("motor"))
        racer_no_v = _to_float(e.get("racer_no"))

        exhibit_v = _to_float(b.get("exhibit_time"))
        st_v = _to_float(b.get("st"))
        course_v = _to_float(b.get("course"))

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

            # ===== 相対特徴（学習側と一致）=====
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

    rows: List[Dict[str, Any]] = []

    for combo in FIXED_ORDER:
        a_s, b_s, c_s = combo.split("-")
        a, b, c = int(a_s), int(b_s), int(c_s)

        fa = _pack("a", a)
        fb = _pack("b", b)
        fc = _pack("c", c)

        feat: Dict[str, Any] = {
            "date": date,
            "venue": venue,
            "race_no": float(race_no),
            "combo": combo,

            # 気象
            "wave_cm": wave_cm,
            "weather": weather,
            "wind_dir": wind_dir,
            "wind_speed_mps": wind_speed_mps,
        }
        feat.update(race_level)
        feat.update(fa)
        feat.update(fb)
        feat.update(fc)

        # ===== 既存差分（維持）=====
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

        # ===== 追加：相対(rank/z)の差分（学習側と一致）=====
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

        rows.append(feat)

    df = pd.DataFrame(rows)

    # =========================
    # ★オッズ由来の列が混入しても落とす（リーク防止）
    # =========================
    if drop_odds_leakage is not None:
        df = drop_odds_leakage(df, verbose=False, context="trifecta_feature_builder")

    # debug が欲しければここをON（普段はOFF推奨）
    # if find_odds_leak_columns is not None:
    #     bad = find_odds_leak_columns(df.columns)
    #     if bad:
    #         print(f"[LEAK_DETECTED] trifecta_feature_builder columns={bad}")

    return df
