# app/main.py
from __future__ import annotations

import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# ====== Controller ======
try:
    from app.controller import RaceController
except Exception:
    try:
        from controller import RaceController  # type: ignore
    except Exception:
        RaceController = None  # type: ignore

# ====== optional preinfo fetcher ======
try:
    from engine.preinfo_fetcher import fetch_racelist_preinfo_and_exhibit  # type: ignore
except Exception:
    fetch_racelist_preinfo_and_exhibit = None  # type: ignore

# ====== Racer stats loader ======
try:
    from engine.racer_stats_loader import enrich_entries_with_racer_stats  # type: ignore
except Exception as e:
    enrich_entries_with_racer_stats = None  # type: ignore
    print("[WARN] cannot import engine.racer_stats_loader.enrich_entries_with_racer_stats:", e)

# ====== Feature Builder ======
FEATURE_BUILDER_FUNCS = []
try:
    from engine.trifecta_feature_builder import build_trifecta_features  # type: ignore
    FEATURE_BUILDER_FUNCS.append(("builder_v1", build_trifecta_features))
except Exception as e:
    print("[WARN] cannot import engine.trifecta_feature_builder.build_trifecta_features:", e)

# ====== model bootstrap ======
try:
    from engine.model_bootstrap import ensure_model_ready  # type: ignore
    ensure_model_ready()
except Exception as e:
    print("[WARN] model bootstrap failed:", e)

# ====== AI model loader / EV ======
AI_ENABLED = True
try:
    from engine.model_loader import BoatRaceModel  # type: ignore
except Exception as e:
    AI_ENABLED = False
    BoatRaceModel = None  # type: ignore
    print("[WARN] cannot import engine.model_loader.BoatRaceModel:", e)

try:
    from engine.ev_calculator import calculate_ev  # type: ignore
except Exception:
    calculate_ev = None  # type: ignore

app = Flask(__name__, template_folder="templates", static_folder="static")

VENUE_CODE_MAP: Dict[str, int] = {
    "丸亀": 15,
    "戸田": 2,
}


# =========================
# utils
# =========================
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.floating)):
            v = float(x)
            return v if np.isfinite(v) else default
        s = str(x).strip()
        if s == "":
            return default
        v = float(s)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return default


def _is_debug_request() -> bool:
    return request.args.get("debug", "").strip() in ("1", "true", "True", "yes", "on")


def _today_yyyymmdd_tokyo() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d")


def _get_date_default() -> str:
    return request.args.get("date", "").strip() or _today_yyyymmdd_tokyo()


def _blank_races() -> List[Dict[str, Any]]:
    return [{"race_no": rn, "entries": []} for rn in range(1, 13)]


def _group_entries_by_race(all_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_race: Dict[int, List[Dict[str, Any]]] = {}
    for e in all_entries or []:
        rn = e.get("race_no")
        try:
            rn_i = int(rn)
        except Exception:
            continue
        by_race.setdefault(rn_i, []).append(e)

    races = []
    for rn in range(1, 13):
        races.append({"race_no": rn, "entries": by_race.get(rn, [])})
    return races


def _flatten_grouped_odds(grouped_odds: Any) -> Dict[str, float]:
    odds_map: Dict[str, float] = {}
    if not grouped_odds:
        return odds_map

    if isinstance(grouped_odds, dict) and "data" in grouped_odds and isinstance(grouped_odds["data"], dict):
        data = grouped_odds["data"]
        for a, m in data.items():
            try:
                a_int = int(a)
            except Exception:
                continue
            if not isinstance(m, dict):
                continue
            for bc, ov in m.items():
                try:
                    b = int(bc[0])
                    c = int(bc[1])
                    odds_map[f"{a_int}-{b}-{c}"] = _safe_float(ov, 0.0)
                except Exception:
                    continue
        return odds_map

    if isinstance(grouped_odds, dict):
        for k, v in grouped_odds.items():
            if k == "data":
                continue
            odds_map[str(k)] = _safe_float(v, 0.0)

    return odds_map


def _safe_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if c == "combo":
            continue
        df2[c] = df2[c].replace("", np.nan)
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df2


def _calc_ev_cutoffs(ev_result: Dict[str, float]) -> Tuple[Optional[float], Optional[float]]:
    if not ev_result:
        return None, None
    arr = np.array([float(v) for v in ev_result.values() if np.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return None, None
    return float(np.quantile(arr, 0.90)), float(np.quantile(arr, 0.95))


def _normalize_beforeinfo_dict(beforeinfo_raw: Any) -> Dict[str, Any]:
    if not beforeinfo_raw:
        return {}
    if isinstance(beforeinfo_raw, dict):
        return beforeinfo_raw

    out: Dict[str, Any] = {}
    for k in ("weather", "wind_speed", "wind_direction", "wind_dir", "wave_cm", "wind_speed_mps"):
        if hasattr(beforeinfo_raw, k):
            out[k] = getattr(beforeinfo_raw, k)

    if hasattr(beforeinfo_raw, "lanes"):
        lanes = getattr(beforeinfo_raw, "lanes")
        if isinstance(lanes, dict):
            for ln, v in lanes.items():
                out[ln] = v

    return out


def _pre_info_from_beforeinfo(beforeinfo: Dict[str, Any]) -> Dict[str, Any]:
    if not beforeinfo:
        return {
            "weather": "",
            "wind_dir": "",
            "wind_direction": "",
            "wind_speed": 0.0,
            "wind_speed_mps": 0.0,
            "wave_cm": 0.0,
        }

    wind_dir = str(beforeinfo.get("wind_dir") or beforeinfo.get("wind_direction") or "").strip()
    wind_speed = _safe_float(
        beforeinfo.get("wind_speed_mps", beforeinfo.get("wind_speed", 0.0)),
        0.0,
    )
    wave_cm = _safe_float(beforeinfo.get("wave_cm", beforeinfo.get("wave", 0.0)), 0.0)
    weather = str(beforeinfo.get("weather") or "").strip()

    return {
        "weather": weather,
        "wind_dir": wind_dir,
        "wind_direction": wind_dir,
        "wind_speed": wind_speed,
        "wind_speed_mps": wind_speed,
        "wave_cm": wave_cm,
    }


def _inject_exhibit_and_st_from_beforeinfo(entries: List[Dict[str, Any]], beforeinfo: Dict[str, Any]) -> None:
    if not entries or not beforeinfo:
        return

    for e in entries:
        lane = _to_int(e.get("lane", 0), 0)
        if lane <= 0:
            continue

        info = beforeinfo.get(lane)
        if info is None:
            info = beforeinfo.get(str(lane))
        if not isinstance(info, dict):
            continue

        ex = info.get("exhibit_time")
        if ex is None:
            ex = info.get("exhibit")

        st = info.get("st")
        if st is None:
            st = info.get("start_timing")

        course = info.get("course")
        if course is None:
            course = info.get("course_no")

        if e.get("exhibit") in (None, "", 0, "0") and ex not in (None, "", 0, "0"):
            e["exhibit"] = ex
        if e.get("start_timing") in (None, "", 0, "0") and st not in (None, "", 0, "0"):
            e["start_timing"] = st
        if e.get("course") in (None, "", 0, "0") and course not in (None, "", 0, "0"):
            e["course"] = course


def _inject_preinfo_lane_map(entries: List[Dict[str, Any]], lane_map: Dict[Any, Any]) -> None:
    if not entries or not lane_map:
        return

    for e in entries:
        lane = _to_int(e.get("lane", 0), 0)
        if lane <= 0:
            continue

        info = lane_map.get(lane)
        if info is None:
            info = lane_map.get(str(lane))
        if not info:
            continue

        if isinstance(info, dict):
            ex = info.get("exhibit") or info.get("exhibit_time")
            st = info.get("start_timing") or info.get("st")
            course = info.get("course")
        else:
            ex = getattr(info, "exhibit", None) or getattr(info, "exhibit_time", None)
            st = getattr(info, "start_timing", None) or getattr(info, "st", None)
            course = getattr(info, "course", None)

        if e.get("exhibit") in (None, "", 0, "0") and ex not in (None, "", 0, "0"):
            e["exhibit"] = ex
        if e.get("start_timing") in (None, "", 0, "0") and st not in (None, "", 0, "0"):
            e["start_timing"] = st
        if e.get("course") in (None, "", 0, "0") and course not in (None, "", 0, "0"):
            e["course"] = course


def _enrich_entries_with_racer_stats_safe(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if enrich_entries_with_racer_stats is None:
        return entries
    try:
        return enrich_entries_with_racer_stats(entries)
    except Exception as e:
        if _is_debug_request():
            print("[RACER_STATS_ENRICH_ERROR]", e)
        return entries


def _build_features_120(
    venue_name: str,
    date: str,
    race_no: int,
    entries: List[Dict[str, Any]],
    beforeinfo: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    for name, fn in FEATURE_BUILDER_FUNCS:
        try:
            df = fn(  # type: ignore
                entries,
                before_info=beforeinfo,
                venue=venue_name,
                race_no=race_no,
                date=date,
            )
            if isinstance(df, pd.DataFrame) and ("combo" in df.columns) and len(df) == 120:
                if _is_debug_request():
                    print(f"[DBG] feature_builder_used={name} shape={df.shape}")
                return df
        except Exception as e:
            print("[WARN] feature builder failed:", name, e)

    combos = []
    for a in range(1, 7):
        for b in range(1, 7):
            if b == a:
                continue
            for c in range(1, 7):
                if c in (a, b):
                    continue
                combos.append(f"{a}-{b}-{c}")
    combos = combos[:120]

    bi = beforeinfo or {}
    df = pd.DataFrame(
        {
            "date": [date] * 120,
            "venue": [venue_name] * 120,
            "race_no": [float(race_no)] * 120,
            "combo": combos,
            "wave_cm": [_safe_float(bi.get("wave_cm", 0.0), 0.0)] * 120,
            "weather": [bi.get("weather", "")] * 120,
            "wind_dir": [bi.get("wind_dir", bi.get("wind_direction", ""))] * 120,
            "wind_speed_mps": [_safe_float(bi.get("wind_speed_mps", bi.get("wind_speed", 0.0)), 0.0)] * 120,
        }
    )
    return df


def _uniform_probs_from_features(features_120: pd.DataFrame) -> Dict[str, float]:
    if not isinstance(features_120, pd.DataFrame) or "combo" not in features_120.columns:
        return {}
    combos = [str(x) for x in features_120["combo"].values]
    if not combos:
        return {}
    p = 1.0 / float(len(combos))
    return {c: p for c in combos}


def _debug_print_top10(probabilities: Dict[str, float]) -> None:
    if not probabilities:
        print("[DBG] TOP10 probs: (empty)")
        return
    top = sorted(probabilities.items(), key=lambda kv: float(kv[1]), reverse=True)[:10]
    print("[DBG] TOP10 probs:")
    for c, p in top:
        try:
            pf = float(p)
        except Exception:
            pf = 0.0
        print(f"  {c} {pf}")


def _calc_ai_outputs(
    venue_name: str,
    date: str,
    race_no: int,
    entries: List[Dict[str, Any]],
    grouped_odds: Any,
    pre_info: Optional[Dict[str, Any]] = None,
    beforeinfo: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "features_120": None,
        "probabilities": {},
        "ev_result": {},
        "ev_cutoff_90": None,
        "ev_cutoff_95": None,
        "ai_error": None,
        "pre_info": pre_info or {"weather": "", "wind_dir": "", "wind_speed": 0.0, "wave_cm": 0.0},
        "beforeinfo": beforeinfo or {},
    }

    odds_map = _flatten_grouped_odds(grouped_odds)

    features_120 = _build_features_120(
        venue_name=venue_name,
        date=date,
        race_no=race_no,
        entries=entries,
        beforeinfo=out["beforeinfo"],
    )
    features_120 = _safe_numeric_features(features_120)
    out["features_120"] = features_120

    if (not AI_ENABLED) or (BoatRaceModel is None):
        out["ai_error"] = "[AI_DISABLED]"
        out["probabilities"] = _uniform_probs_from_features(features_120)
        if odds_map:
            out["ev_result"] = {
                k: _safe_float(odds_map.get(k, 0.0), 0.0) * _safe_float(p, 0.0)
                for k, p in out["probabilities"].items()
            }
            out["ev_cutoff_90"], out["ev_cutoff_95"] = _calc_ev_cutoffs(out["ev_result"])
        return out

    try:
        br_model = BoatRaceModel(
            temperature=_safe_float(request.args.get("temp", 1.8), 1.8),
            output_tau=_safe_float(request.args.get("tau", 1.15), 1.15),
            rescue_max=_safe_float(request.args.get("rmax", 0.85), 0.85),
            rescue_mix_cap=_safe_float(request.args.get("rmix", 0.20), 0.20),
            debug=_is_debug_request(),
        )

        probabilities = br_model.predict_proba(features_120)
        probabilities = dict(probabilities) if probabilities else {}

        s = float(sum(_safe_float(v, 0.0) for v in probabilities.values()))
        if s > 0:
            probabilities = {k: _safe_float(v, 0.0) / s for k, v in probabilities.items()}
        else:
            probabilities = _uniform_probs_from_features(features_120)

        out["probabilities"] = probabilities

        if _is_debug_request():
            _debug_print_top10(probabilities)

        ev_result: Dict[str, float] = {}
        if odds_map:
            if calculate_ev is not None:
                try:
                    ev_any = calculate_ev(probabilities, grouped_odds)  # type: ignore
                except TypeError:
                    try:
                        ev_any = calculate_ev(probabilities=probabilities, grouped_odds=grouped_odds)  # type: ignore
                    except TypeError:
                        ev_any = calculate_ev(probabilities, odds_map)  # type: ignore

                if isinstance(ev_any, dict):
                    for k, v in ev_any.items():
                        ev_result[str(k)] = _safe_float(v, 0.0)
            else:
                for k, p in probabilities.items():
                    ev_result[k] = _safe_float(odds_map.get(k, 0.0), 0.0) * _safe_float(p, 0.0)

        out["ev_result"] = ev_result
        out["ev_cutoff_90"], out["ev_cutoff_95"] = _calc_ev_cutoffs(ev_result)
        return out

    except Exception as e:
        out["ai_error"] = f"[AI_ERROR] {e}"
        print(out["ai_error"])
        print(traceback.format_exc())

        out["probabilities"] = _uniform_probs_from_features(features_120)

        if _is_debug_request():
            _debug_print_top10(out["probabilities"])

        if odds_map:
            out["ev_result"] = {
                k: _safe_float(odds_map.get(k, 0.0), 0.0) * _safe_float(p, 0.0)
                for k, p in out["probabilities"].items()
            }
            out["ev_cutoff_90"], out["ev_cutoff_95"] = _calc_ev_cutoffs(out["ev_result"])
        return out


# =========================
# render
# =========================
def _render_venue_page(venue_name: str):
    date = _get_date_default()
    race_str = request.args.get("race", "").strip()
    mode = request.args.get("mode", "").strip().lower()

    if RaceController is None:
        return "RaceController import failed. Check app/controller.py", 500

    controller = RaceController()
    venue_code = VENUE_CODE_MAP.get(venue_name, 0)

    grouped_odds = None
    probabilities: Dict[str, float] = {}
    ev_result: Dict[str, float] = {}
    ev_cutoff_90 = None
    ev_cutoff_95 = None
    selected_race = 0

    pre_info: Dict[str, Any] = {
        "weather": "",
        "wind_dir": "",
        "wind_direction": "",
        "wind_speed": 0.0,
        "wind_speed_mps": 0.0,
        "wave_cm": 0.0,
    }
    beforeinfo_for_template: Dict[str, Any] = {}
    beforeinfo_for_builder: Dict[str, Any] = {}

    # =========================================
    # 一覧表示: 12R分まとめて取得
    # =========================================
    if not (mode == "full" and race_str.isdigit()):
        try:
            if venue_name == "戸田":
                all_entries = controller.get_all_entries_toda(date)
            else:
                all_entries = controller.get_all_entries(date)
        except Exception as e:
            return f"get_all_entries failed: {e}", 500

        races = _group_entries_by_race(all_entries)

        return render_template(
            "index.html",
            venue=venue_name,
            date=date,
            races=races,
            selected_race=0,
            grouped_odds=None,
            probabilities={},
            ev_result={},
            ev_cutoff_90=None,
            ev_cutoff_95=None,
            pre_info=pre_info,
            beforeinfo={},
        )

    # =========================================
    # 詳細表示: 1Rだけ取得
    # =========================================
    race_no = int(race_str)
    selected_race = race_no

    try:
        if venue_name == "戸田":
            entries = controller.get_entries_toda_race(date, race_no)
        else:
            entries = controller.get_entries_race(date, race_no)
    except Exception as e:
        return f"get_entries_race failed: {e}", 500

    entries = [dict(x) for x in entries]

    races = _blank_races()
    races[race_no - 1]["entries"] = entries

    try:
        if venue_name == "戸田":
            entries = controller.enrich_entries_toda(entries, date=date, race_no=race_no)
        else:
            entries = controller.enrich_entries_marugame(entries, date=date, race_no=race_no)
    except Exception as e:
        if _is_debug_request():
            print("[ENRICH_ERROR]", e)

    entries = _enrich_entries_with_racer_stats_safe(entries)

    if _is_debug_request() and entries:
        sample_e = entries[0]
        print(
            "[DBG] racer_stats_after_enrich:",
            {
                "lane": sample_e.get("lane"),
                "racer_no": sample_e.get("racer_no"),
                "racer_win_rate": sample_e.get("racer_win_rate"),
                "racer_ability_index": sample_e.get("racer_ability_index"),
                "racer_course1_place_rate": sample_e.get("racer_course1_place_rate"),
            },
        )

    try:
        if venue_name == "戸田":
            bi_raw = controller.get_beforeinfo_only_toda(race_no=race_no, date=date)
        else:
            bi_raw = controller.get_beforeinfo_only(race_no=race_no, date=date)

        beforeinfo_for_template = _normalize_beforeinfo_dict(bi_raw)
    except Exception as e:
        beforeinfo_for_template = {}
        if _is_debug_request():
            print("[BEFOREINFO_ERROR]", e)

    _inject_exhibit_and_st_from_beforeinfo(entries, beforeinfo_for_template)
    beforeinfo_for_builder = beforeinfo_for_template
    pre_info = _pre_info_from_beforeinfo(beforeinfo_for_template)

    need_preinfo_fallback = False
    if not pre_info.get("weather"):
        need_preinfo_fallback = True
    if not pre_info.get("wind_dir"):
        need_preinfo_fallback = True
    if all((e.get("exhibit") in (None, "", 0, "0")) for e in entries):
        need_preinfo_fallback = True

    if need_preinfo_fallback and fetch_racelist_preinfo_and_exhibit is not None and venue_code:
        try:
            pre, lane_map = fetch_racelist_preinfo_and_exhibit(  # type: ignore
                venue_code=venue_code,
                date=date,
                race_no=race_no,
            )

            def _get_pre(key: str, default: Any = "") -> Any:
                if isinstance(pre, dict):
                    return pre.get(key, default)
                return getattr(pre, key, default)

            weather = (_get_pre("weather", "") or "").strip()
            wind_dir = (_get_pre("wind_dir", "") or _get_pre("wind_direction", "") or "").strip()
            wind_speed_mps = _safe_float(
                _get_pre("wind_speed_mps", _get_pre("wind_speed", _get_pre("wind_mps", 0.0))),
                0.0,
            )
            wave_cm = _safe_float(_get_pre("wave_cm", _get_pre("wave", 0.0)), 0.0)

            if weather:
                pre_info["weather"] = weather
            if wind_dir:
                pre_info["wind_dir"] = wind_dir
                pre_info["wind_direction"] = wind_dir
            if wind_speed_mps > 0:
                pre_info["wind_speed"] = wind_speed_mps
                pre_info["wind_speed_mps"] = wind_speed_mps
            if wave_cm > 0:
                pre_info["wave_cm"] = wave_cm

            if lane_map:
                _inject_preinfo_lane_map(entries, lane_map)

        except Exception as e:
            if _is_debug_request():
                print("[PREINFO_FETCHER_ERROR]", e)

    try:
        if venue_name == "戸田":
            grouped_odds = controller.get_odds_only_toda(race_no=race_no, date=date)
        else:
            grouped_odds = controller.get_odds_only(race_no=race_no, date=date)
    except Exception as e:
        grouped_odds = None
        if _is_debug_request():
            print("[ODDS_ERROR]", e)

    ai = _calc_ai_outputs(
        venue_name=venue_name,
        date=date,
        race_no=race_no,
        entries=entries,
        grouped_odds=grouped_odds,
        pre_info=pre_info,
        beforeinfo=beforeinfo_for_builder,
    )
    probabilities = ai.get("probabilities") or {}
    ev_result = ai.get("ev_result") or {}
    ev_cutoff_90 = ai.get("ev_cutoff_90")
    ev_cutoff_95 = ai.get("ev_cutoff_95")
    pre_info = ai.get("pre_info") or pre_info

    races[race_no - 1]["entries"] = entries

    return render_template(
        "index.html",
        venue=venue_name,
        date=date,
        races=races,
        selected_race=selected_race,
        grouped_odds=grouped_odds,
        probabilities=probabilities,
        ev_result=ev_result,
        ev_cutoff_90=ev_cutoff_90,
        ev_cutoff_95=ev_cutoff_95,
        pre_info=pre_info,
        beforeinfo=beforeinfo_for_template,
    )


@app.route("/")
def home():
    date = request.args.get("date", "").strip() or _today_yyyymmdd_tokyo()
    return render_template("home.html", date=date)


@app.route("/marugame")
def marugame():
    return _render_venue_page("丸亀")


@app.route("/toda")
def toda():
    return _render_venue_page("戸田")


if __name__ == "__main__":
    app.run(debug=True)
	engine/model_bootstrap.py
from __future__ import annotations

import gzip
import os
import shutil
import urllib.request


MODEL_DIR = "data/models"
MODEL_PATH = os.path.join(MODEL_DIR, "trifecta120_model.joblib")
MODEL_GZ_PATH = os.path.join(MODEL_DIR, "trifecta120_model.joblib.gz")


def ensure_model_ready() -> None:
    """
    Render / 本番環境でモデルを自動配置する
    優先順位:
    1. すでに .joblib がある
    2. .gz があるなら解凍
    3. MODEL_URL から .gz をDLして解凍
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print(f"[MODEL] already exists: {MODEL_PATH}")
        return

    if os.path.exists(MODEL_GZ_PATH):
        print(f"[MODEL] extracting local gzip: {MODEL_GZ_PATH}")
        _extract_gzip(MODEL_GZ_PATH, MODEL_PATH)
        return

    model_url = os.getenv("MODEL_URL", "").strip()
    if not model_url:
        raise FileNotFoundError(
            "Model not found locally and MODEL_URL is empty. "
            "Set MODEL_URL in environment variables."
        )

    print(f"[MODEL] downloading: {model_url}")
    urllib.request.urlretrieve(model_url, MODEL_GZ_PATH)

    print(f"[MODEL] extracting downloaded gzip: {MODEL_GZ_PATH}")
    _extract_gzip(MODEL_GZ_PATH, MODEL_PATH)

    print(f"[MODEL] ready: {MODEL_PATH}")


def _extract_gzip(src_gz: str, dst_path: str) -> None:
    with gzip.open(src_gz, "rb") as f_in:
        with open(dst_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
お願いします。Render の環境変数も入れました。場所は今までのままです。please fix and send full code.
