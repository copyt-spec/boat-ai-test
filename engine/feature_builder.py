# engine/feature_builder.py

import re
from typing import Any, Dict, List, Optional


# ".04" も拾えるようにする
_NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")


def _to_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()
    m = _NUM_RE.search(s)
    if not m:
        return default

    try:
        return float(m.group(0))
    except ValueError:
        return default


def _st_to_float(st: Any, default: float = 0.0) -> float:
    """
    STを数値化
      "0.12" -> 0.12
      "F.04" -> -0.04（フライングはマイナス扱い）
      "F0.04" -> -0.04
      "L" / "---" 等 -> 0
    """
    if st is None:
        return default

    s = str(st).strip().upper()

    # フライング
    if s.startswith("F"):
        v = _to_float(s, default=default)
        return -abs(v)

    return _to_float(s, default=default)


def _safe_get(d: Dict[str, Any], *keys: str, default: Any = 0) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return default


def build_features(entries: List[Dict[str, Any]], before_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    entries: 出走表データ（6艇分） ※list[dict]
    before_info: 直前情報（None可） ※dict: {1:{...},2:{...},..., 'wind_speed':..., ...}

    return: list of dict (6艇分)
    """
    features: List[Dict[str, Any]] = []

    for boat in entries:
        lane = boat.get("lane") or boat.get("course") or boat.get("waku") or boat.get("frame")
        try:
            lane_i = int(lane)
        except Exception:
            lane_i = int(boat.get("lane", 0) or 0)

        # entries側キーは環境により違うので候補を広く取る（不足は0）
        grade_score = _to_float(_safe_get(boat, "grade_score", "gradeScore", "gradePoint", default=0))
        power_score = _to_float(_safe_get(boat, "power_score", "powerScore", "powerPoint", default=0))
        national_win = _to_float(_safe_get(boat, "national_win", "win_rate", "national_win_rate", default=0))
        local_win = _to_float(_safe_get(boat, "local_win", "place_rate", "local_win_rate", default=0))
        motor_2rate = _to_float(_safe_get(boat, "motor_2rate", "motor2rate", "motor_win_rate", default=0))

        feature: Dict[str, Any] = {
            "lane": lane_i,
            "grade_score": grade_score,
            "power_score": power_score,
            "national_win": national_win,
            "local_win": local_win,
            "motor_2rate": motor_2rate,
        }

        if isinstance(before_info, dict):
            info = before_info.get(lane_i, {}) if lane_i in before_info else {}
            if isinstance(info, dict):
                feature.update({
                    "exhibit_time": _to_float(info.get("exhibit_time"), 0.0),
                    "tilt": _to_float(info.get("tilt"), 0.0),
                    "st": _st_to_float(info.get("st"), 0.0),
                })

            feature.update({
                "wind_speed": _to_float(before_info.get("wind_speed"), 0.0),
                "temperature": _to_float(before_info.get("temperature"), 0.0),
            })

        features.append(feature)

    return features
