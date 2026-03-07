# engine/ev_calculator.py
from __future__ import annotations

from typing import Any, Dict, Optional
import math


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            v = float(x)
        else:
            s = str(x).strip()
            if s == "" or s.lower() in ("nan", "none", "null", "-"):
                return None
            v = float(s)
        if math.isnan(v) or math.isinf(v) or v <= 0:
            return None
        return v
    except Exception:
        return None


def calculate_ev(probabilities: Dict[str, float], grouped_odds: Dict[str, Any]) -> Dict[str, float]:
    """
    probabilities: {"1-2-3": 0.01, ...}  合計1想定（多少ズレてもOK）
    grouped_odds:
      {
        "data": {
          1: {(2,3): "8.2", (2,4): "20.2", ...},
          2: {...},
          ...
        },
        "min": ...,
        "max": ...
      }

    return:
      ev: {"1-2-3": prob * odds, ...}  （倍率ベース / テンプレ側では *100 して EV% 表示してOK）
    """
    out: Dict[str, float] = {}

    if not isinstance(probabilities, dict):
        return out

    data = (grouped_odds or {}).get("data")
    if not isinstance(data, dict):
        return out

    for combo, p in probabilities.items():
        try:
            a_s, b_s, c_s = str(combo).split("-")
            a, b, c = int(a_s), int(b_s), int(c_s)
        except Exception:
            continue

        odds_raw = None
        try:
            odds_raw = data.get(a, {}).get((b, c))
        except Exception:
            odds_raw = None

        odds = _to_float(odds_raw)
        if odds is None:
            # オッズ結合できないものは0扱い
            out[str(combo)] = 0.0
            continue

        try:
            pp = float(p)
            if math.isnan(pp) or math.isinf(pp) or pp < 0:
                pp = 0.0
        except Exception:
            pp = 0.0

        out[str(combo)] = pp * odds

    return out
