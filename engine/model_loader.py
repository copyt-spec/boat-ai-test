# engine/model_loader.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

VENUE_CODE_MAP = {"丸亀": 15, "戸田": 2}

DEFAULT_MODEL_PATH = "data/models/trifecta120_model.joblib"
DEFAULT_META_PATH = "data/models/trifecta120_model_meta.json"


def _finite_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("", np.nan)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _entropy(p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / float(np.sum(p))
    return float(-np.sum(p * np.log(p)))


def _ent_ratio(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    k = int(p.size)
    if k <= 1:
        return 0.0
    ent = _entropy(p)
    ent_max = float(np.log(k))
    return float(ent / ent_max) if ent_max > 0 else 0.0


def _apply_temperature_on_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    確率に温度をかける安全な方法：
    p' ∝ exp(log(p)/T)
    T>1 でフラット, T<1 でシャープ
    """
    P = np.asarray(probs, dtype=float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 1e-15, 1.0)
    T = float(temperature)
    if not np.isfinite(T) or T <= 0:
        return P / np.sum(P, axis=1, keepdims=True)

    L = np.log(P) / T
    # 数値安定
    L = L - np.max(L, axis=1, keepdims=True)
    E = np.exp(L)
    S = np.sum(E, axis=1, keepdims=True)
    S = np.where(S <= 0, 1.0, S)
    return E / S


class BoatRaceModel:
    """
    120行（comboごとの特徴量） -> 120クラス確率を作る
    方針：
      - 推論は predict_proba を最優先（ここが “正しい修正”）
      - 集約は幾何平均（安定）
      - 尖り救済は uniform置換ではなく “少量ブレンド”
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        meta_path: str = DEFAULT_META_PATH,
        temperature: float = 1.0,      # >1でフラット
        output_tau: float = 1.10,      # >1でフラット（確率空間）
        out_min_prob: float = 1e-6,
        use_geometric_mean: bool = True,
        rescue_max: float = 0.85,      # 尖り救済の発動閾値（maxがこれ以上）
        rescue_mix_cap: float = 0.20,  # uniformブレンド上限（大きくしない）
        debug: bool = False,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta not found: {meta_path}")

        self.model = joblib.load(model_path)

        self.temperature = float(temperature) if np.isfinite(float(temperature)) else 1.0
        self.output_tau = float(output_tau) if np.isfinite(float(output_tau)) and float(output_tau) > 0 else 1.0
        self.out_min_prob = float(out_min_prob) if np.isfinite(float(out_min_prob)) and float(out_min_prob) > 0 else 1e-6
        self.use_geometric_mean = bool(use_geometric_mean)

        self.rescue_max = float(rescue_max) if np.isfinite(float(rescue_max)) else 0.85
        self.rescue_mix_cap = float(rescue_mix_cap) if np.isfinite(float(rescue_mix_cap)) else 0.20
        self.debug = bool(debug)

        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        self.feature_names: List[str] = meta.get("feature_names") or []
        self.classes_: List[str] = [str(c) for c in (meta.get("classes") or [])]

        if not self.feature_names:
            raise ValueError("meta.feature_names is missing")
        if not self.classes_:
            raise ValueError("meta.classes is missing")

        self.class_index: Dict[str, int] = {str(c): i for i, c in enumerate(self.classes_)}

        # モデル内部 classes_ との順序合わせ（predict_probaの列順が違う可能性）
        self._reorder_idx = self._build_reorder_index(self._get_model_classes(), self.classes_)

    # -------------------------
    # classes alignment
    # -------------------------
    def _get_model_classes(self) -> List[str]:
        est = self.model
        if hasattr(est, "steps") and getattr(est, "steps", None):
            try:
                est = est.steps[-1][1]
            except Exception:
                est = self.model

        if hasattr(est, "classes_"):
            try:
                return [str(c) for c in list(est.classes_)]
            except Exception:
                pass

        if hasattr(self.model, "classes_"):
            try:
                return [str(c) for c in list(self.model.classes_)]
            except Exception:
                pass

        return list(self.classes_)

    def _build_reorder_index(self, model_classes: List[str], meta_classes: List[str]) -> Optional[np.ndarray]:
        if not model_classes or not meta_classes:
            return None
        if len(model_classes) != len(meta_classes):
            return None
        pos = {c: i for i, c in enumerate(model_classes)}
        idx = []
        for c in meta_classes:
            if c not in pos:
                return None
            idx.append(pos[c])
        return np.asarray(idx, dtype=int)

    # -------------------------
    # feature handling
    # -------------------------
    def _coerce_types(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.replace("", np.nan)

        if "combo" in X.columns:
            X = X.drop(columns=["combo"])

        if "date" in X.columns:
            X["date"] = pd.to_numeric(X["date"], errors="coerce")

        if "venue" in X.columns:
            def map_venue(v: Any) -> float:
                s = str(v).strip()
                if s.isdigit():
                    return float(int(s))
                return float(VENUE_CODE_MAP.get(s, 0))
            X["venue"] = X["venue"].map(map_venue)

        if "race_no" in X.columns:
            X["race_no"] = pd.to_numeric(X["race_no"], errors="coerce")

        for c in list(X.columns):
            if not is_numeric_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce")

        X = _finite_df(X)

        # float化（''混入事故を根絶）
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
        return X.astype(float)

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.feature_names:
            if c not in X.columns:
                X[c] = 0.0
        X = X[self.feature_names]
        return X

    # -------------------------
    # proba core（predict_proba優先）
    # -------------------------
    def _predict_proba_matrix(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            P = np.asarray(self.model.predict_proba(X), dtype=float)
        else:
            raise AttributeError("Model has no predict_proba (retrain with prob model).")

        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        # classes順序合わせ
        if self._reorder_idx is not None and P.ndim == 2 and P.shape[1] == len(self.classes_):
            try:
                P = P[:, self._reorder_idx]
            except Exception:
                pass

        # 行正規化
        row_sum = P.sum(axis=1, keepdims=True)
        bad = (row_sum <= 0) | (~np.isfinite(row_sum))
        if np.any(bad):
            P[bad[:, 0], :] = 1.0 / float(P.shape[1])
            row_sum = P.sum(axis=1, keepdims=True)
        P = P / np.where(row_sum <= 0, 1.0, row_sum)

        # 温度（確率空間）
        if np.isfinite(self.temperature) and self.temperature > 0 and abs(self.temperature - 1.0) > 1e-12:
            P = _apply_temperature_on_probs(P, self.temperature)

        return P

    # -------------------------
    # aggregation（幾何平均 + 軽い救済）
    # -------------------------
    def _aggregate_class_distribution(self, P: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        P = np.asarray(P, dtype=float)
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        P = np.clip(P, 1e-15, 1.0)

        if self.use_geometric_mean:
            logm = np.mean(np.log(P), axis=0)
            class_p = np.exp(logm)
        else:
            class_p = np.mean(P, axis=0)

        class_p = np.nan_to_num(class_p, nan=0.0, posinf=0.0, neginf=0.0)
        class_p = np.maximum(class_p, self.out_min_prob)

        s = float(class_p.sum())
        class_p = class_p / (s if s > 0 else 1.0)

        # 出力フラット化（tau > 1）
        tau = self.output_tau
        if np.isfinite(tau) and tau > 0 and abs(tau - 1.0) > 1e-12:
            class_p = np.power(np.clip(class_p, 1e-15, 1.0), 1.0 / tau)
            s2 = float(class_p.sum())
            class_p = class_p / (s2 if s2 > 0 else 1.0)

        mx = float(np.max(class_p))
        er = _ent_ratio(class_p)

        info = {"mx": mx, "ent_ratio": er, "mix": 0.0, "after_mx": mx}

        # 尖り救済：uniform置換は禁止。少量ブレンドのみ。
        if mx >= self.rescue_max:
            k = int(class_p.size)
            u = 1.0 / float(k)

            # mx が高いほど少し混ぜる。上限は rescue_mix_cap。
            # 例: rescue_max=0.85, mx=0.99 -> mix~0.20 くらい
            mix = (mx - self.rescue_max) / (1.0 - self.rescue_max)
            mix = float(np.clip(mix, 0.05, self.rescue_mix_cap))

            uniform = np.full(k, u, dtype=float)
            class_p = (1.0 - mix) * class_p + mix * uniform
            class_p = class_p / float(class_p.sum())

            info["mix"] = mix
            info["after_mx"] = float(np.max(class_p))

        return class_p, info

    # -------------------------
    # public API
    # -------------------------
    def predict_proba(self, features_120: pd.DataFrame) -> Dict[str, float]:
        if not isinstance(features_120, pd.DataFrame):
            raise TypeError("features_120 must be DataFrame")
        if "combo" not in features_120.columns:
            raise ValueError("features_120 must contain 'combo' column")

        combos: List[str] = [str(x) for x in features_120["combo"].values]

        X = self._coerce_types(features_120)
        X = self._align_columns(X)

        P = self._predict_proba_matrix(X)  # (120,120)
        class_p, info = self._aggregate_class_distribution(P)

        out_vec = np.zeros(len(combos), dtype=float)
        for i, combo in enumerate(combos):
            j = self.class_index.get(combo, -1)
            out_vec[i] = float(class_p[j]) if 0 <= j < len(class_p) else 0.0

        out_vec = np.nan_to_num(out_vec, nan=0.0, posinf=0.0, neginf=0.0)
        out_vec = np.maximum(out_vec, self.out_min_prob)
        s = float(out_vec.sum())
        out_vec = out_vec / (s if s > 0 else 1.0)

        if self.debug:
            print(
                f"[DEBUG] agg: mx={info['mx']:.6f} ent_ratio={info['ent_ratio']:.3f} "
                f"mix={info['mix']:.3f} after_mx={info['after_mx']:.6f}"
            )
            print(
                f"[DEBUG] out_vec stats: min={float(np.min(out_vec)):.10f} "
                f"max={float(np.max(out_vec)):.10f} ent={_entropy(out_vec):.6f}"
            )

        return {combos[i]: float(out_vec[i]) for i in range(len(combos))}
