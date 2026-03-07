# engine/model_trainer.py
# -*- coding: utf-8 -*-
"""
ステップB：軽量モデル学習（確率校正込み） NaN完全排除 + StandardScaler + logloss nan対策

- 入力: data/datasets/trifecta_train_features.csv
- 出力: data/models/trifecta_model_calibrated.joblib
        data/models/trifecta_model_calibrated_meta.json

ポイント:
- 未来リーク列は除外（finish系など）
- 全体でNaNを完全排除（Calibrated内部CV対策）
- StandardScaler(with_mean=False) を入れて SGD(log_loss) を安定させる
- logloss は確率を clip して nan を防ぐ

実行例:
  python -m engine.model_trainer --max-rows 200000 --verbose
事前モデル寄せ:
  python -m engine.model_trainer --max-rows 200000 --drop-optional course,exhibit,st,tilt,parts,weather,wind --verbose
"""

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_DROP_COLS_EXACT = {
    "race_id",
    "id",
    "y",
    "label",
    "target",
}

DEFAULT_DROP_REGEX = [
    r"(^|_)finish($|_)",
    r"(^|_)rank($|_)",
    r"(^|_)result($|_)",
    r"(^|_)time_rank($|_)",
]

DEFAULT_OPTIONAL_GROUPS = {
    "course": [r"(^|_)course($|_)"],
    "exhibit": [r"(^|_)exhibit($|_)"],
    "st": [r"(^|_)st($|_)"],
    "tilt": [r"(^|_)tilt($|_)"],
    "parts": [r"(^|_)parts($|_)"],
    "weather": [r"(^|_)weather($|_)"],
    "wind": [r"(^|_)wind($|_)"],
}


@dataclass
class TrainMeta:
    model_type: str
    calibrated: bool
    calibrate_method: str
    test_size: float
    random_state: int
    max_rows_used: int
    n_rows_total_seen: int
    n_features: int
    positive_rate_train: float
    positive_rate_valid: float
    metrics_valid: Dict[str, float]
    dropped_columns_exact: List[str]
    dropped_columns_regex: List[str]
    dropped_optional_groups: List[str]
    dropped_all_nan_columns: List[str]
    feature_columns: List[str]


def _compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p) for p in patterns]


def _match_any(col: str, patterns: List[re.Pattern]) -> bool:
    return any(p.search(col) for p in patterns)


def select_feature_columns(
    columns: List[str],
    label_col: str,
    drop_cols_exact: Optional[set] = None,
    drop_regex: Optional[List[str]] = None,
    drop_optional_groups: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    drop_cols_exact = drop_cols_exact or set(DEFAULT_DROP_COLS_EXACT)
    drop_cols_exact = set(drop_cols_exact) | {label_col}

    drop_regex = drop_regex or DEFAULT_DROP_REGEX
    patterns = _compile_patterns(drop_regex)

    optional_groups = drop_optional_groups or []
    optional_patterns: List[re.Pattern] = []
    for g in optional_groups:
        for p in DEFAULT_OPTIONAL_GROUPS.get(g, []):
            optional_patterns.append(re.compile(p))

    feature_cols: List[str] = []
    dropped_exact: List[str] = []
    dropped_by_regex: List[str] = []

    for c in columns:
        if c in drop_cols_exact:
            dropped_exact.append(c)
            continue
        if optional_patterns and _match_any(c, optional_patterns):
            dropped_by_regex.append(c)
            continue
        if _match_any(c, patterns):
            dropped_by_regex.append(c)
            continue
        feature_cols.append(c)

    return feature_cols, sorted(set(dropped_exact)), sorted(set(dropped_by_regex))


def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            out[c] = df[c]
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def _drop_all_nan_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols)
    return df, all_nan_cols


def _final_impute_no_nan(df: pd.DataFrame, strategy: str = "median") -> Tuple[pd.DataFrame, Dict[str, float]]:
    if strategy == "zero":
        df2 = df.fillna(0.0)
        impute_values = {c: 0.0 for c in df.columns}
        df2 = df2.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df2, impute_values

    if strategy == "median":
        stats = df.median(numeric_only=True)
    elif strategy == "mean":
        stats = df.mean(numeric_only=True)
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    fill_map: Dict[str, float] = {}
    for c in df.columns:
        v = stats.get(c, np.nan)
        if pd.isna(v):
            v = 0.0
        fill_map[c] = float(v)

    df2 = df.fillna(fill_map)
    df2 = df2.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df2, fill_map


def _evaluate_prob(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    m: Dict[str, float] = {}
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)

    try:
        m["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        m["auc"] = float("nan")

    try:
        m["logloss"] = float(log_loss(y_true, y_prob, eps=eps))
    except Exception:
        m["logloss"] = float("nan")

    try:
        m["brier"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        m["brier"] = float("nan")

    m["mean_p"] = float(np.mean(y_prob))
    m["p_extreme_rate"] = float(np.mean((y_prob <= eps) | (y_prob >= 1 - eps)))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/datasets/trifecta_train_features.csv")
    ap.add_argument("--output", default="data/models/trifecta_model_calibrated.joblib")
    ap.add_argument("--label-col", default="y")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--chunksize", type=int, default=250000)
    ap.add_argument("--impute", default="median", choices=["median", "mean", "zero"])
    ap.add_argument("--drop-optional", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output
    label_col = args.label_col

    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    drop_optional_groups = [s.strip() for s in args.drop_optional.split(",") if s.strip()]

    head = pd.read_csv(in_path, nrows=5, low_memory=False)
    cols = list(head.columns)
    if label_col not in cols:
        raise ValueError(f"label column '{label_col}' not found")

    feature_cols, dropped_exact, dropped_regex = select_feature_columns(
        columns=cols,
        label_col=label_col,
        drop_cols_exact=set(DEFAULT_DROP_COLS_EXACT),
        drop_regex=DEFAULT_DROP_REGEX,
        drop_optional_groups=drop_optional_groups,
    )

    if args.verbose:
        print(f"[INFO] features selected: {len(feature_cols)}")
        print(f"[INFO] dropped exact: {dropped_exact}")
        print(f"[INFO] dropped regex count: {len(dropped_regex)}")
        if drop_optional_groups:
            print(f"[INFO] dropped optional groups: {drop_optional_groups}")

    usecols = feature_cols + [label_col]
    total_seen = 0
    parts_X: List[pd.DataFrame] = []
    parts_y: List[np.ndarray] = []

    reader = pd.read_csv(in_path, usecols=usecols, chunksize=args.chunksize, low_memory=False)

    for chunk in reader:
        if args.max_rows and total_seen >= args.max_rows:
            break

        if args.max_rows:
            remain = args.max_rows - total_seen
            if remain <= 0:
                break
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain].copy()

        y = pd.to_numeric(chunk[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
        y = (y > 0).astype(int)

        X = chunk.drop(columns=[label_col])
        Xn = _coerce_numeric_df(X).replace([np.inf, -np.inf], np.nan)

        parts_X.append(Xn)
        parts_y.append(y)

        total_seen += len(chunk)
        if args.verbose:
            print(f"[INFO] read chunk rows={len(chunk):,} total_seen={total_seen:,}")

    if not parts_X:
        raise RuntimeError("No data loaded.")

    X_all = pd.concat(parts_X, axis=0, ignore_index=True)
    y_all = np.concatenate(parts_y, axis=0)

    X_all, dropped_all_nan = _drop_all_nan_columns(X_all)
    if args.verbose and dropped_all_nan:
        print(f"[INFO] dropped all-NaN columns: {len(dropped_all_nan)}")

    X_all, impute_values = _final_impute_no_nan(X_all, strategy=args.impute)

    # 最終チェック
    if np.isnan(X_all.to_numpy()).any():
        raise ValueError("NaN still exists after final imputation (should not happen).")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_all,
        y_all,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_all if len(np.unique(y_all)) == 2 else None,
    )

    if args.verbose:
        print(f"[INFO] train rows={len(X_train):,} valid rows={len(X_valid):,}")
        print(f"[INFO] positive rate train={float(y_train.mean()):.6f} valid={float(y_valid.mean()):.6f}")

    # StandardScalerでスケールを整える（SGDの安定化）
    # with_mean=False は疎行列向けだが、denseでも安全
    base_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-5,
                max_iter=2000,
                tol=1e-3,
                random_state=args.random_state,
                class_weight="balanced",
            )),
        ]
    )

    clf = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=3,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    p_valid = clf.predict_proba(X_valid)[:, 1]
    metrics = _evaluate_prob(y_valid, p_valid)

    if args.verbose:
        print(f"[VALID] AUC={metrics['auc']:.6f} LOGLOSS={metrics['logloss']:.6f} BRIER={metrics['brier']:.6f}")

    payload = {
        "model": clf,
        "feature_columns": list(X_all.columns),
        "label_col": label_col,
        "impute": args.impute,
        "impute_values": impute_values,
        "dropped_exact": dropped_exact,
        "dropped_regex": dropped_regex,
        "dropped_optional_groups": drop_optional_groups,
        "dropped_all_nan_columns": dropped_all_nan,
    }
    joblib.dump(payload, out_path)

    meta = TrainMeta(
        model_type="StandardScaler + SGDClassifier(log_loss) + CalibratedClassifierCV(sigmoid)",
        calibrated=True,
        calibrate_method="sigmoid",
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        max_rows_used=int(args.max_rows or 0),
        n_rows_total_seen=int(total_seen),
        n_features=int(X_all.shape[1]),
        positive_rate_train=float(y_train.mean()) if len(y_train) else float("nan"),
        positive_rate_valid=float(y_valid.mean()) if len(y_valid) else float("nan"),
        metrics_valid=metrics,
        dropped_columns_exact=dropped_exact,
        dropped_columns_regex=dropped_regex,
        dropped_optional_groups=drop_optional_groups,
        dropped_all_nan_columns=dropped_all_nan,
        feature_columns=list(X_all.columns),
    )

    meta_path = os.path.splitext(out_path)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    print(f"[DONE] saved model: {out_path}")
    print(f"[DONE] saved meta : {meta_path}")


if __name__ == "__main__":
    main()
