# scripts/train_trifecta_120cls.py
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

# ===== import path 対策 =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier


INPUT_PATH = "data/datasets/trifecta_train_features.csv"
MODEL_PATH = "data/models/trifecta120_model.joblib"
META_PATH = "data/models/trifecta120_model_meta.json"

# ===== 学習設定 =====
MAX_RACES = 50000
VALID_RATIO = 0.15
RANDOM_STATE = 42

# ===== LightGBM設定 =====
N_ESTIMATORS = 500
LEARNING_RATE = 0.05
NUM_LEAVES = 63
MAX_DEPTH = 8
MIN_CHILD_SAMPLES = 60
SUBSAMPLE = 0.85
COLSAMPLE_BYTREE = 0.85
REG_ALPHA = 0.3
REG_LAMBDA = 1.0

# ===== 進捗表示設定 =====
PROGRESS_EVERY = 10


def _safe_float_series(s: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(s, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def _build_sample_weight(y_str: pd.Series) -> np.ndarray:
    vc = y_str.value_counts(dropna=False)
    freq_map = vc.to_dict()

    raw = y_str.map(lambda x: 1.0 / float(freq_map.get(x, 1))).astype(float)

    mean_w = float(raw.mean()) if len(raw) > 0 else 1.0
    if mean_w <= 0:
        mean_w = 1.0
    raw = raw / mean_w

    raw = raw.clip(lower=0.25, upper=6.0)
    return raw.to_numpy(dtype=float)


def _sample_races(df: pd.DataFrame, max_races: int) -> pd.DataFrame:
    if len(df) % 120 != 0:
        print(f"[WARN] row count is not multiple of 120: {len(df)}")

    race_count = len(df) // 120
    if race_count <= max_races:
        print(f"[INFO] use all races: {race_count}")
        return df.reset_index(drop=True)

    rng = np.random.default_rng(RANDOM_STATE)
    race_indices = np.arange(race_count)
    picked = np.sort(rng.choice(race_indices, size=max_races, replace=False))

    parts = []
    total = len(picked)
    for i, ridx in enumerate(picked, start=1):
        s = ridx * 120
        e = s + 120
        parts.append(df.iloc[s:e])

        if i == 1 or i % 2000 == 0 or i == total:
            print(f"[sampling] {i}/{total} races selected")

    out = pd.concat(parts, axis=0).reset_index(drop=True)
    print(f"[INFO] sampled races: {max_races} / {race_count}")
    print(f"[INFO] sampled rows : {len(out)}")
    return out


def _split_train_valid_by_race(
    df: pd.DataFrame,
    valid_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) % 120 != 0:
        raise ValueError(f"row count must be multiple of 120, got {len(df)}")

    race_count = len(df) // 120
    race_indices = np.arange(race_count)

    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(race_indices)

    valid_races = max(1, int(race_count * valid_ratio))
    valid_set = set(race_indices[:valid_races])

    train_parts = []
    valid_parts = []

    for ridx in range(race_count):
        s = ridx * 120
        e = s + 120
        if ridx in valid_set:
            valid_parts.append(df.iloc[s:e])
        else:
            train_parts.append(df.iloc[s:e])

        if (ridx + 1) == 1 or (ridx + 1) % 2000 == 0 or (ridx + 1) == race_count:
            print(f"[split] {ridx + 1}/{race_count} races processed")

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    valid_df = pd.concat(valid_parts, axis=0).reset_index(drop=True)

    print(f"[INFO] train races: {len(train_df) // 120}")
    print(f"[INFO] valid races: {len(valid_df) // 120}")
    print(f"[INFO] train rows : {len(train_df)}")
    print(f"[INFO] valid rows : {len(valid_df)}")

    return train_df, valid_df


def _prepare_xy(df: pd.DataFrame, label_encoder: LabelEncoder | None = None):
    if "y_combo" not in df.columns:
        raise ValueError("y_combo column not found")
    if "combo" not in df.columns:
        raise ValueError("combo column not found")

    X = df.drop(columns=["y_combo"]).copy()
    if "combo" in X.columns:
        X = X.drop(columns=["combo"])

    total_cols = len(X.columns)
    for i, c in enumerate(X.columns, start=1):
        X[c] = _safe_float_series(X[c])
        if i == 1 or i % 30 == 0 or i == total_cols:
            print(f"[numeric] {i}/{total_cols} columns converted")

    y_str = df["y_combo"].astype(str).fillna("").str.strip()
    if (y_str == "").any():
        raise ValueError("y_combo contains empty values")

    if label_encoder is None:
        le = LabelEncoder()
        y = le.fit_transform(y_str)
    else:
        le = label_encoder
        y = le.transform(y_str)

    sample_weight = _build_sample_weight(y_str)

    return X, y, y_str, sample_weight, le


def _make_progress_callback(total_rounds: int, every: int = 10):
    start_time = time.time()

    def _callback(env):
        # env.iteration は 0始まり
        current = env.iteration + 1

        should_print = (
            current == 1
            or current % every == 0
            or current == total_rounds
        )
        if not should_print:
            return

        elapsed = time.time() - start_time
        msg = f"[train] round {current}/{total_rounds}  elapsed={elapsed:.1f}s"

        if getattr(env, "evaluation_result_list", None):
            metrics = []
            for item in env.evaluation_result_list:
                # 形式: (data_name, eval_name, result, is_higher_better) など
                if len(item) >= 3:
                    data_name = item[0]
                    eval_name = item[1]
                    score = item[2]
                    metrics.append(f"{data_name}-{eval_name}={score:.6f}")
            if metrics:
                msg += "  " + "  ".join(metrics)

        print(msg)

    _callback.order = 10
    _callback.before_iteration = False
    return _callback


def main() -> None:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing: {INPUT_PATH}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print("===== STEP 1: load csv =====")
    print("loading:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)
    print(f"[loaded] rows={len(df):,} cols={len(df.columns)}")

    print("\n===== STEP 2: sample races =====")
    df = _sample_races(df, MAX_RACES)

    print("\n===== STEP 3: split train / valid by race =====")
    train_df, valid_df = _split_train_valid_by_race(df, VALID_RATIO)

    print("\n===== STEP 4: prepare train features =====")
    X_train, y_train, y_train_str, w_train, le = _prepare_xy(train_df, label_encoder=None)

    print("\n===== STEP 5: prepare valid features =====")
    X_valid, y_valid, y_valid_str, w_valid, _ = _prepare_xy(valid_df, label_encoder=le)

    feature_names: List[str] = list(X_train.columns)

    print("\n===== STEP 6: summary =====")
    print("train rows   :", len(X_train))
    print("valid rows   :", len(X_valid))
    print("feature_cols :", len(feature_names))
    print("classes      :", len(le.classes_))
    print("train w min  :", float(w_train.min()))
    print("train w mean :", float(w_train.mean()))
    print("train w max  :", float(w_train.max()))

    print("\n===== STEP 7: build model =====")
    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(le.classes_),
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        max_depth=MAX_DEPTH,
        min_child_samples=MIN_CHILD_SAMPLES,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    print("\n===== STEP 8: training start =====")
    fit_start = time.time()
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_valid, y_valid)],
        eval_sample_weight=[w_valid],
        eval_metric="multi_logloss",
        callbacks=[
            _make_progress_callback(total_rounds=N_ESTIMATORS, every=PROGRESS_EVERY),
        ],
    )
    fit_elapsed = time.time() - fit_start
    print(f"[train] finished in {fit_elapsed:.1f}s")

    print("\n===== STEP 9: validation quick metric =====")
    valid_proba = model.predict_proba(X_valid)
    valid_pred = np.argmax(valid_proba, axis=1)
    valid_acc = float((valid_pred == y_valid).mean())
    print("valid_top1_row_acc:", f"{valid_acc:.6f}")

    print("\n===== STEP 10: save =====")
    joblib.dump(model, MODEL_PATH)

    meta = {
        "feature_names": feature_names,
        "classes": [str(x) for x in le.classes_.tolist()],
        "n_features": len(feature_names),
        "n_classes": int(len(le.classes_)),
        "model_type": "LGBMClassifier(multiclass)",
        "sample_weight": {
            "enabled": True,
            "method": "inverse_class_frequency_clipped",
            "clip_lower": 0.25,
            "clip_upper": 6.0,
        },
        "train_sampling": {
            "max_races": MAX_RACES,
            "valid_ratio": VALID_RATIO,
            "random_state": RANDOM_STATE,
        },
        "params": {
            "objective": "multiclass",
            "num_class": int(len(le.classes_)),
            "n_estimators": N_ESTIMATORS,
            "learning_rate": LEARNING_RATE,
            "num_leaves": NUM_LEAVES,
            "max_depth": MAX_DEPTH,
            "min_child_samples": MIN_CHILD_SAMPLES,
            "subsample": SUBSAMPLE,
            "colsample_bytree": COLSAMPLE_BYTREE,
            "reg_alpha": REG_ALPHA,
            "reg_lambda": REG_LAMBDA,
            "random_state": RANDOM_STATE,
        },
        "quick_metrics": {
            "valid_top1_row_acc": valid_acc,
        },
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("saved_model:", MODEL_PATH)
    print("saved_meta :", META_PATH)
    print("feature_cols:", len(feature_names))
    print("classes     :", len(le.classes_))


if __name__ == "__main__":
    main()
