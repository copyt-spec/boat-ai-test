# scripts/train_trifecta_120cls.py
from __future__ import annotations

import json
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/datasets/trifecta_train_features.csv"
MODEL_PATH = "data/models/trifecta120_model.joblib"
META_PATH = "data/models/trifecta120_model_meta.json"

TARGET_COL = "y_combo"

# ※ この列は学習に使わない（特徴量から除外）
DROP_COLS = {
    TARGET_COL,
    "combo",          # 120行入力のcombo列（特徴ではない）
}

# 学習を安定させるためのハイパラ（まずは無難）
SGD_PARAMS = dict(
    loss="log_loss",          # ←確率を出せる
    penalty="l2",
    alpha=1e-5,
    learning_rate="optimal",
    fit_intercept=True,
    max_iter=1,               # partial_fit を回すので 1
    tol=None,
    random_state=42,
)

CHUNK_ROWS = 200_000          # 既にあなたが回してるサイズ感
EPOCHS = 1                    # まずは1周（必要なら2以上）


def _infer_feature_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in DROP_COLS]
    # 数値化できる列だけにする（念のため）
    keep = []
    for c in cols:
        if c in ("venue", "date", "race_no"):
            keep.append(c)
        else:
            # それ以外も基本は数値想定
            keep.append(c)
    return keep


def _read_head(path: str, n: int = 1000) -> pd.DataFrame:
    return pd.read_csv(path, nrows=n)


def _safe_numeric_frame(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()

    # 空文字→NaN→0、float化
    X = X.replace("", np.nan)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # float32 に落としてメモリ節約
    return X.astype(np.float32)


def _scan_classes(path: str) -> List[str]:
    # 120クラス固定ならここで拾って固定
    # まずは全体から unique を取る（csvが巨大なのでchunkで集める）
    s = set()
    for chunk in pd.read_csv(path, usecols=[TARGET_COL], chunksize=CHUNK_ROWS):
        vals = chunk[TARGET_COL].astype(str).values
        s.update(vals.tolist())
        # 120クラス想定なら、揃ったら打ち切りも可
        if len(s) >= 120:
            break
    classes = sorted(list(s))
    return classes


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    # feature_cols 決定
    head = _read_head(DATA_PATH, 2000)
    if TARGET_COL not in head.columns:
        raise ValueError(f"{TARGET_COL} not found in {DATA_PATH}")

    feature_cols = _infer_feature_cols(head)

    # classes 決定（partial_fit に必要）
    classes = _scan_classes(DATA_PATH)
    if len(classes) != 120:
        print(f"[WARN] classes found={len(classes)} (expected 120). Continue anyway.")
    classes_np = np.array(classes, dtype=object)

    # Pipeline（Scaler + SGD）
    # StandardScaler: with_mean=False は疎行列向けだが、今回はdenseなので with_mean=True のまま
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", SGDClassifier(**SGD_PARAMS)),
        ]
    )

    total_seen = 0
    for epoch in range(EPOCHS):
        print(f"==== EPOCH {epoch + 1}/{EPOCHS} ====")

        for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_ROWS):
            # y
            y = chunk[TARGET_COL].astype(str).values

            # X
            X = _safe_numeric_frame(chunk, feature_cols).values

            if total_seen == 0 and epoch == 0:
                # 初回 partial_fit は classes 指定必須
                model.named_steps["clf"].partial_fit(
                    model.named_steps["scaler"].fit_transform(X),
                    y,
                    classes=classes_np,
                )
            else:
                Xs = model.named_steps["scaler"].transform(X)
                model.named_steps["clf"].partial_fit(Xs, y)

            total_seen += len(chunk)
            if total_seen % 200_000 == 0:
                print(f"trained_rows={total_seen:,}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "feature_names": feature_cols,
        "classes": classes,
        "model_type": "SGDClassifier_log_loss",
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("saved_model:", MODEL_PATH)
    print("saved_meta :", META_PATH)
    print("feature_cols:", len(feature_cols))
    print("classes     :", len(classes))


if __name__ == "__main__":
    main()
