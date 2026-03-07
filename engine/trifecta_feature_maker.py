# engine/trifecta_feature_maker.py
# -*- coding: utf-8 -*-
"""
trifecta_train.csv（120通り展開済み）から、combo依存の特徴量を生成して
trifecta_train_features.csv を作る。

- feature_builder.py とは別物（既存用途を壊さないため）
- 大容量CSV対応：pandas の chunk 読み込み
- まずは --max-rows 200000 で動作確認推奨

想定：
- 入力CSVに combo 列があり、"1-2-3" 形式（または "1_2_3", "1 2 3" など）で艇番/枠番が入る
- 入力CSVに「各艇(1..6)の元特徴量」が何らかの命名規則で入っている（例：lane1_win_rate / p1_win_rate / boat1_win_rate など）
  -> 本スクリプトは複数パターンを自動検出して拾う（検出できない場合はエラーにする）

出力：
- id系（存在するもの） + combo + label（存在するもの） + 生成特徴量
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


LANES = [1, 2, 3, 4, 5, 6]


def _safe_float_series(s: pd.Series) -> pd.Series:
    # 文字列混在を想定して数値化（変換できないものは NaN）
    return pd.to_numeric(s, errors="coerce")


def parse_combo_to_abc(combo_s: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    combo列から a,b,c（1着/2着/3着の艇番）を取り出す。
    許容例: "1-2-3", "1_2_3", "1 2 3", "1,2,3"
    """
    # 非数字を区切りにして3つの数字を拾う
    extracted = combo_s.astype(str).str.extract(r"^\s*(\d)\D+(\d)\D+(\d)\s*$")
    if extracted.isna().any(axis=1).mean() > 0.05:
        # 5%以上パース失敗なら、より緩い抽出に再挑戦（末尾余計な文字など）
        extracted = combo_s.astype(str).str.extract(r"(\d)\D+(\d)\D+(\d)")

    a = pd.to_numeric(extracted[0], errors="coerce").astype("Int64")
    b = pd.to_numeric(extracted[1], errors="coerce").astype("Int64")
    c = pd.to_numeric(extracted[2], errors="coerce").astype("Int64")

    if a.isna().any() or b.isna().any() or c.isna().any():
        bad = int((a.isna() | b.isna() | c.isna()).sum())
        raise ValueError(f"combo parse failed on {bad} rows. combo format must contain 3 lane digits like 1-2-3.")

    return a, b, c


def detect_lane_feature_columns(columns: List[str]) -> Dict[int, Dict[str, str]]:
    """
    入力CSVの列名から「艇(1..6)ごとの特徴量」を自動検出する。

    戻り値:
        lane_cols[lane][basename] = actual_column_name
        例: lane_cols[1]["win_rate"] = "lane1_win_rate"
    検出対象パターン（どれかに一致すればOK）:
        lane{n}_{feat}
        l{n}_{feat}
        p{n}_{feat}
        boat{n}_{feat}
        racer{n}_{feat}
        entry{n}_{feat}
    """
    # よくあるprefix候補
    prefixes = ["lane", "l", "p", "boat", "racer", "entry"]
    # laneごとの辞書
    lane_cols: Dict[int, Dict[str, str]] = {n: {} for n in LANES}

    # 1) prefix + lane + "_" + feat
    #    例: lane1_win_rate / p2_st / boat6_motor2rate
    pat = re.compile(rf"^({'|'.join(prefixes)})([1-6])_(.+)$")

    for col in columns:
        m = pat.match(col)
        if not m:
            continue
        lane = int(m.group(2))
        basename = m.group(3)
        # 同じbasenameに複数候補がある場合、先に見つかったものを優先
        lane_cols[lane].setdefault(basename, col)

    # 最低限：全laneで共通して揃っているbasenameだけを採用
    # （一部laneだけ存在する特徴を混ぜると欠損が増えすぎるため）
    common_basenames = None
    for lane in LANES:
        bset = set(lane_cols[lane].keys())
        common_basenames = bset if common_basenames is None else (common_basenames & bset)

    if not common_basenames:
        raise ValueError(
            "Could not detect per-lane feature columns.\n"
            "Expected columns like lane1_xxx, lane2_xxx ... lane6_xxx (or p1_xxx etc)."
        )

    # commonだけ残す
    common = sorted(common_basenames)
    filtered: Dict[int, Dict[str, str]] = {n: {} for n in LANES}
    for lane in LANES:
        for base in common:
            filtered[lane][base] = lane_cols[lane][base]

    return filtered


def build_features_for_chunk(
    df: pd.DataFrame,
    lane_cols: Dict[int, Dict[str, str]],
    keep_cols: List[str],
    label_col: Optional[str],
) -> pd.DataFrame:
    """
    combo(a,b,c)に応じて、艇別特徴を a_, b_, c_ に展開し、
    さらに差分/和など軽量な派生特徴を作る。
    """
    if "combo" not in df.columns:
        raise ValueError("Input CSV must contain 'combo' column.")

    a, b, c = parse_combo_to_abc(df["combo"])
    df_local = df.copy()

    # 生成特徴量格納
    out = pd.DataFrame(index=df_local.index)

    # keep columns（存在するものだけ）
    for k in keep_cols:
        if k in df_local.columns:
            out[k] = df_local[k]

    out["combo"] = df_local["combo"]
    out["a_lane"] = a.astype(int)
    out["b_lane"] = b.astype(int)
    out["c_lane"] = c.astype(int)

    if label_col and label_col in df_local.columns:
        out[label_col] = df_local[label_col]

    # basenames（共通特徴）
    basenames = list(next(iter(lane_cols.values())).keys())

    # 便利：laneごとの値をまとめた配列を作り、a/b/cで引く
    # ただし列ごとに作ると重いので、basenameごとに処理
    for base in basenames:
        # lane1..6のSeriesを順に
        s_list = []
        for lane in LANES:
            col = lane_cols[lane][base]
            s_list.append(df_local[col])

        # 0-indexにして選択
        # a,b,cは1..6 -> 0..5
        ai = (a.astype(int) - 1).to_numpy()
        bi = (b.astype(int) - 1).to_numpy()
        ci = (c.astype(int) - 1).to_numpy()

        # 値をnumpyで取り出し（objectも許容）
        mat = np.vstack([np.array(s) for s in s_list])  # shape (6, n)

        aval = mat[ai, np.arange(len(df_local))]
        bval = mat[bi, np.arange(len(df_local))]
        cval = mat[ci, np.arange(len(df_local))]

        out[f"a_{base}"] = aval
        out[f"b_{base}"] = bval
        out[f"c_{base}"] = cval

        # numeric派生（数値変換できるものだけ差分を追加）
        a_num = _safe_float_series(out[f"a_{base}"])
        b_num = _safe_float_series(out[f"b_{base}"])
        c_num = _safe_float_series(out[f"c_{base}"])

        # NaNが多すぎる（ほぼ非数値）なら派生を作らない
        non_nan_ratio = float(pd.concat([a_num, b_num, c_num], axis=1).notna().mean().mean())
        if non_nan_ratio >= 0.6:
            out[f"ab_diff_{base}"] = a_num - b_num
            out[f"ac_diff_{base}"] = a_num - c_num
            out[f"bc_diff_{base}"] = b_num - c_num
            out[f"abc_sum_{base}"] = a_num + b_num + c_num
            out[f"abc_mean_{base}"] = (a_num + b_num + c_num) / 3.0

    # combo構造の簡易特徴
    out["is_a_inside12"] = out["a_lane"].isin([1, 2]).astype(int)
    out["is_b_inside12"] = out["b_lane"].isin([1, 2]).astype(int)
    out["is_c_inside12"] = out["c_lane"].isin([1, 2]).astype(int)
    out["abc_is_sorted"] = ((out["a_lane"] < out["b_lane"]) & (out["b_lane"] < out["c_lane"])).astype(int)

    return out


def guess_label_column(columns: List[str]) -> Optional[str]:
    # ありがちなラベル列名候補
    candidates = ["label", "y", "target", "is_hit", "hit", "result", "win", "trifecta_hit"]
    for c in candidates:
        if c in columns:
            return c
    return None


def default_keep_columns(columns: List[str]) -> List[str]:
    # ありがちな識別子（存在するものだけ保存）
    candidates = [
        "race_id",
        "id",
        "date",
        "hd",
        "venue",
        "venue_code",
        "jcd",
        "race_no",
        "rno",
    ]
    return [c for c in candidates if c in columns]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/datasets/trifecta_train.csv", help="input csv path")
    ap.add_argument("--output", default="data/datasets/trifecta_train_features.csv", help="output csv path")
    ap.add_argument("--chunksize", type=int, default=200000, help="pandas read_csv chunksize")
    ap.add_argument("--max-rows", type=int, default=0, help="limit total processed rows (0 = no limit)")
    ap.add_argument("--encoding", default="utf-8", help="csv encoding (utf-8 / cp932 etc)")
    ap.add_argument("--sep", default=",", help="csv separator")
    ap.add_argument("--label-col", default="", help="label column name (empty = auto-guess)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output

    if not os.path.exists(in_path):
        print(f"[ERROR] input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # ヘッダだけ読んで列推定
    head = pd.read_csv(in_path, nrows=5, encoding=args.encoding, sep=args.sep)
    cols = list(head.columns)

    if "combo" not in cols:
        print("[ERROR] input CSV must have 'combo' column.", file=sys.stderr)
        print(f"        columns: {cols[:50]} ...", file=sys.stderr)
        sys.exit(1)

    label_col = args.label_col.strip() or guess_label_column(cols)
    keep_cols = default_keep_columns(cols)

    # laneごとの元特徴量列を検出
    lane_cols = detect_lane_feature_columns(cols)

    if args.verbose:
        basenames = list(next(iter(lane_cols.values())).keys())
        print(f"[INFO] detected {len(basenames)} common per-lane basenames")
        print(f"[INFO] label_col = {label_col}")
        print(f"[INFO] keep_cols = {keep_cols}")
        print(f"[INFO] output = {out_path}")

    # 出力ファイルは上書き（最初に消す）
    if os.path.exists(out_path):
        os.remove(out_path)

    total = 0
    first_write = True

    reader = pd.read_csv(
        in_path,
        encoding=args.encoding,
        sep=args.sep,
        chunksize=args.chunksize,
        low_memory=False,
    )

    for chunk_idx, df in enumerate(reader, start=1):
        if args.max_rows and total >= args.max_rows:
            break

        if args.max_rows:
            remain = args.max_rows - total
            if remain <= 0:
                break
            if len(df) > remain:
                df = df.iloc[:remain].copy()

        feat_df = build_features_for_chunk(
            df=df,
            lane_cols=lane_cols,
            keep_cols=keep_cols,
            label_col=label_col,
        )

        feat_df.to_csv(
            out_path,
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
            encoding="utf-8",
        )
        first_write = False

        total += len(df)
        if args.verbose:
            print(f"[INFO] chunk {chunk_idx}: wrote {len(df):,} rows (total={total:,})")

    print(f"[DONE] wrote: {out_path}")
    print(f"[DONE] total rows: {total:,}")


if __name__ == "__main__":
    main()
