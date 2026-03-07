# scripts/build_dataset_from_txt.py

import os
from engine.txt_dataset_builder import TxtDatasetConfig, build_dataset_from_txt

def main():
    raw_dir = os.path.join("data", "raw_txt")
    out_csv = os.path.join("data", "datasets", "startk_dataset.csv")

    # まず「ラベル無しも含めて」作ってみる（原因切り分け用）
    # これで行数が爆増するなら → 払戻(3連単)だけ取れてない
    # これでも増えないなら → そもそもレースブロックが拾えてない
    cfg = TxtDatasetConfig(
        raw_txt_dir=raw_dir,
        out_csv_path=out_csv,
        keep_unlabeled=False,
        verbose=True
    )

    build_dataset_from_txt(cfg)

    # 生成結果の行数（ヘッダ込み）を表示
    try:
        with open(out_csv, "r", encoding="utf-8", errors="ignore") as f:
            lines = sum(1 for _ in f)
        print("CSV lines (including header):", lines)
        print("CSV races:", max(0, lines - 1))
    except Exception as e:
        print("Could not read output csv:", e)

    print("\n次：keep_unlabeled=False に戻して学習用を作る（ラベル取れる状態になったら）")

if __name__ == "__main__":
    main()
