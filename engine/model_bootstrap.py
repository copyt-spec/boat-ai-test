from __future__ import annotations

import gzip
import os
import shutil
import urllib.request


MODEL_DIR = "data/models"
MODEL_PATH = os.path.join(MODEL_DIR, "trifecta120_model.joblib")
MODEL_GZ_PATH = os.path.join(MODEL_DIR, "trifecta120_model.joblib.gz")


def ensure_model_ready() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    # すでに展開済みなら何もしない
    if os.path.exists(MODEL_PATH):
        print(f"[MODEL] already exists: {MODEL_PATH}")
        return

    # gzip がローカルにあれば展開だけ
    if os.path.exists(MODEL_GZ_PATH):
        print(f"[MODEL] extracting local gzip: {MODEL_GZ_PATH}")
        _extract_gzip(MODEL_GZ_PATH, MODEL_PATH)
        return

    # 環境変数からURL取得
    model_url = os.getenv("MODEL_URL", "").strip()
    if not model_url:
        raise FileNotFoundError(
            "Model not found locally and MODEL_URL is empty. "
            "Set MODEL_URL in Render Environment Variables."
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
