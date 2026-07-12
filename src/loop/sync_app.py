"""Streamlit Cloud 同期：predictions テーブルを軽量CSVに書き出し、git push する。

669MB DB は上げられん/持てんので、Cloudアプリが読む predictions.csv(数十KB)だけを
更新して push。毎晩の予想・突合の後に呼ぶ想定。

  python -m src.loop.sync_app            # CSV書き出し + commit + push
  python -m src.loop.sync_app --no-push  # 書き出しのみ
"""
from __future__ import annotations

import argparse
import subprocess

import pandas as pd

from .. import storage

CSV = "data/predictions.csv"


def export_csv() -> int:
    conn = storage.connect()
    try:
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    df.to_csv(CSV, index=False)
    return len(df)


def _git(*args):
    return subprocess.run(["git", *args], cwd=".", capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    n = export_csv()
    print(f"predictions.csv 書き出し: {n}行")
    if args.no_push:
        return
    _git("add", CSV)
    r = _git("commit", "-m", f"data: predictions.csv 更新 ({n}行)")
    if r.returncode != 0 and "nothing to commit" in (r.stdout + r.stderr):
        print("変更なし(commitスキップ)")
        return
    p = _git("push")
    print("push:", "OK" if p.returncode == 0 else p.stderr.strip()[:200])


if __name__ == "__main__":
    main()
