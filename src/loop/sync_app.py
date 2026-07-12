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
CSV_R = "data/race_results.csv"


def export_csv() -> int:
    conn = storage.connect()
    n = 0
    for tbl, path in [("predictions", CSV), ("race_results", CSV_R)]:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {tbl}", conn)
        except Exception:
            df = pd.DataFrame()
        df.to_csv(path, index=False)
        if tbl == "predictions":
            n = len(df)
    conn.close()
    return n


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
    _git("add", CSV, CSV_R)
    r = _git("commit", "-m", f"data: predictions.csv 更新 ({n}行)")
    if r.returncode != 0 and "nothing to commit" in (r.stdout + r.stderr):
        print("変更なし(commitスキップ)")
        return
    p = _git("push")
    print("push:", "OK" if p.returncode == 0 else p.stderr.strip()[:200])


if __name__ == "__main__":
    main()
