"""回収率バックテスト（単勝・払戻ベース）＋時系列ウォークフォワード予測。

オートレースは**単勝が発売される**ので、競艇と同じ「モデル本命の単勝を買う」を
そのまま検証できる。候補戦略＝「モデル本命が市場本命(単勝最低オッズ)以外のレース
だけ、その本命を単勝で買う」（市場の穴を突けるかの妙味狙い）。

例:
  python -m src.backtest            # value: 本命≠市場本命のみ
  python -m src.backtest --bet all  # 全レース単勝（ベース比較）
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import history, storage
from .features import FEATURES, build_frame
from .train import _make_estimator


def load() -> pd.DataFrame:
    conn = storage.connect()
    ent = pd.read_sql_query("SELECT * FROM entries WHERE win IS NOT NULL", conn)
    odds = pd.read_sql_query("SELECT date,jcd,rno,car,odds AS win_odds FROM win_odds", conn)
    conn.close()
    ent["jcd"] = ent["jcd"].astype(str).str.zfill(2)
    odds["jcd"] = odds["jcd"].astype(str).str.zfill(2)
    ent = ent.merge(odds, on=["date", "jcd", "rno", "car"], how="left")
    return history.add_history(ent)


def _walk_predict(df: pd.DataFrame, n_folds: int = 4) -> np.ndarray:
    """日付を n_folds ブロックに区切り、各ブロックを「それ以前の日付だけ」で
    学習して予測（リークなし時系列WF）。先頭ブロックは学習元が無く NaN。"""
    df = df.reset_index(drop=True)
    dates = np.sort(df["date"].unique())
    bounds = [dates[int(len(dates) * k / n_folds)] for k in range(1, n_folds)]
    edges = [dates[0]] + bounds + [None]

    d = df["date"].to_numpy()
    feat = build_frame(df, impute=False)[FEATURES].to_numpy(dtype=float)
    y = df["win"].to_numpy(dtype=int)
    proba = np.full(len(df), np.nan)

    for k in range(1, len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        te = (d >= lo) if hi is None else ((d >= lo) & (d < hi))
        tr = d < lo
        if tr.sum() < 50 or te.sum() == 0:
            continue
        m = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
        m.fit(feat[tr], y[tr])
        proba[te] = m.predict_proba(feat[te])[:, 1]
    return proba


def backtest(bet: str = "value") -> None:
    df = load()
    if len(df) < 200:
        print(f"データ不足（{len(df)}行）。src.collect で収集してください。")
        return
    df = df.reset_index(drop=True)
    df["proba"] = _walk_predict(df)
    df = df[df["proba"].notna()]

    staked = returned = bets = hits = 0
    for _, g in df.groupby(["date", "jcd", "rno"]):
        if g["proba"].isna().any() or g["win_odds"].isna().all():
            continue
        honmei = g.loc[g["proba"].idxmax()]
        market = g.loc[g["win_odds"].idxmin()]  # 市場本命
        if bet == "value" and honmei["car"] == market["car"]:
            continue
        bets += 1
        staked += 100
        if honmei["win"] == 1:
            hits += 1
            returned += float(honmei["win_odds"]) * 100
    if bets == 0:
        print("対象ベットなし。")
        return
    roi = returned / staked * 100
    print(f"=== 単勝バックテスト（{bet} / モデル本命1点）===")
    print(f"  ベット数 : {bets}")
    print(f"  的中率   : {hits / bets:.1%}")
    print(f"  回収率   : {roi:.1f}%   ({'✓ プラス' if roi >= 100 else '× マイナス'})")
    print("  ※確定オッズ前提。実弾は締切変動で乖離しうる。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bet", choices=["value", "all"], default="value")
    args = ap.parse_args()
    backtest(args.bet)


if __name__ == "__main__":
    main()
