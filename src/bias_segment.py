"""②バイアスの居所スキャン：単勝の効率をセグメント別に割る。

人が集まる大レース(高グレード)・特定場・頭数・天候など、casual マネーが厚く
②嗜好の歪みが最も乗りやすい断面で、市場1〜2番人気の単勝フラットROIが控除の壁を
超えるか。全セグメントを（cherry-pick せず）CI 付きで列挙する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage

RNG = np.random.RandomState(0)


def _load() -> pd.DataFrame:
    conn = storage.connect()
    ent = pd.read_sql_query(
        "SELECT date,jcd,rno,car,win,race_class,weather,track_cond FROM entries WHERE win IS NOT NULL", conn)
    od = pd.read_sql_query(
        "SELECT date,jcd,rno,car,odds FROM win_odds WHERE odds IS NOT NULL AND odds>0", conn)
    conn.close()
    ent["jcd"] = ent["jcd"].astype(str).str.zfill(2)
    od["jcd"] = od["jcd"].astype(str).str.zfill(2)
    df = ent.merge(od, on=["date", "jcd", "rno", "car"], how="inner")
    df["fav_rank"] = df.groupby(["date", "jcd", "rno"])["odds"].rank(method="first")
    df["ncars"] = df.groupby(["date", "jcd", "rno"])["car"].transform("size")
    return df


def _roi_ci(odds, win, B=8000):
    payoff = odds * win
    n = len(payoff)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    roi = payoff.mean() * 100
    means = np.array([payoff[RNG.randint(0, n, n)].mean() for _ in range(B)]) * 100
    p5, p95 = np.percentile(means, [5, 95])
    return roi, p5, p95, n


def _scan(df, col, label):
    print(f"\n=== {label}別：市場1番人気の単勝フラットROI ===")
    print(f"{'セグメント':>16s}{'頭数':>8s}{'勝率':>8s}{'ROI':>7s}{'CI(5-95%)':>15s}")
    fav = df[df["fav_rank"] == 1]
    vc = fav[col].value_counts()
    for val in vc.index:
        sub = fav[fav[col] == val]
        if len(sub) < 150:
            continue
        roi, p5, p95, n = _roi_ci(sub["odds"].to_numpy(), sub["win"].to_numpy())
        flag = " <-" if p5 >= 100 else ""
        print(f"{str(val):>16s}{n:8,d}{sub['win'].mean()*100:7.1f}%"
              f"{roi:6.0f}%{f'[{p5:.0f},{p95:.0f}]':>15s}{flag}")


def main():
    df = _load()
    print(f"データ: {df.groupby(['date','jcd','rno']).ngroups:,}レース")
    for col, lab in [("race_class", "グレード"), ("jcd", "場"),
                     ("ncars", "出走頭数"), ("track_cond", "走路")]:
        _scan(df, col, lab)
    print("\n ※控除の壁≈75%。全セグメントで1番人気ROIが75%前後なら②の居所は無い。")
    print(" ※ <- はCI下限≥100%（本物のエッジ候補）。")


if __name__ == "__main__":
    main()
