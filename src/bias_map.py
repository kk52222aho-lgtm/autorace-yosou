"""人気-穴バイアスの地図（単勝プール）。モデル不要・市場構造のみ。

前回までは「モデル vs 市場」で edge を探した（全プール陰性）。本モジュールは切り口を
変え、**市場オッズそのものの偏り（favorite-longshot bias ＝ ②嗜好の歪みの古典形）**を
20,477レース全体で直接測る。パラメータ調整が無いので過学習が原理的に起きず、
whole-sample＋ブートストラップCIで正直に判定できる。

出す表:
  A) オッズ帯別: 頭数・平均インプライド確率(控除除去後)・実勝率・フラット単勝ROI・CI
  B) 人気順位別: 市場1番人気/2番人気/.../最下位人気のフラット単勝ROI・CI

判定: どの帯/順位でも ROI CI 上限が 100% を割れば「単勝プールは全帯で効率的」。
      逆にどこかで CI 下限が 100% を跨がず超えれば ②バイアスの実在候補。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage

RNG = np.random.RandomState(0)
# オッズ帯（下限含む・上限含まず）
BANDS = [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 7.0), (7.0, 10.0),
         (10.0, 15.0), (15.0, 30.0), (30.0, 1e9)]


def _load() -> pd.DataFrame:
    conn = storage.connect()
    ent = pd.read_sql_query(
        "SELECT date,jcd,rno,car,win FROM entries WHERE win IS NOT NULL", conn)
    od = pd.read_sql_query(
        "SELECT date,jcd,rno,car,odds FROM win_odds WHERE odds IS NOT NULL AND odds>0", conn)
    conn.close()
    ent["jcd"] = ent["jcd"].astype(str).str.zfill(2)
    od["jcd"] = od["jcd"].astype(str).str.zfill(2)
    df = ent.merge(od, on=["date", "jcd", "rno", "car"], how="inner")
    # レース内で人気順位(1=最低オッズ=1番人気) と 控除除去インプライド確率
    df["fav_rank"] = df.groupby(["date", "jcd", "rno"])["odds"].rank(method="first")
    inv = 1.0 / df["odds"]
    denom = inv.groupby([df["date"], df["jcd"], df["rno"]]).transform("sum")
    df["implied"] = inv / denom
    return df


def _roi_ci(odds: np.ndarray, win: np.ndarray, B: int = 10000):
    """フラット単勝ROI(%) と ブートCI(5-95%) と P(ROI<100%)。"""
    payoff = odds * win  # 当たり=odds倍, 外れ=0（100円賭けの倍率）
    n = len(payoff)
    if n == 0:
        return (float("nan"),) * 4 + (0,)
    roi = payoff.mean() * 100
    means = np.array([payoff[RNG.randint(0, n, n)].mean() for _ in range(B)]) * 100
    p5, p95 = np.percentile(means, [5, 95])
    return roi, p5, p95, float((means < 100).mean() * 100), n


def main() -> None:
    df = _load()
    nraces = df.groupby(["date", "jcd", "rno"]).ngroups
    print(f"データ: {len(df):,}車 / {nraces:,}レース / {df['date'].min()}..{df['date'].max()}\n")

    print("=== A) オッズ帯別（単勝フラット買い）===")
    print(f"{'帯':>12s}{'頭数':>9s}{'平均implied':>11s}{'実勝率':>9s}"
          f"{'ROI':>8s}{'CI(5-95%)':>16s}{'P<100%':>8s}")
    for lo, hi in BANDS:
        m = (df["odds"] >= lo) & (df["odds"] < hi)
        sub = df[m]
        if len(sub) == 0:
            continue
        roi, p5, p95, plt, n = _roi_ci(sub["odds"].to_numpy(), sub["win"].to_numpy())
        label = f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
        flag = " <-" if p5 >= 100 else ""
        print(f"{label:>12s}{n:9,d}{sub['implied'].mean()*100:10.1f}%"
              f"{sub['win'].mean()*100:8.1f}%{roi:7.0f}%"
              f"{f'[{p5:.0f},{p95:.0f}]':>16s}{plt:7.1f}%{flag}")

    print("\n=== B) 人気順位別（市場N番人気を単勝フラット買い）===")
    print(f"{'人気':>6s}{'頭数':>9s}{'平均odds':>9s}{'実勝率':>9s}"
          f"{'ROI':>8s}{'CI(5-95%)':>16s}{'P<100%':>8s}")
    for rank in range(1, 9):
        sub = df[df["fav_rank"] == rank]
        if len(sub) == 0:
            continue
        roi, p5, p95, plt, n = _roi_ci(sub["odds"].to_numpy(), sub["win"].to_numpy())
        flag = " <-" if p5 >= 100 else ""
        print(f"{rank:>5d}番{n:9,d}{sub['odds'].median():8.1f}"
              f"{sub['win'].mean()*100:8.1f}%{roi:7.0f}%"
              f"{f'[{p5:.0f},{p95:.0f}]':>16s}{plt:7.1f}%{flag}")

    print("\n ※控除率25%なら全帯ROI約75%が効率市場の帰無。どこかで有意に上振れれば②バイアス候補。")
    print(" ※ <- はブートCI下限≥100%（本物のエッジ候補）。")


if __name__ == "__main__":
    main()
