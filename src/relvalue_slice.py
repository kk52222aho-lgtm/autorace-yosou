"""買い目宇宙のスライス検定：evK(EV≥1)の中に勝てる部分集合があるか。

集約(evK全部)は負けでも、特定のオッズ帯/EV帯だけプラスかもしれん——を潰す。ただし
負け集団を割れば偶然プラスのバケツは必ず出る(p-hacking)。よって:
  - tune(前半0.6) で各バケツの床ROIを見て、
  - **同じバケツを test(後半0.4) に適用**して両方で残るか、
  - レース単位ブロックブートで CI、
を並べて出す。tune だけ良くて test で崩れる=スライスの蜃気楼。

確率源=市場単勝オッズ控除除去(最シャープ)。exotic 4プール(20k R)で検出力最大。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .relvalue import _races, POOLS

RNG = np.random.RandomState(0)
ODDS_BUCKETS = [(1, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 1e9)]
EV_BUCKETS = [(1.0, 1.1), (1.1, 1.3), (1.3, 1.6), (1.6, 2.5), (2.5, 1e9)]


def _collect(races):
    """各レースの候補ベット(EV≥1)を (date,pool,odds,ev,hit) の行にばらす。"""
    rows = []
    for d, pools in races:
        for pool, (real, items) in pools.items():
            for c, p, o in items:
                ev = p * o
                if ev >= 1.0 and o > 0:
                    rows.append((d, pool, o, ev, 1 if c == real else 0))
    df = pd.DataFrame(rows, columns=["date", "pool", "odds", "ev", "hit"])
    df["ret"] = df["odds"] * df["hit"] * 100
    return df


def _block_boot(df, B=8000):
    """レース(date+全行を1ブロックにはできんが、ここは行=独立ベット近似でなく
    date単位のブロック)で ROI 床の CI。df は同一バケツのベット行。"""
    if len(df) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    roi = df["ret"].sum() / (len(df) * 100) * 100
    # ブロック=date（同日・同レースのベット相関を粗く吸収）
    groups = [g[["ret"]].to_numpy() for _, g in df.groupby("date")]
    stakes = [len(g) * 100 for g in groups]
    rets = [g.sum() for g in groups]
    stakes, rets = np.array(stakes, float), np.array(rets, float)
    n = len(groups)
    boot = np.empty(B)
    for b in range(B):
        idx = RNG.randint(0, n, n)
        boot[b] = rets[idx].sum() / stakes[idx].sum() * 100
    p5, p95 = np.percentile(boot, [5, 95])
    return roi, p5, p95, float((boot < 100).mean() * 100), len(df)


def _report(df, col, buckets, title):
    print(f"\n=== {title}（tune床ROI → test床ROI）===")
    print(f"{'bucket':>12s}{'tune:bets':>10s}{'ROI':>7s}"
          f"{'|  test:bets':>13s}{'ROI':>7s}{'CI(5-95%)':>15s}{'P<100%':>8s}")
    cut = df["date"].quantile(0.6, interpolation="lower")
    for lo, hi in buckets:
        m = (df[col] >= lo) & (df[col] < hi)
        tune = df[m & (df["date"] < cut)]
        test = df[m & (df["date"] >= cut)]
        if len(tune) < 50 or len(test) < 50:
            continue
        troi = tune["ret"].sum() / (len(tune) * 100) * 100
        roi, p5, p95, plt, n = _block_boot(test)
        lab = f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
        flag = " ✓" if p5 >= 100 else (" ~" if roi >= 100 else "")
        print(f"{lab:>12s}{len(tune):10d}{troi:6.0f}%"
              f"{len(test):13d}{roi:6.0f}%{f'[{p5:.0f},{p95:.0f}]':>15s}{plt:7.1f}%{flag}")


def main():
    print("確率源=市場単勝控除除去。exotic候補(EV≥1)をスライス...")
    df = _collect(_races())
    print(f"候補ベット総数 {len(df):,} / {df['date'].min()}..{df['date'].max()}")

    # プール別 × オッズ帯
    for pool in POOLS:
        sub = df[df["pool"] == pool]
        if len(sub) < 200:
            continue
        _report(sub, "odds", ODDS_BUCKETS, f"{pool}：オッズ帯別")

    # 全プール込み × EV帯（割高さの強さでより分ける）
    _report(df, "ev", EV_BUCKETS, "全プール：EV帯別")

    print("\n ✓=testのCI下限≥100%(本物のスライス) / ~=点ROI≥100%だがCI跨ぐ / 無印=<100%")
    print(" ※tuneだけ高くtestで崩れる帯はスライスの蜃気楼。両方で残る帯のみ本物。")


if __name__ == "__main__":
    main()
