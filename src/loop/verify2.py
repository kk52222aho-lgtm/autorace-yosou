"""検証1の続き(Task1+2)：読み出しを argmax→較正済み確率に替え、素の記憶確率で
単勝を買ったROIがサイコロ・本命ベタを超えるか。RL方策は通さない。

Task1(読み出し): kNN近傍の「勝者試走順位」を距離重みで分布化(argmax廃止)→各車に
  その順位スロットの確率を配る = 較正済み per-car 勝率。
Task2(素のROI): WF・オッズ隔離で単勝フラット。戦略:
  dice    : レース内ランダム1車
  fav     : 市場本命(最低オッズ)
  mem_top : 記憶確率が最大の車
  value   : (記憶確率 − 市場含意)が最大の車 = 市場が最も過小評価と記憶が言う車
  value+  : value のうち gap>0 だけ(市場超過分がある時だけ賭ける, 他は見送り)

  python -m src.loop.verify2
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import env as E

RNG = np.random.RandomState(0)
K = 80


def _memory_car_probs(dd, ii, ya, cars_by_rank, win_odds):
    """近傍から各順位の勝率分布を作り、存在する車に配る。→ {car: prob}。"""
    w = 1.0 / (dd + 1e-6)
    rank_p = np.zeros(E.MAXC + 1)
    for lab, wi in zip(ya[ii], w):
        if 1 <= lab <= E.MAXC:
            rank_p[lab] += wi
    rank_p /= rank_p.sum() + 1e-12
    probs = {}
    for r in range(1, E.MAXC + 1):
        car = cars_by_rank[r - 1]
        if car is not None and car in win_odds:
            probs[car] = rank_p[r]
    s = sum(probs.values())
    return {c: p / s for c, p in probs.items()} if s > 0 else {}


def _roi_boot(bets, B=10000):
    if not bets:
        return float("nan"), float("nan"), float("nan"), 0
    pay = np.array([o if hit else 0.0 for o, hit in bets])
    roi = pay.mean() * 100
    n = len(pay)
    boot = np.array([pay[RNG.randint(0, n, n)].mean() for _ in range(B)]) * 100
    return roi, np.percentile(boot, 5), np.percentile(boot, 95), n


def main():
    races = E.load_block_a()
    mu, sd = E.standardizer(races)
    for r in races:
        r.emb_std = (r.emb - mu) / sd
    months = sorted({r.ym for r in races})
    by = {m: [r for r in races if r.ym == m] for m in months}

    memX, memY = [], []
    strat = {s: [] for s in ["dice", "fav", "mem_top", "value", "value+"]}
    for mi, m in enumerate(months):
        if mi > 0 and len(memX) >= K:
            nn = NearestNeighbors(n_neighbors=min(K, len(memX)))
            nn.fit(np.vstack(memX))
            ya = np.array(memY)
            Q = np.vstack([r.emb_std for r in by[m]])
            D, I = nn.kneighbors(Q)
            for r, dd, ii in zip(by[m], D, I):
                if len(r.win_odds) < 2:
                    continue
                mp = _memory_car_probs(dd, ii, ya, r.cars_by_rank, r.win_odds)
                if not mp:
                    continue
                s = sum(1.0 / o for o in r.win_odds.values() if o > 0)
                imp = {c: (1.0 / o) / s for c, o in r.win_odds.items() if o > 0}
                won = lambda c: c == r.winner_car
                cars = list(mp.keys())
                # dice
                dc = RNG.choice(cars)
                strat["dice"].append((r.win_odds[dc], won(dc)))
                # fav
                fv = min(r.win_odds, key=lambda c: r.win_odds[c])
                strat["fav"].append((r.win_odds[fv], won(fv)))
                # mem_top
                mt = max(mp, key=lambda c: mp[c])
                strat["mem_top"].append((r.win_odds[mt], won(mt)))
                # value: max(mem - implied)
                vc = max(cars, key=lambda c: mp[c] - imp.get(c, 0))
                strat["value"].append((r.win_odds[vc], won(vc)))
                # value+: gap>0 の時だけ
                if mp[vc] - imp.get(vc, 0) > 0:
                    strat["value+"].append((r.win_odds[vc], won(vc)))
        for r in by[m]:
            memX.append(r.emb_std)
            memY.append(r.winner_rank)

    print(f"WF・オッズ隔離・単勝フラット。記憶読み出し=較正済み確率(argmax廃止)\n")
    print(f"{'戦略':>10s}{'bets':>7s}{'的中率':>8s}{'ROI':>7s}{'ブートCI':>15s}")
    for s in ["dice", "fav", "mem_top", "value", "value+"]:
        roi, p5, p95, n = _roi_boot(strat[s])
        hit = np.mean([h for _, h in strat[s]]) * 100 if strat[s] else 0
        f = " ✓" if p5 >= 100 else ""
        print(f"{s:>10s}{n:7d}{hit:7.1f}%{roi:6.0f}%{f'[{p5:.0f},{p95:.0f}]':>15s}{f}")
    print("\n 関門: mem_top/value がサイコロ・本命ベタを明確に超えるか。超えれば生きた確率は")
    print(" 賭けに使える(次はRL)。全部同じ壁(≈70-75%)なら記憶確率も市場に飲まれてる。")


if __name__ == "__main__":
    main()
