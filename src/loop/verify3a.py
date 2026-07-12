"""検証3-A：記憶vs市場の食い違い(gap=記憶確率−市場含意)を方向×大きさで層別。
value全体68%の下に、特定gap帯だけ控除を越える受け皿があるか。無ければ全帯で確定。

Part1(記憶>市場=強気を買う): gap>0 の車を gap帯別に単勝フラットROI(レース単位ブート)。
Part2(記憶<市場=市場過剰人気を消す): 市場本命を記憶が強気/弱気で層別。弱気本命が
  実際に負けるなら「危険な本命の回避(消し)」に価値。弱気本命を見送る戦略で本命ROIが上がるか。
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import env as E
from .verify2 import _memory_car_probs

RNG = np.random.RandomState(0)
K = 80


def _block_boot(recs, B=10000):
    """recs: [(race_id, odds, hit)]。レース単位ブロックブートでROI床/CI。"""
    if not recs:
        return float("nan"), float("nan"), float("nan"), 0, 0.0
    races = {}
    for rid, o, h in recs:
        races.setdefault(rid, []).append(o if h else 0.0)
    keys = list(races)
    stakes = {k: len(v) for k, v in races.items()}
    rets = {k: sum(v) for k, v in races.items()}
    tot_ret = sum(rets.values()); tot_stk = sum(stakes.values())
    roi = tot_ret / tot_stk * 100
    hit = sum(1 for _, _, h in recs if h) / len(recs) * 100
    n = len(keys)
    ka = np.array(keys, dtype=object)
    rr = np.array([rets[k] for k in keys]); ss = np.array([stakes[k] for k in keys])
    boot = np.empty(B)
    for b in range(B):
        idx = RNG.randint(0, n, n)
        boot[b] = rr[idx].sum() / ss[idx].sum() * 100
    return roi, np.percentile(boot, 5), np.percentile(boot, 95), len(recs), hit


def main():
    races = E.load_block_a()
    mu, sd = E.standardizer(races)
    for r in races:
        r.emb_std = (r.emb - mu) / sd
    months = sorted({r.ym for r in races})
    by = {m: [r for r in races if r.ym == m] for m in months}

    memX, memY = [], []
    bull = []          # (rid, odds, hit, gap)  記憶>市場
    fav_recs = []      # (rid, odds, hit, fav_gap)  市場本命
    for mi, m in enumerate(months):
        if mi > 0 and len(memX) >= K:
            nn = NearestNeighbors(n_neighbors=min(K, len(memX)))
            nn.fit(np.vstack(memX)); ya = np.array(memY)
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
                rid = r.key
                for c in mp:
                    gap = mp[c] - imp.get(c, 0)
                    hit = (c == r.winner_car)
                    if gap > 0:
                        bull.append((rid, r.win_odds[c], hit, gap))
                fav = min(r.win_odds, key=lambda c: r.win_odds[c])
                fav_recs.append((rid, r.win_odds[fav], fav == r.winner_car,
                                 mp.get(fav, 0) - imp.get(fav, 0)))
        for r in by[m]:
            memX.append(r.emb_std); memY.append(r.winner_rank)

    print("=== Part1: 記憶>市場(強気)の車を買う・gap帯別 単勝ROI(レース単位ブート) ===")
    print(f"{'gap帯':>12s}{'bets':>7s}{'的中':>7s}{'ROI':>7s}{'CI(5-95)':>14s}")
    for lo, hi in [(0, .02), (.02, .05), (.05, .10), (.10, .20), (.20, 1.0)]:
        sub = [(rid, o, h) for rid, o, h, g in bull if lo <= g < hi]
        roi, p5, p95, n, hit = _block_boot(sub)
        if n < 50:
            continue
        f = " ✓" if p5 >= 100 else ""
        print(f"{f'{lo:.2f}-{hi:.2f}':>12s}{n:7d}{hit:6.1f}%{roi:6.0f}%{f'[{p5:.0f},{p95:.0f}]':>14s}{f}")

    print("\n=== Part2: 市場本命を記憶が強気/弱気で層別(消しの価値) ===")
    print(f"{'本命の記憶view':>16s}{'races':>7s}{'本命勝率':>9s}{'本命ROI':>8s}{'CI(5-95)':>14s}")
    for name, cond in [("強気(記憶≥市場)", lambda g: g >= 0),
                       ("弱気(記憶<市場)", lambda g: g < 0),
                       ("強弱気弱め(gap<-0.05)", lambda g: g < -0.05)]:
        sub = [(rid, o, h) for rid, o, h, g in fav_recs if cond(g)]
        roi, p5, p95, n, hit = _block_boot(sub)
        if n < 50:
            continue
        print(f"{name:>16s}{n:7d}{hit:8.1f}%{roi:7.0f}%{f'[{p5:.0f},{p95:.0f}]':>14s}")
    # 消し戦略: 弱気本命を見送り、強気本命だけ買う → 本命ベタ70%を超えるか
    keep = [(rid, o, h) for rid, o, h, g in fav_recs if g >= 0]
    roi, p5, p95, n, hit = _block_boot(keep)
    print(f"\n 消し戦略(弱気本命を見送り強気本命だけ単勝): ROI {roi:.0f}% [{p5:.0f},{p95:.0f}] "
          f"(本命ベタ70%比 {'改善' if roi>71 else '不変〜悪化'})")
    print("\n ✓=CI下限≥100%。どのgap帯も✓無しなら食い違いは全帯で儲けにならん=競艇型エッジ無し確定。")


if __name__ == "__main__":
    main()
