"""最後の未収集プール＝ワイド(quinella_place)を実スクレイプして検定。

ワイドは race詳細の odds フィールドが0で弾かれ未収集だった。専用 /odds エンドポイントは
minOdds/maxOdds のレンジで返す。ここでは**保守側=minOdds**(実払戻はこれ以上)で ROI 床を測る。
床でも壁(≈70-75%)前後なら本物のエッジは無い。

確率源は relvalue と同じく**市場単勝オッズ(控除除去)**。ワイド{i,j}の的中確率は
「i,jが共に3着以内」＝ Σ_k trio_prob({i,j,k})。realized は winners['quinella_place']。

  python -m src.wide_probe --n 800
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from . import storage, winticket
from .backtest_multi import _harville


def _wide_probs(wp: dict[int, float]) -> dict[str, float]:
    """単勝確率 wp から ワイド{i,j}=P(i,jともに3着内) を Harville trio 集約で。"""
    h = _harville(wp)
    trio = h.get("trio", {})
    out: dict[str, float] = {}
    cars = sorted(wp)
    for i, j in combinations(cars, 2):
        s = 0.0
        for k in cars:
            if k == i or k == j:
                continue
            s += trio.get("-".join(map(str, sorted([i, j, k]))), 0.0)
        out[f"{i}-{j}"] = s
    return out


def _sample_races(n: int):
    conn = storage.connect()
    rows = conn.execute(
        "SELECT DISTINCT date,jcd,rno FROM entries WHERE win IS NOT NULL "
        "ORDER BY date DESC, jcd, rno LIMIT ?", (n,)).fetchall()
    win = {}
    for d, j, r in rows:
        cur = conn.execute(
            "SELECT car,odds FROM win_odds WHERE date=? AND jcd=? AND rno=? AND odds>0",
            (d, j, r))
        win[(d, j, r)] = {int(c): float(o) for c, o in cur.fetchall()}
        cur2 = conn.execute(
            "SELECT car,finish FROM entries WHERE date=? AND jcd=? AND rno=? AND finish IS NOT NULL",
            (d, j, r))
        win[(d, j, r, "fin")] = {int(c): int(f) for c, f in cur2.fetchall()}
    conn.close()
    return rows, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--min-ev", type=float, default=1.0)
    args = ap.parse_args()

    rows, cache = _sample_races(args.n)
    print(f"直近{len(rows)}レースのワイドを実スクレイプ（minOdds=保守側）...")

    # 集計: strategy -> [staked, returned, bets, hits]
    acc = {s: [0, 0, 0, 0] for s in ["market", "relval_ev1", "relval_evK"]}
    got = 0
    for idx, (d, j, r) in enumerate(rows):
        wo = cache.get((d, j, r)) or {}
        fin = cache.get((d, j, r, "fin")) or {}
        if not wo or not fin:
            continue
        race = winticket.fetch_race(d, j, r, with_odds=True)
        if not race:
            continue
        # ワイドの minOdds を専用エンドポイントから
        cup = winticket.resolve_cup(d, j)
        det = winticket.cup_detail(cup) if cup else None
        sch = winticket._schedule(det, d) if det else None
        if not sch:
            continue
        od = winticket._get(
            f"{winticket.BASE}/cups/{cup}/schedules/{sch['index']}/races/{r}/odds")
        if not od:
            continue
        wide = {}
        for e in od.get("quinellaPlace", []) or []:
            key = "-".join(str(x) for x in e.get("key", []))
            mn = e.get("minOdds") or 0
            if key and mn > 0:
                wide[key] = float(mn)
        if not wide:
            continue
        winners = set(race.get("winners", {}).get("quinella_place", []) or [])
        if not winners:
            continue
        got += 1

        # 確率源=単勝控除除去
        inv = {c: 1 / o for c, o in wo.items() if o > 0}
        ssum = sum(inv.values())
        wp = {c: v / ssum for c, v in inv.items()}
        wpr = _wide_probs(wp)

        # market: 最低オッズ1点
        mkt = min(wide, key=lambda c: wide[c])
        _bet(acc["market"], mkt, wide, winners)
        # relval: EV(=単勝Harville確率×minOdds)≥閾値
        cand = [(c, wpr.get(c, 0) * o, o) for c, o in wide.items()
                if wpr.get(c, 0) * o >= args.min_ev]
        if cand:
            best = max(cand, key=lambda x: wpr.get(x[0], 0))  # 割安な中で最確
            _bet(acc["relval_ev1"], best[0], wide, winners)
            for c, ev, o in cand:
                _bet(acc["relval_evK"], c, wide, winners)
        if (idx + 1) % 100 == 0:
            print(f"  ...{idx+1}/{len(rows)} 取得{got}")

    print(f"\n=== ワイド検定（{got}レース / minOdds床 / EV≥{args.min_ev}）===")
    print(f"{'strategy':12s}{'bets':>7s}{'hit':>8s}{'ROI床':>8s}")
    for s, (st, rt, b, h) in acc.items():
        if b == 0:
            continue
        print(f"{s:12s}{b:7d}{h/b*100:7.1f}%{rt/st*100:7.1f}%")
    print(" ※minOdds床。実払戻は床以上なので、床が壁(≈70-75%)以下なら本物のエッジは無い。")


def _bet(a, combo, wide, winners):
    o = wide.get(combo)
    if not o:
        return
    a[2] += 1
    a[0] += 100
    if combo in winners:
        a[3] += 1
        a[1] += o * 100


if __name__ == "__main__":
    main()
