"""マルチプールEVバックテスト（Harville × 全賭式の確定オッズ台帳）。

「単勝が効率的でも exotic(3連単等) も効率的とは限らん」を実測する。単勝の1着確率
p_i から Harville(Plackett-Luce の逐次抽出) で各組合せ確率を導出し、実オッズと EV
比較して券種ごとに回収率を出す。

  exacta   i>j        : p_i · p_j/(1-p_i)
  trifecta i>j>k      : p_i · p_j/(1-p_i) · p_k/(1-p_i-p_j)
  quinella {i,j}      : P(i>j)+P(j>i)
  trio     {i,j,k}    : 6通りの trifecta 確率の和

各プールで戦略を比較：
  market … 最低オッズ(市場本命)1点   ← 市場効率の基準
  prob   … モデル最確1点              ← 当てにいく
  ev1    … 最大EV1点                  ← 割安1点
  evK    … EV≥閾値を全点              ← 割安総取り

例:
  python -m src.backtest_multi
  python -m src.backtest_multi --min-ev 1.15 --max-odds 200
"""
from __future__ import annotations

import argparse
from itertools import permutations

import numpy as np
import pandas as pd

from . import storage
from .backtest import _walk_predict


def _harville(p: dict[int, float]) -> dict[str, dict[str, float]]:
    """1着確率 p{car:prob} から各賭式の組合せ確率を返す（combo文字列→確率）。"""
    cars = [c for c in p if p[c] > 0]
    s = sum(p[c] for c in cars)
    if s <= 0:
        return {}
    p = {c: p[c] / s for c in cars}

    exacta, trifecta = {}, {}
    for i, j in permutations(cars, 2):
        di = 1 - p[i]
        if di <= 1e-9:
            continue
        exacta[f"{i}-{j}"] = p[i] * p[j] / di
    for i, j, k in permutations(cars, 3):
        di, dij = 1 - p[i], 1 - p[i] - p[j]
        if di <= 1e-9 or dij <= 1e-9:
            continue
        trifecta[f"{i}-{j}-{k}"] = p[i] * p[j] / di * p[k] / dij

    quinella, trio = {}, {}
    for a, b in permutations(cars, 2):
        if a < b:
            quinella[f"{a}-{b}"] = exacta.get(f"{a}-{b}", 0) + exacta.get(f"{b}-{a}", 0)
    for combo in {tuple(sorted(t)) for t in permutations(cars, 3)}:
        tot = 0.0
        for perm in permutations(combo):
            tot += trifecta.get("-".join(map(str, perm)), 0)
        trio["-".join(map(str, combo))] = tot
    return {"exacta": exacta, "trifecta": trifecta,
            "quinella": quinella, "trio": trio}


def _realized(order: dict[int, int]) -> dict[str, str]:
    """{car:着順} から各賭式の的中 combo 文字列を作る。"""
    top = sorted(order, key=lambda c: order[c])[:3]
    if len(top) < 3:
        return {}
    a, b, c = top
    return {
        "exacta": f"{a}-{b}",
        "trifecta": f"{a}-{b}-{c}",
        "quinella": "-".join(map(str, sorted([a, b]))),
        "trio": "-".join(map(str, sorted([a, b, c]))),
    }


def _load():
    conn = storage.connect()
    ent = pd.read_sql_query("SELECT * FROM entries WHERE win IS NOT NULL", conn)
    od = pd.read_sql_query("SELECT date,jcd,rno,bet,combo,odds FROM odds", conn)
    conn.close()
    ent["jcd"] = ent["jcd"].astype(str).str.zfill(2)
    od["jcd"] = od["jcd"].astype(str).str.zfill(2)
    return ent, od


POOLS = ["exacta", "quinella", "trio", "trifecta"]


def backtest(min_ev: float = 1.0, max_odds: float = 300.0,
             min_prob: float = 0.0) -> None:
    ent, od = _load()
    if od.empty:
        print("オッズ台帳が空。src.collect --no-skip で再収集を。")
        return
    ent = ent.reset_index(drop=True)
    ent["proba"] = _walk_predict(ent)
    ent = ent[ent["proba"].notna()]

    # レース→券種→{combo:odds}
    odds_map: dict[tuple, dict[str, dict[str, float]]] = {}
    for (d, j, r, b), g in od.groupby(["date", "jcd", "rno", "bet"]):
        odds_map.setdefault((d, j, r), {})[b] = dict(zip(g["combo"], g["odds"]))

    # 集計器: pool → strategy → [staked, returned, bets, hits]
    strat = ["market", "prob", "ev1", "evK"]
    acc = {p: {s: [0, 0, 0, 0] for s in strat} for p in POOLS}
    overround = {p: [] for p in POOLS}
    races = 0

    for (d, j, r), g in ent.groupby(["date", "jcd", "rno"]):
        if g["proba"].isna().any():
            continue
        om = odds_map.get((d, j, r))
        if not om:
            continue
        order = {int(row.car): int(row.finish) for row in g.itertuples()
                 if pd.notna(row.finish)}
        real = _realized(order)
        if not real:
            continue
        wp = {int(row.car): float(row.proba) for row in g.itertuples()}
        probs = _harville(wp)
        races += 1

        for pool in POOLS:
            ladder = om.get(pool)
            pr = probs.get(pool)
            if not ladder or not pr:
                continue
            inv = sum(1 / o for o in ladder.values() if o and o > 0)
            if inv > 0:
                overround[pool].append(inv)
            hit_combo = real.get(pool)

            # market: 最低オッズ1点
            mkt = min(ladder, key=lambda c: ladder[c])
            _bet(acc[pool]["market"], mkt, ladder, hit_combo)
            # prob: モデル最確1点
            best_p = max(pr, key=lambda c: pr[c])
            _bet(acc[pool]["prob"], best_p, ladder, hit_combo)
            # EV 候補
            cand = []
            for combo, p in pr.items():
                o = ladder.get(combo)
                if not o or o <= 0 or o > max_odds or p < min_prob:
                    continue
                ev = p * o
                if ev >= min_ev:
                    cand.append((combo, ev, o, p))
            if cand:
                best_ev = max(cand, key=lambda x: x[3])  # 割安な中で最も当たりやすい
                _bet(acc[pool]["ev1"], best_ev[0], ladder, hit_combo)
                for combo, ev, o, p in cand:
                    _bet(acc[pool]["evK"], combo, ladder, hit_combo)

    print(f"=== マルチプールEVバックテスト（{races}レース / EV≥{min_ev} / oddsキャップ{max_odds:.0f}）===")
    print(f"{'pool':9s}{'控除率':>7s}  {'strategy':9s}{'bets':>7s}{'hit':>8s}{'ROI':>9s}")
    for pool in POOLS:
        ov = np.mean(overround[pool]) if overround[pool] else float("nan")
        takeout = (1 - 1 / ov) * 100 if ov and ov > 0 else float("nan")
        for i, s in enumerate(strat):
            st, rt, b, h = acc[pool][s]
            if b == 0:
                continue
            head = f"{pool:9s}{takeout:6.1f}%" if i == 0 else " " * 16
            roi = rt / st * 100
            flag = " ✓" if roi >= 100 else ""
            print(f"{head}  {s:9s}{b:7d}{h/b*100:7.1f}%{roi:8.1f}%{flag}")
    print("  ※確定オッズ前提。market=最低オッズ1点/prob=モデル最確/ev1=最大EV1点/evK=EV≥閾値全点")


def _bet(a, combo, ladder, hit_combo):
    o = ladder.get(combo)
    if not o or o <= 0:
        return
    a[2] += 1
    a[0] += 100
    if combo == hit_combo:
        a[3] += 1
        a[1] += o * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ev", type=float, default=1.0)
    ap.add_argument("--max-odds", type=float, default=300.0)
    ap.add_argument("--min-prob", type=float, default=0.0)
    args = ap.parse_args()
    backtest(args.min_ev, args.max_odds, args.min_prob)


if __name__ == "__main__":
    main()
