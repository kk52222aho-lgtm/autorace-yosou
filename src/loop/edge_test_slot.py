"""edge_test のミッドナイト断面版：モデルEV選択買いが midnight 系でだけ生きるか。

手続きは edge_test と同一（walk-forward確率→Harville→tune/test分割→tune固定パラ
→test CI）。違いは races に開催区分を持たせ、midnight系(midnight+over_midnight)
と day(統制群) を別々に検定すること。学習はリークなしの全レース walk-forward。

  python -m src.loop.edge_test_slot
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import storage
from ..backtest import _walk_predict
from ..backtest_multi import _harville, _realized
from ..edge_test import (GRID_CAP, GRID_EV, POOLS, RNG, _boot, _ev1_bets,
                         _market_bets, _roi)

NIGHT = {"midnight", "over_midnight"}


def _races_with_probs_slot():
    cmap = pd.read_csv("data/cup_map.csv", dtype={"date": str, "jcd": str})
    slot_of = {(r.date, r.jcd): r.slot for r in cmap.itertuples()}

    conn = storage.connect()
    ent = pd.read_sql_query("SELECT * FROM entries WHERE win IS NOT NULL", conn)
    od = pd.read_sql_query("SELECT date,jcd,rno,bet,combo,odds FROM odds", conn)
    conn.close()
    ent["jcd"] = ent["jcd"].astype(str).str.zfill(2)
    od["jcd"] = od["jcd"].astype(str).str.zfill(2)
    ent = ent.reset_index(drop=True)
    ent["proba"] = _walk_predict(ent)
    ent = ent[ent["proba"].notna()]

    ladders: dict[tuple, dict[str, dict[str, float]]] = {}
    for (d, j, r, b), g in od.groupby(["date", "jcd", "rno", "bet"]):
        ladders.setdefault((d, j, r), {})[b] = dict(zip(g["combo"], g["odds"]))

    races = []  # (date, slot, {pool: (realized, [(combo,p,o)...])})
    for (d, j, r), g in ent.groupby(["date", "jcd", "rno"]):
        lad = ladders.get((d, j, r))
        if not lad:
            continue
        order = {int(x.car): int(x.finish) for x in g.itertuples() if pd.notna(x.finish)}
        real = _realized(order)
        if not real:
            continue
        wp = {int(x.car): float(x.proba) for x in g.itertuples()}
        probs = _harville(wp)
        pools = {}
        for pool in POOLS:
            L, pr = lad.get(pool), probs.get(pool)
            if not L or not pr:
                continue
            items = [(c, p, L[c]) for c, p in pr.items() if c in L and L[c] > 0]
            pools[pool] = (real.get(pool), items)
        races.append((d, slot_of.get((d, j), "unknown"), pools))
    races.sort(key=lambda x: x[0])
    return races


def _run_group(label, races, split=0.65):
    dates = sorted({d for d, _s, _p in races})
    if len(dates) < 40:
        print(f"[{label}] レース日不足 ({len(dates)}日) — スキップ")
        return
    cut = dates[int(len(dates) * split)]
    tune = [(d, p) for d, _s, p in races if d < cut]
    test = [(d, p) for d, _s, p in races if d >= cut]
    print(f"\n[{label}] {len(races)}レース / tune {len(tune)}({dates[0]}..{cut}) "
          f"/ test {len(test)}({cut}..{dates[-1]})")
    print(f"{'pool':9s}{'固定パラ(tune最良)':>18s}{'tuneROI':>9s}"
          f"{'|  test:bets':>13s}{'hit':>7s}{'ROI':>8s}{'CI(5-95%)':>16s}{'P<100%':>8s}{'  市場ROI':>9s}")
    for pool in POOLS:
        best = None
        for me in GRID_EV:
            for cap in GRID_CAP:
                arr = _ev1_bets(tune, pool, me, cap)
                if len(arr) >= 100:
                    roi = _roi(arr)
                    if best is None or roi > best[0]:
                        best = (roi, me, cap)
        if best is None:
            continue
        tune_roi, me, cap = best
        t = _ev1_bets(test, pool, me, cap)
        mk = _market_bets(test, pool)
        p5, p50, p95, plt = _boot(t)
        n, h = len(t), int((t > 0).sum())
        flag = " ✓" if p5 >= 100 else (" ~" if _roi(t) >= 100 else "")
        print(f"{pool:9s}{f'ev≥{me},cap{cap}':>18s}{tune_roi:8.0f}%"
              f"{n:13d}{h/n*100 if n else 0:6.1f}%{_roi(t):7.0f}%"
              f"{f'[{p5:.0f},{p95:.0f}]':>16s}{plt:7.1f}%{_roi(mk):8.0f}%{flag}")


def main():
    races = _races_with_probs_slot()
    _run_group("midnight系", [x for x in races if x[1] in NIGHT])
    _run_group("day(統制群)", [x for x in races if x[1] == "day"])
    print("\n ✓=testのCI下限≥100% / ~=点ROI≥100%だがCI跨ぐ / 無印=<100%")
    print(" ※確定オッズ前提＋薄い深夜プールは自分の票で潰れる haircut が昼より大きい点に注意。")


if __name__ == "__main__":
    main()
