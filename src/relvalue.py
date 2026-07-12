"""プール間相対価値エッジ検定（cross-pool relative value）。

edge_test は自作モデル(AUC0.81, 市場より弱い)の確率で exotic を測った。本モジュールは
確率源を替える：**市場の単勝オッズ**（本日 bias_map で「美しく較正」と実証＝市場で
最もシャープな確率）を真値とし、Harville で exotic の理論価格を導出。実際の exotic
オッズがそれより割高な目(EV≥閾値)を突く。

仮説(②の未検証メカニズム)：群衆は単勝より exotic を雑に買う。ならシャープな単勝プールが
exotic の歪みを暴けるはず。単勝プール自身が真値なので「モデルが弱いから負けた」を排除する。

手続き(p-hacking排除・edge_test と同型)：
  1) 各レース、単勝オッズ→控除除去の1着確率 p_i（＝市場の合議、リーク無・当日確定）
  2) Harville で各 exotic プールの組合せ確率
  3) 日付で tune(前半)/test(後半) 分割、閾値は tune だけで固定
  4) test に適用、ブートで CI と P(ROI<100%)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage
from .backtest_multi import _harville, _realized

POOLS = ["exacta", "quinella", "trio", "trifecta"]
GRID_EV = [1.0, 1.1, 1.2, 1.3, 1.5]
GRID_CAP = [30, 50, 100, 200]
RNG = np.random.RandomState(0)


def _races():
    conn = storage.connect()
    ent = pd.read_sql_query(
        "SELECT date,jcd,rno,car,finish FROM entries WHERE win IS NOT NULL", conn)
    win = pd.read_sql_query(
        "SELECT date,jcd,rno,car,odds FROM win_odds WHERE odds IS NOT NULL AND odds>0", conn)
    od = pd.read_sql_query("SELECT date,jcd,rno,bet,combo,odds FROM odds", conn)
    conn.close()
    for d in (ent, win, od):
        d["jcd"] = d["jcd"].astype(str).str.zfill(2)

    # レース→単勝オッズ {car:odds}
    winmap: dict[tuple, dict[int, float]] = {}
    for (d, j, r), g in win.groupby(["date", "jcd", "rno"]):
        winmap[(d, j, r)] = dict(zip(g["car"].astype(int), g["odds"].astype(float)))
    # レース→賭式→{combo:odds}
    ladders: dict[tuple, dict[str, dict[str, float]]] = {}
    for (d, j, r, b), g in od.groupby(["date", "jcd", "rno", "bet"]):
        ladders.setdefault((d, j, r), {})[b] = dict(zip(g["combo"], g["odds"]))

    races = []  # (date, {pool:(realized,[(combo,p,o)...])})
    for (d, j, r), g in ent.groupby(["date", "jcd", "rno"]):
        wo = winmap.get((d, j, r))
        lad = ladders.get((d, j, r))
        if not wo or not lad:
            continue
        order = {int(x.car): int(x.finish) for x in g.itertuples() if pd.notna(x.finish)}
        real = _realized(order)
        if not real:
            continue
        # 単勝オッズ→控除除去1着確率（市場の合議）
        inv = {c: 1.0 / o for c, o in wo.items() if o > 0}
        s = sum(inv.values())
        wp = {c: v / s for c, v in inv.items()} if s > 0 else {}
        if not wp:
            continue
        probs = _harville(wp)
        pools = {}
        for pool in POOLS:
            L, pr = lad.get(pool), probs.get(pool)
            if not L or not pr:
                continue
            items = [(c, p, L[c]) for c, p in pr.items() if c in L and L[c] > 0]
            if items:
                pools[pool] = (real.get(pool), items)
        if pools:
            races.append((d, pools))
    races.sort(key=lambda x: x[0])
    return races


def _ev1(subset, pool, min_ev, cap):
    """EV≥min_ev, odds≤cap の中で最確1点の払戻配列(0 or odds*100)。"""
    out = []
    for _d, pools in subset:
        if pool not in pools:
            continue
        real, items = pools[pool]
        cand = [(c, p, o) for c, p, o in items if p * o >= min_ev and o <= cap]
        if not cand:
            continue
        c, p, o = max(cand, key=lambda x: x[1])
        out.append(o * 100 if c == real else 0.0)
    return np.array(out)


def _market(subset, pool):
    out = []
    for _d, pools in subset:
        if pool not in pools:
            continue
        real, items = pools[pool]
        c, p, o = min(items, key=lambda x: x[2])
        out.append(o * 100 if c == real else 0.0)
    return np.array(out)


def _roi(a):
    return a.mean() / 100 * 100 if len(a) else float("nan")


def _boot(a, B=20000):
    n = len(a)
    if n == 0:
        return (float("nan"),) * 4
    means = np.array([a[RNG.randint(0, n, n)].mean() for _ in range(B)]) / 100 * 100
    p5, p50, p95 = np.percentile(means, [5, 50, 95])
    return p5, p95, float((means < 100).mean() * 100), p50


def main(split=0.65):
    races = _races()
    dates = sorted({d for d, _ in races})
    cut = dates[int(len(dates) * split)]
    tune = [x for x in races if x[0] < cut]
    test = [x for x in races if x[0] >= cut]
    print(f"確率源=市場単勝オッズ(控除除去)。データ {len(races):,}R / "
          f"tune {len(tune):,}({dates[0]}..{cut}) / test {len(test):,}({cut}..{dates[-1]})\n")
    print(f"{'pool':9s}{'固定パラ(tune最良)':>18s}{'tuneROI':>9s}"
          f"{'test:bets':>11s}{'hit':>7s}{'ROI':>7s}{'CI(5-95%)':>15s}{'P<100%':>8s}{'市場ROI':>8s}")
    for pool in POOLS:
        best = None
        for me in GRID_EV:
            for cap in GRID_CAP:
                a = _ev1(tune, pool, me, cap)
                if len(a) >= 100:
                    roi = _roi(a)
                    if best is None or roi > best[0]:
                        best = (roi, me, cap)
        if best is None:
            continue
        troi, me, cap = best
        t = _ev1(test, pool, me, cap)
        mk = _market(test, pool)
        p5, p95, plt, _ = _boot(t)
        n, h = len(t), int((t > 0).sum())
        flag = " ✓" if p5 >= 100 else (" ~" if _roi(t) >= 100 else "")
        print(f"{pool:9s}{f'ev≥{me},cap{cap}':>18s}{troi:8.0f}%"
              f"{n:11d}{h/n*100 if n else 0:6.1f}%{_roi(t):6.0f}%"
              f"{f'[{p5:.0f},{p95:.0f}]':>15s}{plt:7.1f}%{_roi(mk):7.0f}%{flag}")
    print("\n ✓=CI下限≥100%(本物候補) / ~=点ROI≥100%だがCI跨ぐ / 無印=<100%")


if __name__ == "__main__":
    main()
