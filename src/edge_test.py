"""エッジ検定：3連単等の ev1 戦略が本物か検出力込みで判定する。

手続き（p-hacking を排す厳密版）:
  1) 時系列ウォークフォワードで各車の1着確率（リークなし）
  2) Harville で各プールの組合せ確率
  3) 日付で tune(前半) / test(後半) に分割
  4) 閾値(min_ev, cap) は **tune だけ** で最良を選び固定
  5) その固定パラを test に適用 → ブートストラップで ROI の信頼区間と P(ROI<100%)

market を明確に超え、かつ test の CI が 100% を跨がなければ「本物のエッジ候補」。
跨ぐなら検出力不足 or 変動として保留/棄却。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage
from .backtest import _walk_predict
from .backtest_multi import _harville, _realized

POOLS = ["exacta", "quinella", "trio", "trifecta"]
GRID_EV = [1.0, 1.1, 1.2, 1.3]
GRID_CAP = [30, 50, 100, 200]
RNG = np.random.RandomState(0)


def _races_with_probs():
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

    races = []  # (date, {pool: (realized, [(combo,p,o)...])})
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
        races.append((d, pools))
    races.sort(key=lambda x: x[0])
    return races


def _ev1_bets(subset, pool, min_ev, cap):
    """ev1（EV≥min_ev, odds≤cap の中でモデル最確1点）の払戻配列(0 or odds*100)。"""
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


def _roi(arr):
    return arr.mean() / 100 * 100 if len(arr) else float("nan")


def _boot(arr, B=20000):
    n = len(arr)
    if n == 0:
        return (float("nan"),) * 3 + (float("nan"),)
    means = np.array([arr[RNG.randint(0, n, n)].mean() for _ in range(B)]) / 100 * 100
    p5, p50, p95 = np.percentile(means, [5, 50, 95])
    return p5, p50, p95, float((means < 100).mean() * 100)


def _market_bets(subset, pool):
    """市場本命(最低オッズ1点)の払戻配列（比較基準）。"""
    out = []
    for _d, pools in subset:
        if pool not in pools:
            continue
        real, items = pools[pool]
        if not items:
            continue
        c, p, o = min(items, key=lambda x: x[2])
        out.append(o * 100 if c == real else 0.0)
    return np.array(out)


def main(split=0.65):
    races = _races_with_probs()
    dates = sorted({d for d, _ in races})
    cut = dates[int(len(dates) * split)]
    tune = [x for x in races if x[0] < cut]
    test = [x for x in races if x[0] >= cut]
    print(f"データ: {len(races)}レース / tune {len(tune)}({dates[0]}..{cut}) "
          f"/ test {len(test)}({cut}..{dates[-1]})\n")

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
    print("\n ✓=testのCI下限≥100%(本物候補) / ~=点ROI≥100%だがCIが100%跨ぐ(未決) / 無印=<100%")
    print(" ※確定オッズ前提。実弾は薄いexoticプールで自分の投票がオッズを下げる haircut が別途かかる。")


if __name__ == "__main__":
    main()
