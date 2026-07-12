"""ワイド相対価値エッジの厳密検定（オフライン、wide_odds テーブルを使用）。

規律の要:
  - 確率源=市場単勝オッズ控除除去(最もシャープ) → Harville trio → ワイド{i,j}確率
  - realized=entries.finish の上位3着の全ペア（当り目再取得不要）
  - 払戻=min_odds（保守=床。実払戻はこれ以上）
  - **レース単位ブロック・ブートストラップ**：evKは1レース複数点で相関するため、
    ベット単位ではなくレースを再標本化して CI を出す（分散の過小評価を防ぐ）
  - tune/test 日付分割 ＋ 期間を伸ばした減衰方向（蜃気楼判定）
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import storage
from .backtest_multi import _harville

RNG = np.random.RandomState(0)


def _wide_probs(wp):
    trio = _harville(wp).get("trio", {})
    cars = sorted(wp)
    out = {}
    for i, j in combinations(cars, 2):
        out[f"{i}-{j}"] = sum(
            trio.get("-".join(map(str, sorted([i, j, k]))), 0.0)
            for k in cars if k not in (i, j))
    return out


def _load_races():
    conn = storage.connect()
    ent = pd.read_sql_query(
        "SELECT date,jcd,rno,car,finish FROM entries WHERE finish IS NOT NULL", conn)
    win = pd.read_sql_query(
        "SELECT date,jcd,rno,car,odds FROM win_odds WHERE odds>0", conn)
    wide = pd.read_sql_query(
        "SELECT date,jcd,rno,combo,min_odds FROM wide_odds WHERE min_odds>0", conn)
    conn.close()
    for d in (ent, win, wide):
        d["jcd"] = d["jcd"].astype(str).str.zfill(2)

    winmap, finmap = {}, {}
    for (d, j, r), g in ent.groupby(["date", "jcd", "rno"]):
        finmap[(d, j, r)] = {int(c): int(f) for c, f in zip(g["car"], g["finish"])}
    for (d, j, r), g in win.groupby(["date", "jcd", "rno"]):
        winmap[(d, j, r)] = {int(c): float(o) for c, o in zip(g["car"], g["odds"])}

    races = []
    for (d, j, r), g in wide.groupby(["date", "jcd", "rno"]):
        wo = winmap.get((d, j, r))
        fin = finmap.get((d, j, r))
        if not wo or not fin:
            continue
        top3 = sorted(fin, key=lambda c: fin[c])[:3]
        if len(top3) < 3:
            continue
        winners = {"-".join(map(str, sorted(p))) for p in combinations(top3, 2)}
        inv = {c: 1 / o for c, o in wo.items() if o > 0}
        s = sum(inv.values())
        wp = {c: v / s for c, v in inv.items()}
        wpr = _wide_probs(wp)
        ladder = {c: float(o) for c, o in zip(g["combo"], g["min_odds"])}
        items = [(c, wpr.get(c, 0.0), o) for c, o in ladder.items() if o > 0]
        races.append((d, winners, items))
    races.sort(key=lambda x: x[0])
    return races


def _race_payoffs(races, strat, min_ev=1.0, cap=1e9):
    """各レースの (staked, returned) を返す（レース単位集計＝ブロックブート用）。"""
    rows = []
    for _d, winners, items in races:
        bets = []
        if strat == "market":
            c, p, o = min(items, key=lambda x: x[2])
            bets = [(c, o)]
        else:
            cand = [(c, p, o) for c, p, o in items if p * o >= min_ev and o <= cap]
            if strat == "ev1" and cand:
                c, p, o = max(cand, key=lambda x: x[1])
                bets = [(c, o)]
            elif strat == "evK":
                bets = [(c, o) for c, p, o in cand]
        if not bets:
            continue
        st = 100 * len(bets)
        rt = sum(o * 100 for c, o in bets if c in winners)
        rows.append((st, rt, len(bets), sum(1 for c, o in bets if c in winners)))
    return rows


def _summ(rows, B=20000):
    """レース単位ブロックブート。ROI点推定・CI(5-95%)・P(<100%)・bets・hits。"""
    if not rows:
        return None
    st = np.array([r[0] for r in rows], float)
    rt = np.array([r[1] for r in rows], float)
    nb = sum(r[2] for r in rows)
    nh = sum(r[3] for r in rows)
    roi = rt.sum() / st.sum() * 100
    n = len(rows)
    boot = np.empty(B)
    for b in range(B):
        idx = RNG.randint(0, n, n)
        boot[b] = rt[idx].sum() / st[idx].sum() * 100
    p5, p95 = np.percentile(boot, [5, 95])
    return dict(roi=roi, p5=p5, p95=p95, plt=float((boot < 100).mean() * 100),
                bets=nb, hits=nh, races=n)


def _line(name, s):
    if not s:
        print(f"{name:12s}  (対象なし)")
        return
    flag = " ✓" if s["p5"] >= 100 else (" ~" if s["roi"] >= 100 else "")
    ci = f"[{s['p5']:.0f},{s['p95']:.0f}]"
    hit = s["hits"] / s["bets"] * 100 if s["bets"] else 0.0
    print(f"{name:12s}{s['races']:7d}{s['bets']:8d}{hit:7.1f}%"
          f"{s['roi']:8.1f}%{ci:>15s}{s['plt']:8.1f}%{flag}")


def main():
    races = _load_races()
    if not races:
        print("wide_odds が空。先に python -m src.wide_collect を。")
        return
    dates = sorted({d for d, _, _ in races})
    print(f"ワイド検定: {len(races):,}レース / {dates[0]}..{dates[-1]}（払戻=min_odds床）\n")

    # 全期間（レース単位ブロックブート）
    print("=== 全期間（確定オッズ床／レース単位ブロックブート）===")
    print(f"{'strategy':12s}{'races':>7s}{'bets':>8s}{'hit':>8s}{'ROI床':>9s}{'CI(5-95%)':>15s}{'P<100%':>8s}")
    for name, kw in [("market", {}), ("relval_ev1", dict(min_ev=1.0)),
                     ("relval_evK", dict(min_ev=1.0)),
                     ("evK_ev1.3", dict(min_ev=1.3)),
                     ("evK_cap20", dict(min_ev=1.0, cap=20))]:
        strat = "market" if name == "market" else ("ev1" if "ev1" == name.split("_")[-1] else "evK")
        rows = _race_payoffs(races, strat, **kw)
        _line(name, _summ(rows))

    # tune/test 分割（evKショットガンが OOS で生き残るか）
    print("\n=== tune(前半0.6)/test(後半0.4) evK ===")
    cut = dates[int(len(dates) * 0.6)]
    tune = [x for x in races if x[0] < cut]
    test = [x for x in races if x[0] >= cut]
    for label, sub in [("tune", tune), ("test", test)]:
        _line(f"evK/{label}", _summ(_race_payoffs(sub, "evK", min_ev=1.0)))

    # 減衰方向：期間を伸ばした evK の点ROI
    print("\n=== 減衰方向（直近から期間を伸ばす）===")
    print(f"{'window':>10s}{'races':>7s}{'ROI床':>9s}{'CI(5-95%)':>15s}{'P<100%':>8s}")
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = max(1, int(len(dates) * frac))
        keep = set(dates[-k:])
        sub = [x for x in races if x[0] in keep]
        s = _summ(_race_payoffs(sub, "evK", min_ev=1.0))
        if s:
            ci = f"[{s['p5']:.0f},{s['p95']:.0f}]"
            print(f"{f'直近{int(frac*100)}%':>10s}{s['races']:7d}{s['roi']:8.1f}%"
                  f"{ci:>15s}{s['plt']:7.1f}%")

    print("\n ✓=CI下限≥100% / ~=点ROI≥100%だがCI跨ぐ / 無印=<100%。減衰=控除壁へ落ちれば蜃気楼。")


if __name__ == "__main__":
    main()
