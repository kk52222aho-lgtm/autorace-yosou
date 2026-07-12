"""B2の荒れ検出を金に変えられるか：本命が飛ぶと読んだレースで本命をフェードして
2番人気/3番人気を張り、ROIが控除を超えるかをWF・ブロックブートで測る。

検出器=config(45次元)+全オッズ の GBM で「本命(最低オッズ)が3着外に飛ぶ確率」。
過去月だけで学習(リーク無)→当月に適用。荒れ確率デシル別に、
  fav1 : 本命単勝(参考・市場)
  fav2 : 2番人気単勝(本命フェードの受け皿)
  fav3 : 3番人気単勝
のROIを出す。高荒れデシルで fav2/fav3 が控除超え(≥100%)なら金になる。
※払戻は確定オッズ(リーク)=上限値。ここで超えなければ締切前では更に届かない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import env as E
from .. import storage

RNG = np.random.RandomState(0)


def _load():
    races = E.load_block_a()
    mu, sd = E.standardizer(races)
    for r in races:
        r.emb_std = (r.emb - mu) / sd
    conn = storage.connect()
    fin = pd.read_sql_query(
        "SELECT date,jcd,rno,car,finish FROM entries WHERE finish IS NOT NULL AND date BETWEEN ? AND ?",
        conn, params=E.BLOCK_A)
    conn.close()
    fin["jcd"] = fin["jcd"].astype(str).str.zfill(2)
    finmap = {(d, j, r): dict(zip(g["car"].astype(int), g["finish"].astype(int)))
              for (d, j, r), g in fin.groupby(["date", "jcd", "rno"])}
    data = []
    for r in races:
        if len(r.win_odds) < 3:
            continue
        fm = finmap.get(r.key, {})
        # オッズ昇順=人気順の車と確定オッズ
        ranked = sorted(r.win_odds.items(), key=lambda kv: kv[1])   # [(car,odds)...]
        fav = ranked[0][0]
        ff = fm.get(fav)
        if ff is None:
            continue
        s = sum(1.0 / o for o in r.win_odds.values() if o > 0)
        imps = sorted([(1.0 / o) / s for o in r.win_odds.values() if o > 0], reverse=True)
        imps = (imps + [0.0] * 8)[:8]
        win_by_pop = []      # 人気k番の (odds, 勝ったか)
        for car, o in ranked[:3]:
            win_by_pop.append((o, int(fm.get(car) == 1)))
        data.append(dict(ym=r.ym, x=list(r.emb_std) + imps, blow=int(ff > 3),
                         pop=win_by_pop))
    return data


def _wf_scores(data):
    """WFで各レースの荒れ確率を付与。"""
    yms = sorted({d["ym"] for d in data})
    by = {m: [d for d in data if d["ym"] == m] for m in yms}
    Xh, yh = [], []
    out = []
    for mi, m in enumerate(yms):
        if mi > 0 and len(yh) >= 500 and len(set(yh)) > 1:
            clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                                 learning_rate=0.05, random_state=0)
            clf.fit(np.array(Xh), np.array(yh))
            p = clf.predict_proba(np.array([d["x"] for d in by[m]]))[:, 1]
            for d, pi in zip(by[m], p):
                out.append((pi, d["pop"]))
        for d in by[m]:
            Xh.append(d["x"])
            yh.append(d["blow"])
    return out


def _roi_boot(bets, B=10000):
    """bets: [(odds, hit)]。フラットROIとブロックブートCI。"""
    if not bets:
        return float("nan"), float("nan"), float("nan")
    pay = np.array([o * h for o, h in bets])
    roi = pay.mean() * 100
    n = len(pay)
    boot = np.array([pay[RNG.randint(0, n, n)].mean() for _ in range(B)]) * 100
    return roi, np.percentile(boot, 5), np.percentile(boot, 95)


def main():
    data = _load()
    scored = _wf_scores(data)
    probs = np.array([s[0] for s in scored])
    print(f"WF採点レース {len(scored):,}  荒れ確率デシル別に人気1/2/3番の単勝ROI(確定オッズ=上限)\n")
    print(f"{'荒れ確率帯':>14s}{'n':>6s}"
          f"{'fav1 ROI':>10s}{'fav2 ROI':>10s}{'fav2 CI':>14s}{'fav3 ROI':>10s}{'fav3 CI':>14s}")
    # デシル(荒れ確率で10分割)
    q = np.quantile(probs, np.linspace(0, 1, 11))
    for i in range(10):
        lo, hi = q[i], q[i + 1]
        m = (probs >= lo) & (probs <= hi) if i == 9 else (probs >= lo) & (probs < hi)
        sub = [scored[j] for j in np.where(m)[0]]
        if len(sub) < 100:
            continue
        b1 = [pop[0] for _, pop in sub]
        b2 = [pop[1] for _, pop in sub]
        b3 = [pop[2] for _, pop in sub]
        r1, _, _ = _roi_boot(b1)
        r2, p2l, p2h = _roi_boot(b2)
        r3, p3l, p3h = _roi_boot(b3)
        f2 = "✓" if p2l >= 100 else ""
        f3 = "✓" if p3l >= 100 else ""
        print(f"{f'D{i+1}[{lo:.2f},{hi:.2f}]':>14s}{len(sub):6d}"
              f"{r1:9.0f}%{r2:9.0f}%{f'[{p2l:.0f},{p2h:.0f}]{f2}':>14s}"
              f"{r3:9.0f}%{f'[{p3l:.0f},{p3h:.0f}]{f3}':>14s}")
    print("\n ✓=ブロックブートCI下限≥100%。高荒れデシルでfav2/fav3が✓なら本命フェードが金になる。")


if __name__ == "__main__":
    main()
