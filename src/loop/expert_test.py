"""専門家(予想屋)の暗黙知が config+オッズの壁を越えさせるか。

expert_picks を結合し、config+全オッズ に「専門家特徴」を足して:
  (1) 本命飛び(3着外) AUC の純増分(ペアブートで有意か)
  (2) 本命除外3連複ボックスROIが高荒れ帯で控除の壁(≈82%)を越えるか
を測る。専門家特徴:
  winrate, n_picks, fav_in_picks(本命が推奨に居るか), fav_is_head(本命が専門家の1着か),
  fav_excluded(専門家が本命を切ったか=フェード信号), comment_len

  python -m src.loop.expert_test
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

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
    exp = pd.read_sql_query("SELECT * FROM expert_picks", conn)
    od = pd.read_sql_query(
        "SELECT date,jcd,rno,combo,odds FROM odds WHERE bet='trio' AND date BETWEEN ? AND ?",
        conn, params=E.BLOCK_A)
    conn.close()
    for dd in (fin, exp, od):
        dd["jcd"] = dd["jcd"].astype(str).str.zfill(2)
    finmap = {(d, j, r): dict(zip(g["car"].astype(int), g["finish"].astype(int)))
              for (d, j, r), g in fin.groupby(["date", "jcd", "rno"])}
    trmap = {(d, j, r): dict(zip(g["combo"], g["odds"]))
             for (d, j, r), g in od.groupby(["date", "jcd", "rno"])}
    expmap = {(row.date, row.jcd, int(row.rno)): row for row in exp.itertuples()}

    data = []
    for r in races:
        if len(r.win_odds) < 4:
            continue
        e = expmap.get(r.key)
        if e is None:
            continue
        fm = finmap.get(r.key, {})
        ranked = sorted(r.win_odds.items(), key=lambda kv: kv[1])
        fav = ranked[0][0]
        if fm.get(fav) is None:
            continue
        s = sum(1.0 / o for o in r.win_odds.values() if o > 0)
        imps = sorted([(1.0 / o) / s for o in r.win_odds.values() if o > 0], reverse=True)
        imps = (imps + [0.0] * 8)[:8]
        cars = set(int(x) for x in (e.cars_csv or "").split(",") if x)
        exp_feats = [
            float(e.winrate or 0), float(e.n_picks or 0),
            1.0 if fav in cars else 0.0,                       # 本命が推奨に居る
            1.0 if e.head_car == fav else 0.0,                 # 本命が専門家の1着
            1.0 if (cars and fav not in cars) else 0.0,        # 専門家が本命を切った
            float(e.comment_len or 0),
        ]
        pop = [c for c, o in ranked]
        top3 = sorted(fm, key=lambda c: fm[c])[:3]
        box = tuple(sorted(pop[1:4]))
        trio = trmap.get(r.key, {})
        trio_ret = (trio.get("-".join(map(str, box)), 0) * 100
                    if set(top3) == set(box) else 0)
        data.append(dict(ym=r.ym, base=list(r.emb_std) + imps, exp=exp_feats,
                         blow=int(fm[fav] > 3), trio_ret=trio_ret))
    return data


def _wf(data, mode):
    yms = sorted({d["ym"] for d in data})
    by = {m: [d for d in data if d["ym"] == m] for m in yms}
    Xh, yh = [], []
    preds, acts, rets = [], [], []

    def feat(d):
        return d["base"] + d["exp"] if mode == "with" else d["base"]

    for mi, m in enumerate(yms):
        if mi > 0 and len(yh) >= 400 and len(set(yh)) > 1:
            clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                                 learning_rate=0.05, random_state=0)
            clf.fit(np.array(Xh), np.array(yh))
            p = clf.predict_proba(np.array([feat(d) for d in by[m]]))[:, 1]
            preds.extend(p.tolist())
            acts.extend([d["blow"] for d in by[m]])
            rets.extend([d["trio_ret"] for d in by[m]])
        for d in by[m]:
            Xh.append(feat(d))
            yh.append(d["blow"])
    return np.array(preds), np.array(acts), np.array(rets)


def main():
    data = _load()
    if len(data) < 500:
        print(f"expert_picks 結合レースが少ない({len(data)})。収集を進めてから。")
        return
    base = np.mean([d["blow"] for d in data]) * 100
    print(f"expert結合レース {len(data):,}  本命飛びベース率 {base:.1f}%\n")

    pB, aB, _ = _wf(data, "base")
    pW, aW, rW = _wf(data, "with")
    aucB, aucW = roc_auc_score(aB, pB), roc_auc_score(aW, pW)
    n = len(aB)
    diffs = np.array([roc_auc_score(aW[i], pW[i]) - roc_auc_score(aB[i], pB[i])
                      for i in (RNG.randint(0, n, n) for _ in range(2000))])
    print("=== (1) 本命飛びAUC: 専門家特徴の純増分 ===")
    print(f"  config+オッズ        : AUC={aucB:.3f}")
    print(f"  +専門家(暗黙知)      : AUC={aucW:.3f}")
    print(f"  純増分 {aucW-aucB:+.4f}  ペアブートCI[{np.percentile(diffs,5):+.4f},"
          f"{np.percentile(diffs,95):+.4f}]  P(≤0)={(diffs<=0).mean()*100:.1f}%")

    print("\n=== (2) 本命除外3連複ボックス ROI: 荒れ確率(専門家込モデル)デシル別 ===")
    print(f"{'荒れ帯':>12s}{'n':>6s}{'ROI':>7s}{'ブートCI':>14s}")
    q = np.quantile(pW, np.linspace(0, 1, 11))
    for i in range(10):
        lo, hi = q[i], q[i + 1]
        m = (pW >= lo) & (pW <= hi) if i == 9 else (pW >= lo) & (pW < hi)
        if m.sum() < 80:
            continue
        ret = rW[m]
        roi = ret.mean()
        nn = len(ret)
        boot = np.array([ret[RNG.randint(0, nn, nn)].mean() for _ in range(4000)])
        p5, p95 = np.percentile(boot, [5, 95])
        f = "✓" if p5 >= 100 else ""
        print(f"{f'D{i+1}':>12s}{nn:6d}{roi:6.0f}%{f'[{p5:.0f},{p95:.0f}]{f}':>14s}")
    print("\n ✓=CI下限≥100%。純増分が有意&高荒れデシルで✓なら暗黙知が壁を越えた。")


if __name__ == "__main__":
    main()
