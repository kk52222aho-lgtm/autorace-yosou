"""最後のカード：予想屋の地の文(コメント本文)＝暗黙知が config+オッズの壁を越えるか。

核心仮説：プロースは「専門家が本命をどう見てるか」を漏らす。本命騎手を文頭で名指す=確信、
後回し/無名=疑念(市場オッズに乗らない飛び予兆)。本文から:
  fav_mention_rank : 本命騎手がコメント内で何番目に登場するか(1=専門家の先頭=確信)
  fav_not_named    : 本命が本文に一度も出ない(専門家が眼中に置いてない=強い疑念)
  n_named          : 名指しされた騎手数(多い=開いた混戦)
  rough_kw/conf_kw : 荒れ語彙/確信語彙の数
を config+全オッズ に足し、(1)本命飛びAUC純増分(ペアブート) (2)本命除外3連複ボックスの
高荒れデシルROIが控除の壁を越えるか、を測る。

  python -m src.loop.expert_prose
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import env as E
from .. import storage

RNG = np.random.RandomState(0)
ROUGH = ["荒", "波乱", "混戦", "難解", "難し", "一発", "展開次第", "展開向", "ヒモ",
         "注意", "逆転", "まくり", "恵まれ", "たら", "でも", "警戒"]
CONF = ["本命", "中心", "断然", "主役", "軸", "逃げ切", "抜け出", "安定", "格上", "実績上位"]


def _mention_index(name: str, comment: str) -> int:
    """騎手名(の長い接頭辞)がコメントに最初に出る位置。無ければ大きい値。"""
    if not name or not comment:
        return 9999
    for L in (3, 2):
        if len(name) >= L:
            idx = comment.find(name[:L])
            if idx >= 0:
                return idx
    return 9999


def _load():
    races = E.load_block_a()
    mu, sd = E.standardizer(races)
    for r in races:
        r.emb_std = (r.emb - mu) / sd
    conn = storage.connect()
    ent = pd.read_sql_query(
        "SELECT date,jcd,rno,car,finish,name FROM entries WHERE finish IS NOT NULL AND date BETWEEN ? AND ?",
        conn, params=E.BLOCK_A)
    exp = pd.read_sql_query(
        "SELECT date,jcd,rno,comment FROM expert_picks WHERE comment IS NOT NULL AND length(comment)>0", conn)
    od = pd.read_sql_query(
        "SELECT date,jcd,rno,combo,odds FROM odds WHERE bet='trio' AND date BETWEEN ? AND ?",
        conn, params=E.BLOCK_A)
    conn.close()
    for dd in (ent, exp, od):
        dd["jcd"] = dd["jcd"].astype(str).str.zfill(2)
    finmap, namemap = {}, {}
    for (d, j, r), g in ent.groupby(["date", "jcd", "rno"]):
        finmap[(d, j, r)] = dict(zip(g["car"].astype(int), g["finish"].astype(int)))
        namemap[(d, j, r)] = dict(zip(g["car"].astype(int), g["name"]))
    trmap = {(d, j, r): dict(zip(g["combo"], g["odds"]))
             for (d, j, r), g in od.groupby(["date", "jcd", "rno"])}
    cmap = {(row.date, row.jcd, int(row.rno)): row.comment for row in exp.itertuples()}

    data = []
    for r in races:
        cm = cmap.get(r.key)
        if not cm or len(r.win_odds) < 4:
            continue
        fm = finmap.get(r.key, {})
        nm = namemap.get(r.key, {})
        ranked = sorted(r.win_odds.items(), key=lambda kv: kv[1])
        fav = ranked[0][0]
        if fm.get(fav) is None:
            continue
        # プロース特徴
        idxs = {c: _mention_index(nm.get(c, ""), cm) for c in r.win_odds}
        named = sorted([c for c in idxs if idxs[c] < 9999], key=lambda c: idxs[c])
        fav_rank = (named.index(fav) + 1) if fav in named else (len(r.win_odds) + 1)
        prose = [
            float(fav_rank),
            1.0 if fav not in named else 0.0,
            float(len(named)),
            float(sum(cm.count(k) for k in ROUGH)),
            float(sum(cm.count(k) for k in CONF)),
            float(len(cm)),
        ]
        s = sum(1.0 / o for o in r.win_odds.values() if o > 0)
        imps = sorted([(1.0 / o) / s for o in r.win_odds.values() if o > 0], reverse=True)
        imps = (imps + [0.0] * 8)[:8]
        pop = [c for c, o in ranked]
        top3 = sorted(fm, key=lambda c: fm[c])[:3]
        box = tuple(sorted(pop[1:4]))
        trio = trmap.get(r.key, {})
        trio_ret = (trio.get("-".join(map(str, box)), 0) * 100 if set(top3) == set(box) else 0)
        data.append(dict(ym=r.ym, base=list(r.emb_std) + imps, prose=prose,
                         blow=int(fm[fav] > 3), trio_ret=trio_ret))
    return data


def _wf(data, mode):
    yms = sorted({d["ym"] for d in data})
    by = {m: [d for d in data if d["ym"] == m] for m in yms}
    Xh, yh, preds, acts, rets = [], [], [], [], []

    def feat(d):
        return d["base"] + d["prose"] if mode == "with" else d["base"]

    for mi, m in enumerate(yms):
        if mi > 0 and len(yh) >= 400 and len(set(yh)) > 1:
            clf = HistGradientBoostingClassifier(max_iter=250, max_depth=4,
                                                 learning_rate=0.05, random_state=0)
            clf.fit(np.array(Xh), np.array(yh))
            p = clf.predict_proba(np.array([feat(d) for d in by[m]]))[:, 1]
            preds.extend(p.tolist()); acts.extend([d["blow"] for d in by[m]])
            rets.extend([d["trio_ret"] for d in by[m]])
        for d in by[m]:
            Xh.append(feat(d)); yh.append(d["blow"])
    return np.array(preds), np.array(acts), np.array(rets)


def main():
    data = _load()
    print(f"地の文つきレース {len(data):,}  本命飛びベース {np.mean([d['blow'] for d in data])*100:.1f}%\n")
    if len(data) < 800:
        print("地の文の量が足りない。expert_collect の本文収集を進めてから。")
        return
    pB, aB, _ = _wf(data, "base")
    pW, aW, rW = _wf(data, "with")
    aucB, aucW = roc_auc_score(aB, pB), roc_auc_score(aW, pW)
    n = len(aB)
    diffs = np.array([roc_auc_score(aW[i], pW[i]) - roc_auc_score(aB[i], pB[i])
                      for i in (RNG.randint(0, n, n) for _ in range(2000))])
    print("=== (1) 本命飛びAUC: 地の文の純増分 ===")
    print(f"  config+全オッズ      : AUC={aucB:.3f}")
    print(f"  +地の文(暗黙知)      : AUC={aucW:.3f}")
    print(f"  純増分 {aucW-aucB:+.4f}  CI[{np.percentile(diffs,5):+.4f},{np.percentile(diffs,95):+.4f}]"
          f"  P(≤0)={(diffs<=0).mean()*100:.1f}%")

    print("\n=== (2) 本命除外3連複ボックス ROI: 荒れ確率(地の文込)デシル別 ===")
    print(f"{'荒れ帯':>8s}{'n':>6s}{'ROI':>7s}{'ブートCI':>14s}")
    q = np.quantile(pW, np.linspace(0, 1, 11))
    for i in range(10):
        lo, hi = q[i], q[i + 1]
        m = (pW >= lo) & (pW <= hi) if i == 9 else (pW >= lo) & (pW < hi)
        if m.sum() < 60:
            continue
        ret = rW[m]; nn = len(ret)
        boot = np.array([ret[RNG.randint(0, nn, nn)].mean() for _ in range(4000)])
        p5, p95 = np.percentile(boot, [5, 95])
        f = "✓" if p5 >= 100 else ""
        print(f"{f'D{i+1}':>8s}{nn:6d}{ret.mean():6.0f}%{f'[{p5:.0f},{p95:.0f}]{f}':>14s}")
    print("\n ✓=CI下限≥100%。純増分が有意&高荒れデシルで✓なら地の文の暗黙知が壁を越えた。")


if __name__ == "__main__":
    main()
