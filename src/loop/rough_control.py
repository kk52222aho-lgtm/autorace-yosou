"""タスクB2の決定的統制：本命飛び予兆は「市場が織り込めてない布置情報」か、
それとも「本命オッズが既に語る弱さ」を布置がなぞってるだけか。

AUC 0.607 は、弱い本命(高オッズ)ほど飛ぶ→オッズだけで説明できる可能性がある。
市場が価格に入れてるならエッジでない。統制:
  A) オッズのみ(本命含意勝率1特徴)で飛び予測 → AUC_odds
  B) 布置45次元のみ → AUC_config(=0.607)
  C) 布置+オッズ → AUC_both
  AUC_both ≈ AUC_odds なら布置の増分ゼロ=市場が既に織り込み済み=エッジなし。
  さらに: 本命オッズ帯を固定した中で、布置高確信の飛び率がその帯のベースを超えるか
  (市場強度を揃えた上での純増分)。超えれば市場未織り込みの痕跡=C投資の根拠。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
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
    conn.close()
    fin["jcd"] = fin["jcd"].astype(str).str.zfill(2)
    finmap = {(d, j, r): dict(zip(g["car"].astype(int), g["finish"].astype(int)))
              for (d, j, r), g in fin.groupby(["date", "jcd", "rno"])}
    rows = []
    for r in races:
        if not r.win_odds:
            continue
        fav = min(r.win_odds, key=lambda c: r.win_odds[c])
        ff = finmap.get(r.key, {}).get(fav)
        if ff is None:
            continue
        s = sum(1.0 / o for o in r.win_odds.values() if o > 0)
        fav_imp = (1.0 / r.win_odds[fav]) / s      # 本命の市場含意勝率
        rows.append((r.ym, r.emb_std, int(ff > 3), fav_imp, r.win_odds[fav]))
    return rows


def _wf_auc(rows, mode):
    """WFで飛び予測。mode: 'odds'/'config'/'both'。(auc, preds, acts, favimp) を返す。"""
    yms = sorted({x[0] for x in rows})
    by = {m: [x for x in rows if x[0] == m] for m in yms}
    preds, acts, favimp = [], [], []
    Xh, yh = [], []

    def feat(x):
        if mode == "odds":
            return [x[3]]
        if mode == "config":
            return list(x[1])
        return list(x[1]) + [x[3]]

    for mi, m in enumerate(yms):
        if mi > 0 and len(yh) >= 500 and len(set(yh)) > 1:
            clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                                 learning_rate=0.05, random_state=0)
            clf.fit(np.array(Xh), np.array(yh))
            Xm = np.array([feat(x) for x in by[m]])
            p = clf.predict_proba(Xm)[:, 1]
            preds.extend(p.tolist())
            acts.extend([x[2] for x in by[m]])
            favimp.extend([x[3] for x in by[m]])
        for x in by[m]:
            Xh.append(feat(x))
            yh.append(x[2])
    return (roc_auc_score(acts, preds), np.array(preds), np.array(acts), np.array(favimp))


def _boot(a, B=10000):
    a = np.array(a); n = len(a)
    return np.percentile([a[RNG.randint(0, n, n)].mean() for _ in range(B)], [5, 95]) * 100


def main():
    rows = _load()
    base = np.mean([x[2] for x in rows]) * 100
    print(f"本命飛び(3着外) 対象{len(rows):,}R ベース率{base:.1f}%\n")

    print("=== 統制A/B/C: 飛び予測 AUC ===")
    auc_o, *_ = _wf_auc(rows, "odds")
    auc_c, *_ = _wf_auc(rows, "config")
    auc_b, pb, ab, fi = _wf_auc(rows, "both")
    print(f"  A オッズのみ(本命含意1特徴): AUC={auc_o:.3f}")
    print(f"  B 布置45次元のみ          : AUC={auc_c:.3f}")
    print(f"  C 布置+オッズ             : AUC={auc_b:.3f}")
    print(f"  → 布置の増分(C−A)= {auc_b-auc_o:+.3f}  "
          f"({'布置に市場外の飛び情報あり' if auc_b-auc_o>0.005 else '増分ほぼ無し=市場が織り込み済み'})\n")

    print("=== 本命オッズ帯を固定した中で、布置(C)高確信の飛び率がその帯ベースを超えるか ===")
    print(f"{'本命含意帯':>12s}{'n':>7s}{'帯ベース飛び':>11s}{'上位1/3確信の飛び':>16s}{'CI(5-95)':>13s}{'純増':>8s}")
    for lo, hi in [(0.0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 1.01)]:
        m = (fi >= lo) & (fi < hi)
        if m.sum() < 150:
            continue
        bl_base = ab[m].mean() * 100
        # その帯内で予測飛び確率上位1/3
        idx = np.where(m)[0]
        pin = pb[idx]
        thr = np.quantile(pin, 2 / 3)
        hi_idx = idx[pin >= thr]
        bl_hi = ab[hi_idx].mean() * 100
        p5, p95 = _boot(ab[hi_idx])
        flag = " ★純増" if p5 > bl_base else ""
        print(f"{f'{lo:.2f}-{hi:.2f}':>12s}{m.sum():7d}{bl_base:10.1f}%{bl_hi:15.1f}%"
              f"{f'[{p5:.0f},{p95:.0f}]':>13s}{bl_hi-bl_base:+7.1f}pt{flag}")
    print("\n ★純増=本命の市場強度を揃えても布置高確信が飛びを上乗せ予測＝市場未織り込みの痕跡。")
    print(" 無印なら飛び予兆は本命オッズが既に語る内容の焼き直し=個別エッジ無し。")


if __name__ == "__main__":
    main()


def full_control():
    """決定版統制: 市場の全オッズベクトル(全車含意勝率)で条件付けても布置が増分を持つか。"""
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rows = _load()
    # 各行に full odds vector(降順含意勝率8スロット)を付与
    races = E.load_block_a()
    key2 = {}
    for r in races:
        if not r.win_odds:
            continue
        s = sum(1.0/o for o in r.win_odds.values() if o>0)
        imps = sorted([(1.0/o)/s for o in r.win_odds.values() if o>0], reverse=True)
        imps = (imps+[0.0]*8)[:8]
        key2[r.key]= imps
    # rows は (_load由来で順序はenv同じ) -> race key を再取得するため再構築
    # _load は key を返さないので、ここで再ロードして揃える
    conn_rows=[]
    mu,sd=E.standardizer(races)
    for r in races: r.emb_std=(r.emb-mu)/sd
    import pandas as pd
    from .. import storage
    conn=storage.connect()
    fin=pd.read_sql_query("SELECT date,jcd,rno,car,finish FROM entries WHERE finish IS NOT NULL AND date BETWEEN ? AND ?",conn,params=E.BLOCK_A)
    conn.close()
    fin["jcd"]=fin["jcd"].astype(str).str.zfill(2)
    finmap={(d,j,r):dict(zip(g["car"].astype(int),g["finish"].astype(int))) for (d,j,r),g in fin.groupby(["date","jcd","rno"])}
    data=[]
    for r in races:
        if not r.win_odds: continue
        fav=min(r.win_odds,key=lambda c:r.win_odds[c])
        ff=finmap.get(r.key,{}).get(fav)
        if ff is None: continue
        data.append((r.ym, list(r.emb_std), key2[r.key], int(ff>3)))
    yms=sorted({x[0] for x in data})
    by={m:[x for x in data if x[0]==m] for m in yms}
    def run(mode):
        preds,acts=[],[]; Xh,yh=[],[]
        def feat(x):
            if mode=='oddsfull': return x[2]
            if mode=='config': return x[1]
            return x[1]+x[2]
        for mi,m in enumerate(yms):
            if mi>0 and len(yh)>=500 and len(set(yh))>1:
                clf=HistGradientBoostingClassifier(max_iter=200,max_depth=4,learning_rate=0.05,random_state=0)
                clf.fit(np.array(Xh),np.array(yh))
                Xm=np.array([feat(x) for x in by[m]])
                preds.extend(clf.predict_proba(Xm)[:,1].tolist()); acts.extend([x[3] for x in by[m]])
            for x in by[m]: Xh.append(feat(x)); yh.append(x[3])
        return roc_auc_score(acts,preds)
    ad=run('oddsfull'); ae=run('config'); af=run('oddsfull') # placeholder
    print("=== 決定版統制: 全オッズベクトルで条件付け ===")
    print(f"  D 全オッズベクトル(8含意)のみ : AUC={ad:.3f}")
    print(f"  B 布置のみ                   : AUC={ae:.3f}")
    both=run('both')
    print(f"  E 布置+全オッズ              : AUC={both:.3f}")
    print(f"  → 布置の純増分(E−D)= {both-ad:+.3f}  "
          f"({'市場未織り込みの布置情報あり=C投資の根拠' if both-ad>0.005 else 'ほぼ無し=市場が全て織り込み済み=個別エッジ無し'})")

if __name__ != "__main__":
    pass
