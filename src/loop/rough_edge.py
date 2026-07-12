"""タスクB2：市場の値付けの歪み(較正の破れ)を個別レース布置から予測できるか。

Bは「勝者当て」で市場と同じ土俵に負けた。だが暗黙知の本命は勝者当てでなく
「市場オッズが間違ったレースの検出」。ここでは順位でなく荒れの予兆を測る:

  1) 全体較正監査: 市場含意勝率(単勝オッズ正規化) vs 実現。綺麗なら市場は平均で正しい。
  2) 個別の破れ: ターゲット=市場1番人気(最低オッズ)が飛んだ(3着外)か否か。
     布置(45次元・オッズ非使用)から強い分類器(GBM)で予測できるか。WF・リーク無。
  3) 布置から「本命飛ぶ」と高確信で絞った群の実飛び率が母集団ベース率を有意に超えるか。
     超える→市場が織り込めてない布置情報あり(C投資の根拠)。超えない→歪みも布置に痕跡なし。

  python -m src.loop.rough_edge
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import env as E
from .. import storage

RNG = np.random.RandomState(0)


def _calibration_audit():
    """市場含意勝率 vs 実現(ブロックA・単勝)。"""
    conn = storage.connect()
    ent = pd.read_sql_query(
        "SELECT date,jcd,rno,car,win FROM entries WHERE win IS NOT NULL AND date BETWEEN ? AND ?",
        conn, params=E.BLOCK_A)
    od = pd.read_sql_query(
        "SELECT date,jcd,rno,car,odds FROM win_odds WHERE odds>0 AND date BETWEEN ? AND ?",
        conn, params=E.BLOCK_A)
    conn.close()
    ent["jcd"] = ent["jcd"].astype(str).str.zfill(2)
    od["jcd"] = od["jcd"].astype(str).str.zfill(2)
    df = ent.merge(od, on=["date", "jcd", "rno", "car"], how="inner")
    inv = 1.0 / df["odds"]
    df["implied"] = inv / inv.groupby([df["date"], df["jcd"], df["rno"]]).transform("sum")
    print("=== 1) 全体較正監査(市場含意勝率 vs 実現・ブロックA) ===")
    print(f"{'含意帯':>12s}{'車数':>9s}{'平均含意':>9s}{'実現勝率':>9s}{'ズレ':>7s}")
    for lo, hi in [(0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .45), (.45, .7), (.7, 1.01)]:
        m = (df["implied"] >= lo) & (df["implied"] < hi)
        s = df[m]
        if len(s) < 50:
            continue
        imp, act = s["implied"].mean() * 100, s["win"].mean() * 100
        print(f"{f'{lo:.2f}-{hi:.2f}':>12s}{len(s):9,d}{imp:8.1f}%{act:8.1f}%{act-imp:+6.1f}pt")
    print(" → ズレが各帯で小さければ市場は平均で正しく較正(効率市場)。\n")


def _load_blowout():
    """各レースの (emb, 本命飛んだか) を time-order で。本命=最低オッズ車、飛ぶ=3着外。"""
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
        fm = finmap.get(r.key, {})
        ff = fm.get(fav)
        if ff is None:
            continue
        blow = int(ff > 3)                    # 本命が3着外=飛んだ
        rows.append((r.ym, r.emb_std, blow, r.win_odds[fav]))
    return rows


def _boot_rate(a, B=10000):
    a = np.array(a)
    n = len(a)
    boot = np.array([a[RNG.randint(0, n, n)].mean() for _ in range(B)]) * 100
    return np.percentile(boot, [5, 95])


def main():
    _calibration_audit()

    rows = _load_blowout()
    yms = sorted({x[0] for x in rows})
    base = np.mean([x[2] for x in rows]) * 100
    print(f"=== 2)3) 本命(市場1番人気)飛び=3着外を布置(45次元・オッズ非使用)から予測(WF/GBM) ===")
    print(f"対象 {len(rows):,}R / 本命飛びベース率={base:.1f}%\n")

    # WF: 各月を過去月だけで学習したGBMで予測
    by = {m: [x for x in rows if x[0] == m] for m in yms}
    preds, acts = [], []
    Xhist, yhist = [], []
    for mi, m in enumerate(yms):
        if mi > 0 and len(yhist) >= 500 and len(set(yhist)) > 1:
            clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                                 learning_rate=0.05, random_state=0)
            clf.fit(np.vstack(Xhist), np.array(yhist))
            Xm = np.vstack([x[1] for x in by[m]])
            p = clf.predict_proba(Xm)[:, 1]
            preds.extend(p.tolist())
            acts.extend([x[2] for x in by[m]])
        for x in by[m]:
            Xhist.append(x[1])
            yhist.append(x[2])
    preds, acts = np.array(preds), np.array(acts)
    auc = roc_auc_score(acts, preds)
    print(f"WF予測 {len(preds):,}R  AUC={auc:.3f}（0.5=布置に飛び予兆なし）\n")

    # 予測確信の五分位ごと実飛び率
    print(f"{'予測飛び確率帯':>14s}{'n':>7s}{'平均予測':>9s}{'実飛び率':>9s}{'CI(5-95)':>15s}{'vsベース':>9s}")
    order = np.argsort(preds)
    q = np.quantile(preds, [0, .2, .4, .6, .8, 1.0])
    for i in range(5):
        lo, hi = q[i], q[i + 1]
        m = (preds >= lo) & (preds <= hi) if i == 4 else (preds >= lo) & (preds < hi)
        if m.sum() < 50:
            continue
        act = acts[m].mean() * 100
        p5, p95 = _boot_rate(acts[m])
        flag = " ★超" if p5 > base else ""
        print(f"{f'Q{i+1}[{lo:.2f},{hi:.2f}]':>14s}{m.sum():7d}{preds[m].mean()*100:8.1f}%"
              f"{act:8.1f}%{f'[{p5:.0f},{p95:.0f}]':>15s}{act-base:+7.1f}pt{flag}")

    # 最高確信トップ群(上位10%)
    k = max(50, len(preds) // 10)
    top = order[-k:]
    act = acts[top].mean() * 100
    p5, p95 = _boot_rate(acts[top])
    print(f"\n上位10%確信({k}R): 実飛び率 {act:.1f}% [CI {p5:.0f},{p95:.0f}] vs ベース{base:.1f}% "
          f"→ {'★ベース有意超え(歪み痕跡あり)' if p5 > base else '未達(痕跡なし)'}")
    print("\n ★=実飛び率がベース率をCI下限>ベースで超過＝市場が織り込めてない布置歪みの痕跡。")


if __name__ == "__main__":
    main()
