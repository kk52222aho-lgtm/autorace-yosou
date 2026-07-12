"""(a)を縛り込みで：3-C受け皿を二本立て(生の相対 / リーク割引後)で判定する。

今日の教訓の実装＝確定オッズの甘い相対値を絶対の勝ちと勘違いさせない。
  1. 生の相対改善(確定オッズ, 高荒れ絞る vs 絞らない)をレース単位ブロックブートで。
  2. その相対値に「リーク割引」をかけた保守値。今夜実測の単勝中央13%を下限に、
     荒れレース・exoticはさらに大きい前提で 差別割引(低荒れ13% / 高荒れ30%) も出す。
  3. 合格ライン(事前登録): リーク割引後も相対改善が有意(ブートCI下限>0)に残る時だけ受け皿候補。
     割引で消えるなら「確定オッズ蜃気楼の可能性」と明記して保留。
  4. 合格しても「勝ち」と書かない。ステータスは「(b)ライブ蓄積で絶対額を殴る価値がある候補」まで。

  python -m src.loop.verify3c_discount
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import env as E
from .. import storage
from .verify2 import _memory_car_probs
from ..backtest_multi import _harville

RNG = np.random.RandomState(0)
K = 80
TUNE_FRAC = 0.6
ROUGH_Q = 0.667
LEAK_WIN_MEASURED = 0.13         # 今夜実測(単勝中央値)
DISCLAIMER = ("※これは『勝ち』ではない。確定オッズ由来の相対候補であり、真の絶対額は "
              "(b)締切前オッズのライブ蓄積で殴るまで名乗らない。")


def _recs():
    races = E.load_block_a()
    mu, sd = E.standardizer(races)
    for r in races:
        r.emb_std = (r.emb - mu) / sd
    import pandas as pd
    conn = storage.connect()
    od = pd.read_sql_query(
        "SELECT date,jcd,rno,combo,odds FROM odds WHERE bet='trio' AND date BETWEEN ? AND ?",
        conn, params=E.BLOCK_A)
    fin = pd.read_sql_query(
        "SELECT date,jcd,rno,car,finish FROM entries WHERE finish IS NOT NULL AND date BETWEEN ? AND ?",
        conn, params=E.BLOCK_A)
    conn.close()
    for dd in (od, fin):
        dd["jcd"] = dd["jcd"].astype(str).str.zfill(2)
    lad = {(d, j, r): dict(zip(g["combo"], g["odds"])) for (d, j, r), g in od.groupby(["date", "jcd", "rno"])}
    finmap = {(d, j, r): dict(zip(g["car"].astype(int), g["finish"].astype(int)))
              for (d, j, r), g in fin.groupby(["date", "jcd", "rno"])}
    months = sorted({r.ym for r in races})
    by = {m: [r for r in races if r.ym == m] for m in months}
    memX, memY, rows = [], [], []
    for mi, m in enumerate(months):
        if mi > 0 and len(memX) >= K:
            nn = NearestNeighbors(n_neighbors=min(K, len(memX))); nn.fit(np.vstack(memX)); ya = np.array(memY)
            Q = np.vstack([r.emb_std for r in by[m]]); D, I = nn.kneighbors(Q)
            for r, dd, ii in zip(by[m], D, I):
                L = lad.get(r.key); fm = finmap.get(r.key, {})
                if not L or len(r.win_odds) < 4:
                    continue
                mp = _memory_car_probs(dd, ii, ya, r.cars_by_rank, r.win_odds)
                if not mp:
                    continue
                fav = min(r.win_odds, key=lambda c: r.win_odds[c])
                rough = 1.0 - mp.get(fav, 0.0)
                top3 = sorted([c for c in fm if fm[c] > 0], key=lambda c: fm[c])[:3]
                if len(top3) < 3:
                    continue
                real = "-".join(map(str, sorted(top3)))
                probs = _harville(mp).get("trio", {})
                staked = won_ret = 0.0
                for combo, p in probs.items():
                    cs = set(int(x) for x in combo.split("-"))
                    if fav in cs:
                        continue
                    o = L.get(combo)
                    if o and o > 0 and p * o > 1.0:
                        staked += 100
                        won_ret += o * 100 if combo == real else 0.0
                if staked > 0:
                    rows.append((r.date, rough, staked, won_ret))
        for r in by[m]:
            memX.append(r.emb_std); memY.append(r.winner_rank)
    return rows


def _roi_boot_diff(hi, lo, dh, dl, B=10000):
    """hi/lo: [(staked, returned)]。割引 dh/dl 適用後の (ROI_hi, ROI_lo, diff分布)。"""
    def arrs(g, d):
        st = np.array([x[0] for x in g], float)
        rt = np.array([x[1] * (1 - d) for x in g], float)
        return st, rt
    sh, rh = arrs(hi, dh); sl, rl = arrs(lo, dl)
    roi_h = rh.sum() / sh.sum() * 100; roi_l = rl.sum() / sl.sum() * 100
    nh, nl = len(hi), len(lo)
    diff = np.empty(B)
    for b in range(B):
        ih = RNG.randint(0, nh, nh); il = RNG.randint(0, nl, nl)
        diff[b] = rh[ih].sum() / sh[ih].sum() * 100 - rl[il].sum() / sl[il].sum() * 100
    return roi_h, roi_l, diff


def main():
    rows = _recs()
    dates = sorted({x[0] for x in rows})
    cut = dates[int(len(dates) * TUNE_FRAC)]
    tune = [x for x in rows if x[0] < cut]
    test = [x for x in rows if x[0] >= cut]
    thr = np.quantile([x[1] for x in tune], ROUGH_Q)
    hi = [(s, r) for d, ro, s, r in test if ro >= thr]
    lo = [(s, r) for d, ro, s, r in test if ro < thr]
    print(f"事前登録1セル: 本命抜き3連複・記憶EV>1・高荒れ(tune上位1/3 roughness≥{thr:.3f})")
    print(f"test: 高荒れ{len(hi):,}R / 低荒れ{len(lo):,}R\n")

    print(f"{'割引モデル':>22s}{'高荒れROI':>10s}{'低荒れROI':>10s}{'相対改善':>10s}{'CI(5-95)':>14s}{'P(≤0)':>8s}")
    scenarios = [
        ("生(割引なし)", 0.0, 0.0),
        (f"一律{int(LEAK_WIN_MEASURED*100)}%(実測単勝)", LEAK_WIN_MEASURED, LEAK_WIN_MEASURED),
        ("一律30%(保守)", 0.30, 0.30),
        ("差別 低13%/高30%(最厳)", 0.30, 0.13),
    ]
    survive = True
    for name, dh, dl in scenarios:
        roi_h, roi_l, diff = _roi_boot_diff(hi, lo, dh, dl)
        p5, p95 = np.percentile(diff, [5, 95]); ple = (diff <= 0).mean() * 100
        flag = " ✓残存" if p5 > 0 else " ✗消失"
        if name.startswith("差別") and p5 <= 0:
            survive = False
        print(f"{name:>22s}{roi_h:9.0f}%{roi_l:9.0f}%{roi_h-roi_l:+8.0f}pt"
              f"{f'[{p5:+.0f},{p95:+.0f}]':>14s}{ple:7.1f}%{flag}")

    print()
    if survive:
        print("判定: リーク割引(最厳・差別)後も相対改善が有意に残る → 受け皿候補として (b) へ進む価値あり。")
    else:
        print("判定: 最厳割引で相対改善が消失 → 確定オッズ蜃気楼の可能性。(b)で殴るまで保留。")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
