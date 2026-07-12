"""records で鋭くした荒れ較正で 3-C を回し直す。合格ライン=最厳リーク割引後も相対改善が残るか。

荒れスコア = 正則化GBM(config+records → P(本命勝ち)) の 1−P。WF・point-in-time。
(config天井を records が破ったのを受け、荒れフィルタを鋭くする。過学習抑制に正則化)
EV選択の確率源は読み出し修正済みの記憶(Harville→3連複)。本命抜き・記憶EV>1のみ。
二本立て(生 / リーク割引: 一律13/30, 差別 低13高30)でCI下限>0が残るか。

  python -m src.loop.verify3c_records
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors

from . import env as E
from .. import storage, history
from .verify2 import _memory_car_probs
from .verify_records import FAV_FEATS
from ..backtest_multi import _harville

RNG = np.random.RandomState(0)
K = 80
TUNE_FRAC = 0.6
ROUGH_Q = 0.667


def _prep():
    races = E.load_block_a()
    mu, sd = E.standardizer(races)
    for r in races:
        r.emb_std = (r.emb - mu) / sd
    conn = storage.connect()
    ent = pd.read_sql_query(
        "SELECT date,jcd,rno,car,player_id,vehicle_id,trial_record,start_timing,finish,win "
        "FROM entries WHERE finish IS NOT NULL AND date BETWEEN ? AND ?", conn, params=E.BLOCK_A)
    wo = pd.read_sql_query("SELECT date,jcd,rno,car,odds FROM win_odds WHERE odds>0 AND date BETWEEN ? AND ?",
                           conn, params=E.BLOCK_A)
    od = pd.read_sql_query("SELECT date,jcd,rno,combo,odds FROM odds WHERE bet='trio' AND date BETWEEN ? AND ?",
                           conn, params=E.BLOCK_A)
    conn.close()
    for dd in (ent, wo, od):
        dd["jcd"] = dd["jcd"].astype(str).str.zfill(2)
    ent = history.add_history(ent)
    hmap = {(r.date, r.jcd, int(r.rno), int(r.car)): r for r in ent.itertuples()}
    finmap = {}
    for (d, j, r), g in ent.groupby(["date", "jcd", "rno"]):
        finmap[(d, j, r)] = dict(zip(g["car"].astype(int), g["finish"].astype(int)))
    womap = {(d, j, r): dict(zip(g["car"].astype(int), g["odds"].astype(float)))
             for (d, j, r), g in wo.groupby(["date", "jcd", "rno"])}
    lad = {(d, j, r): dict(zip(g["combo"], g["odds"])) for (d, j, r), g in od.groupby(["date", "jcd", "rno"])}
    return races, hmap, finmap, womap, lad


def _rec_feat(key, fav, hmap, wodds):
    hf = hmap.get((key[0], key[1], key[2], fav))
    if hf is None:
        return None
    fav_v = [getattr(hf, c, np.nan) for c in FAV_FEATS]
    r3 = [getattr(hmap.get((key[0], key[1], key[2], c)), "p_r3_order", np.nan)
          for c in wodds if c != fav]
    r3 = [v for v in r3 if v == v]
    fb = min(r3) if r3 else np.nan
    return fav_v + [fb, np.mean(r3) if r3 else np.nan,
                    (getattr(hf, "p_r3_order", np.nan) - fb) if r3 else np.nan]


def main():
    races, hmap, finmap, womap, lad = _prep()
    months = sorted({r.ym for r in races})
    by = {m: [r for r in races if r.ym == m] for m in months}

    # WF: 記憶car確率 + records荒れスコア + 本命抜きEV>1・3連複の(staked,returned)
    memX, memY = [], []
    Xhist, yhist = [], []          # 荒れGBM学習用(config+records, label=本命勝ち)
    recs = []                      # (date, rough_sharp, staked, returned)
    for mi, m in enumerate(months):
        rough_model = None
        if mi > 0 and len(yhist) >= 500 and len(set(yhist)) > 1:
            rough_model = HistGradientBoostingClassifier(
                max_iter=150, max_leaf_nodes=15, min_samples_leaf=50,
                l2_regularization=1.0, learning_rate=0.05, random_state=0)
            rough_model.fit(np.array(Xhist), np.array(yhist))
        if mi > 0 and len(memX) >= K:
            nn = NearestNeighbors(n_neighbors=min(K, len(memX))); nn.fit(np.vstack(memX)); ya = np.array(memY)
            Q = np.vstack([r.emb_std for r in by[m]]); D, I = nn.kneighbors(Q)
            for r, dd, ii in zip(by[m], D, I):
                wodds = womap.get(r.key); L = lad.get(r.key); fm = finmap.get(r.key, {})
                if not wodds or not L or len(wodds) < 4:
                    continue
                fav = min(wodds, key=lambda c: wodds[c])
                rf = _rec_feat(r.key, fav, hmap, wodds)
                if rf is None or rough_model is None:
                    continue
                feat = np.concatenate([r.emb, rf]).reshape(1, -1)
                rough = 1.0 - rough_model.predict_proba(feat)[0, 1]    # 鋭い荒れスコア
                mp = _memory_car_probs(dd, ii, ya, r.cars_by_rank, wodds)
                if not mp:
                    continue
                top3 = sorted([c for c in fm if fm[c] > 0], key=lambda c: fm[c])[:3]
                if len(top3) < 3:
                    continue
                real = "-".join(map(str, sorted(top3)))
                probs = _harville(mp).get("trio", {})
                st = rt = 0.0
                for combo, p in probs.items():
                    cs = set(int(x) for x in combo.split("-"))
                    if fav in cs:
                        continue
                    o = L.get(combo)
                    if o and o > 0 and p * o > 1.0:
                        st += 100; rt += o * 100 if combo == real else 0.0
                if st > 0:
                    recs.append((r.date, rough, st, rt))
        # メモリ & 荒れGBM学習データを追記
        for r in by[m]:
            memX.append(r.emb_std); memY.append(r.winner_rank)
            wodds = womap.get(r.key)
            if not wodds:
                continue
            fav = min(wodds, key=lambda c: wodds[c])
            rf = _rec_feat(r.key, fav, hmap, wodds)
            if rf is None:
                continue
            Xhist.append(list(r.emb) + rf)
            yhist.append(1 if finmap.get(r.key, {}).get(fav) == 1 else 0)

    # tune/test + 二本立て割引
    dates = sorted({x[0] for x in recs})
    cut = dates[int(len(dates) * TUNE_FRAC)]
    tune = [x for x in recs if x[0] < cut]
    thr = np.quantile([x[1] for x in tune], ROUGH_Q)
    test = [x for x in recs if x[0] >= cut]
    hi = [(s, r) for d, ro, s, r in test if ro >= thr]
    lo = [(s, r) for d, ro, s, r in test if ro < thr]
    print(f"records鋭化・荒れフィルタで3-C再試行。test 高荒れ{len(hi):,}R / 低荒れ{len(lo):,}R "
          f"(閾値roughness≥{thr:.3f})\n")
    print(f"{'割引モデル':>22s}{'高ROI':>7s}{'低ROI':>7s}{'相対':>7s}{'CI(5-95)':>13s}{'P(≤0)':>7s}")
    for name, dh, dl in [("生", 0., 0.), ("一律13%", .13, .13), ("一律30%", .30, .30),
                         ("差別 低13/高30(最厳)", .30, .13)]:
        sh = np.array([x[0] for x in hi]); rh = np.array([x[1] * (1 - dh) for x in hi])
        sl = np.array([x[0] for x in lo]); rl = np.array([x[1] * (1 - dl) for x in lo])
        roi_h = rh.sum() / sh.sum() * 100; roi_l = rl.sum() / sl.sum() * 100
        nh, nl = len(hi), len(lo)
        diff = np.array([rh[RNG.randint(0, nh, nh)].sum() / sh[RNG.randint(0, nh, nh)].sum() * 100
                         - rl[RNG.randint(0, nl, nl)].sum() / sl[RNG.randint(0, nl, nl)].sum() * 100
                         for _ in range(8000)])
        p5 = np.percentile(diff, 5); ple = (diff <= 0).mean() * 100
        flag = "✓残存" if p5 > 0 else "✗消失"
        print(f"{name:>22s}{roi_h:6.0f}%{roi_l:6.0f}%{roi_h-roi_l:+6.0f}{f'[{p5:+.0f},{np.percentile(diff,95):+.0f}]':>13s}{ple:6.1f}% {flag}")
    print("\n ※これは『勝ち』ではない。確定オッズ由来の相対候補。真の絶対額は(b)締切前オッズで殴るまで名乗らない。")


if __name__ == "__main__":
    main()
