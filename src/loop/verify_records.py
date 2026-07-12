"""records(選手別1走ずつの近況=history.py)を新情報として足し、config天井0.60を破れるか。

records は point-in-time(shift().expanding()でリーク無)で configに無い『直近の動き』:
  本命の 直近3走フォーム/個人試走比/ST/マシン地力 と、挑戦馬群の直近フォーム。
label=市場本命が勝つ。config天井0.60 を超えるか、訓練/未触ギャップと較正を併記して判定。

  python -m src.loop.verify_records
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import env as E
from .. import storage, history

RNG = np.random.RandomState(0)
FAV_FEATS = ["p_n", "p_avg_order", "p_win_rate", "p_r3_order", "p_avg_st",
             "trial_dev", "v_avg_order", "v_win_rate"]


def _load():
    races = E.load_block_a()
    conn = storage.connect()
    ent = pd.read_sql_query(
        "SELECT date,jcd,rno,car,player_id,vehicle_id,trial_record,start_timing,finish,win "
        "FROM entries WHERE finish IS NOT NULL AND date BETWEEN ? AND ?", conn, params=E.BLOCK_A)
    wo = pd.read_sql_query(
        "SELECT date,jcd,rno,car,odds FROM win_odds WHERE odds>0 AND date BETWEEN ? AND ?",
        conn, params=E.BLOCK_A)
    conn.close()
    ent["jcd"] = ent["jcd"].astype(str).str.zfill(2)
    wo["jcd"] = wo["jcd"].astype(str).str.zfill(2)
    ent = history.add_history(ent)     # point-in-time 近況特徴を付与
    hmap = {(r.date, r.jcd, int(r.rno), int(r.car)): r for r in ent.itertuples()}
    womap = {}
    for (d, j, r), g in wo.groupby(["date", "jcd", "rno"]):
        womap[(d, j, r)] = dict(zip(g["car"].astype(int), g["odds"].astype(float)))

    X, y, M = [], [], []
    for r in races:
        wodds = womap.get(r.key)
        if not wodds:
            continue
        fav = min(wodds, key=lambda c: wodds[c])
        hf = hmap.get((r.key[0], r.key[1], r.key[2], fav))
        if hf is None:
            continue
        fav_v = [getattr(hf, c, np.nan) for c in FAV_FEATS]
        # 挑戦馬群(非本命)の直近フォーム集約
        others = [hmap.get((r.key[0], r.key[1], r.key[2], c)) for c in wodds if c != fav]
        r3 = [getattr(o, "p_r3_order", np.nan) for o in others if o is not None]
        r3 = [v for v in r3 if v == v]
        field_best = min(r3) if r3 else np.nan
        field_mean = np.mean(r3) if r3 else np.nan
        gap_r3 = (getattr(hf, "p_r3_order", np.nan) - field_best) if r3 else np.nan
        rec = fav_v + [field_best, field_mean, gap_r3]
        X.append((list(r.emb), rec))
        y.append(1 if r.winner_car == fav else 0)
        M.append(r.ym)
    cfg = np.array([x[0] for x in X])
    rec = np.array([x[1] for x in X], dtype=float)
    return cfg, rec, np.array(y), np.array(M)


def _gbm_auc(Xtr, ytr, Xte, yte):
    g = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=0)
    g.fit(Xtr, ytr)
    return (roc_auc_score(ytr, g.predict_proba(Xtr)[:, 1]),
            roc_auc_score(yte, g.predict_proba(Xte)[:, 1]),
            g.predict_proba(Xte)[:, 1])


def main():
    cfg, rec, y, M = _load()
    tr = np.array([m <= E.TRAIN_END_YM for m in M])
    hd = np.array([m in E.HELDOUT_YMS for m in M])
    print(f"対象 {len(y):,} (train {tr.sum():,} / held {hd.sum():,}) 本命勝率 未触{y[hd].mean()*100:.1f}%")
    print("label=市場本命が勝つ。config天井=未触0.60。records=point-in-time近況(configに無い)\n")

    print(f"{'特徴':>16s}{'訓練AUC':>9s}{'未触AUC':>9s}{'ギャップ':>9s}")
    combos = {
        "config のみ": cfg,
        "records のみ": rec,
        "config+records": np.hstack([cfg, rec]),
    }
    held = {}
    for name, X in combos.items():
        a_tr, a_hd, p_hd = _gbm_auc(X[tr], y[tr], X[hd], y[hd])
        held[name] = (a_hd, p_hd)
        print(f"{name:>16s}{a_tr:8.3f}{a_hd:8.3f}{a_tr-a_hd:+8.3f}")

    base = held["config のみ"][0]
    ceiling = 0.60
    best_name = "config+records"
    a_hd, p_hd = held[best_name]
    gap = a_hd - base
    print(f"\n config+records 未触AUC {a_hd:.3f} − config単独 {base:.3f} = {gap:+.3f} "
          f"(config天井{ceiling}比 {a_hd-ceiling:+.3f})")

    # 較正(config+records 未触)
    print("\n--- config+records 未触の較正(予測P(本命勝ち)五分位 → 実勝率) ---")
    q = np.quantile(p_hd, [0, .2, .4, .6, .8, 1.0])
    yhd = y[hd]
    calib_ok = True
    for i in range(5):
        m = (p_hd >= q[i]) & (p_hd <= q[i + 1]) if i == 4 else (p_hd >= q[i]) & (p_hd < q[i + 1])
        pe, ac = p_hd[m].mean() * 100, yhd[m].mean() * 100
        if abs(pe - ac) > 8:
            calib_ok = False
        print(f"    予測{pe:4.0f}% → 実{ac:4.0f}%  (n={m.sum()})")

    # 訓練/未触ギャップの過学習判定
    ov_gap = _gbm_auc(np.hstack([cfg, rec])[tr], y[tr], np.hstack([cfg, rec])[hd], y[hd])
    over = ov_gap[0] - ov_gap[1]
    print(f"\n判定: 未触AUC{a_hd:.3f} / 天井超え{'YES' if a_hd > ceiling + 0.005 else 'NO'} "
          f"/ 過学習ギャップ{over:+.3f}({'小' if over < 0.08 else '大=丸暗記の顔'}) "
          f"/ 較正{'維持' if calib_ok else '崩れ'}")
    if a_hd > ceiling + 0.005 and over < 0.08 and calib_ok:
        print(" → 三つ揃い＝天井突破。次は records込みの鋭い記憶で 3-C を回し直す。")
    else:
        print(" → 天井破れず(または過学習/較正崩れ)＝configもrecordsも天井。幅は(b)を待つしかない。")


if __name__ == "__main__":
    main()
