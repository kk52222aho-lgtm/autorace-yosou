"""状況クラスを鋭くする：(1)K探索 (2)records込みクラス。過学習は訓練vs未触で監視。

クラス定義の埋め込みに config(布置45) と records(近況11=history.py) を選べる。
K を掃引し、未触AUC(P本命勝ち)とクラスtop単勝ROIで最良を選ぶ。Kを上げると各クラスが
薄くなり教訓が過学習→未触が頭打ち/悪化。訓練AUCと並べて「鋭くなった顔」を却下する。

  python -m src.loop.situation_sharpen
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

from . import env as E
from .. import storage, history
from .situation_class import SituationClassAgent, _roi_boot
from .verify_records import FAV_FEATS

RNG = np.random.RandomState(0)
ALPHA = 25.0


def _records_map():
    """レースキー → 本命中心の近況特徴(11)。point-in-time(history.py)。"""
    conn = storage.connect()
    ent = pd.read_sql_query(
        "SELECT date,jcd,rno,car,player_id,vehicle_id,trial_record,start_timing,finish,win "
        "FROM entries WHERE finish IS NOT NULL AND date BETWEEN ? AND ?", conn, params=E.BLOCK_A)
    wo = pd.read_sql_query("SELECT date,jcd,rno,car,odds FROM win_odds WHERE odds>0 AND date BETWEEN ? AND ?",
                           conn, params=E.BLOCK_A)
    conn.close()
    ent["jcd"] = ent["jcd"].astype(str).str.zfill(2); wo["jcd"] = wo["jcd"].astype(str).str.zfill(2)
    ent = history.add_history(ent)
    hmap = {(r.date, r.jcd, int(r.rno), int(r.car)): r for r in ent.itertuples()}
    womap = {(d, j, r): dict(zip(g["car"].astype(int), g["odds"].astype(float)))
             for (d, j, r), g in wo.groupby(["date", "jcd", "rno"])}
    out = {}
    for key, wod in womap.items():
        fav = min(wod, key=lambda c: wod[c])
        hf = hmap.get((key[0], key[1], key[2], fav))
        if hf is None:
            continue
        fav_v = [getattr(hf, c, np.nan) for c in FAV_FEATS]
        r3 = [getattr(hmap.get((key[0], key[1], key[2], c)), "p_r3_order", np.nan) for c in wod if c != fav]
        r3 = [v for v in r3 if v == v]
        fb = min(r3) if r3 else np.nan
        out[key] = fav_v + [fb, np.mean(r3) if r3 else np.nan,
                            (getattr(hf, "p_r3_order", np.nan) - fb) if r3 else np.nan]
    return out


def _run(races, embs, train_mask, K):
    """embs: 各レースのクラス用埋め込み(標準化済)。時系列predict-then-learnで未触AUC/ROI。"""
    km = KMeans(n_clusters=K, n_init=4, random_state=0)
    km.fit(embs[train_mask])
    grank = np.zeros(E.MAXC + 1)
    for i, r in enumerate(races):
        if train_mask[i] and 1 <= r.winner_rank <= E.MAXC:
            grank[r.winner_rank] += 1
    grank /= grank.sum()
    agent = SituationClassAgent(km.cluster_centers_, grank)
    agent.counts = np.zeros((K, E.MAXC + 1)); agent.n = np.zeros(K)
    p_h, y_h, bets_h = [], [], []
    p_t, y_t = [], []
    for i, r in enumerate(races):
        cls = int(np.argmin(((km.cluster_centers_ - embs[i]) ** 2).sum(1)))
        if r.win_odds and agent.n[cls] >= 1:
            cp = agent.car_probs(cls, r.cars_by_rank, r.win_odds)
            if cp:
                fav = min(r.win_odds, key=lambda c: r.win_odds[c])
                pf = cp.get(fav, 0.0); yf = 1 if r.winner_car == fav else 0
                if r.ym in E.HELDOUT_YMS:
                    mt = max(cp, key=lambda c: cp[c])
                    p_h.append(pf); y_h.append(yf); bets_h.append((r.win_odds[mt], mt == r.winner_car))
                elif train_mask[i]:
                    p_t.append(pf); y_t.append(yf)
        agent.learn(cls, r.winner_rank)
    auc_h = roc_auc_score(y_h, p_h) if len(set(y_h)) > 1 else float("nan")
    auc_t = roc_auc_score(y_t, p_t) if len(set(y_t)) > 1 else float("nan")
    roi, p5, p95 = _roi_boot(bets_h)
    return auc_h, auc_t, roi, p5, p95


def main():
    races = E.load_block_a()
    mu, sd = E.standardizer(races)
    for r in races:
        r.emb_std = (r.emb - mu) / sd
    races.sort(key=lambda r: (r.date, r.key[2], r.key[1]))
    train_mask = np.array([r.ym <= E.TRAIN_END_YM for r in races])
    cfg = np.vstack([r.emb_std for r in races])

    # records を付与(欠損は訓練中央値で補完→標準化)
    rmap = _records_map()
    rec_raw = np.array([rmap.get(r.key, [np.nan] * 11) for r in races], dtype=float)
    med = np.nanmedian(rec_raw[train_mask], axis=0)
    inds = np.where(np.isnan(rec_raw))
    rec_raw[inds] = np.take(med, inds[1])
    rmu, rsd = rec_raw[train_mask].mean(0), rec_raw[train_mask].std(0); rsd[rsd == 0] = 1
    rec = (rec_raw - rmu) / rsd
    cfg_rec = np.hstack([cfg, rec])

    print("状況クラスの鋭化：K探索 × records込み(訓練AUCを併記して過学習監視)\n")
    print(f"{'埋め込み':>14s}{'K':>5s}{'未触AUC':>9s}{'訓練AUC':>9s}{'gap':>7s}"
          f"{'クラスtopROI':>12s}{'CI':>12s}")
    best = None
    for name, embs in [("config", cfg), ("config+records", cfg_rec)]:
        for K in [30, 60, 100, 150, 250, 400]:
            ah, at, roi, p5, p95 = _run(races, embs, train_mask, K)
            gap = at - ah
            mark = ""
            if best is None or (ah > best[0] and gap < 0.06):
                best = (ah, name, K, roi, p5, p95); mark = " ←最良候補"
            print(f"{name:>14s}{K:5d}{ah:8.3f}{at:8.3f}{gap:+7.3f}"
                  f"{roi:10.0f}%{f'[{p5:.0f},{p95:.0f}]':>12s}{mark}")
    print(f"\n連続kNN基準: AUC0.604 / mem_top ROI74%")
    print(f"最良(未触AUC最大かつgap<0.06): {best[1]} K={best[2]} "
          f"未触AUC{best[0]:.3f} / クラスtopROI{best[3]:.0f}%[{best[4]:.0f},{best[5]:.0f}]")
    print(" ※gap大(訓練≫未触)はKを上げ過ぎた過学習=却下。未触が頭打ちするKの手前を選ぶ。")


if __name__ == "__main__":
    main()
