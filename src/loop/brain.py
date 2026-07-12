"""ライブ予想の頭脳：ブロックA記憶(読み出し修正=較正済み確率)で、生の番組表から
各車の勝率・roughness・買い目を出す。過去データで検証した器をそのまま前向きに使う。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans

from . import env as E
from .. import storage, history
from .verify2 import _memory_car_probs
from .verify_records import FAV_FEATS
from .situation_class import SituationClassAgent
from ..backtest_multi import _harville

# 3-Cで事前登録した高荒れ閾値(memory scale, tune上位1/3)。STリーク除去後の再校正値。前向きでは動かさない。
ROUGH_THR = 0.801
K_SELECTOR = 150             # Stream Bセレクタの状況クラス数(鋭化探索の最良)
_TRACK = E._TRACK


class Brain:
    def __init__(self):
        self.races = E.load_block_a()
        self.mu, self.sd = E.standardizer(self.races)
        for r in self.races:
            r.emb_std = (r.emb - self.mu) / self.sd
        self.nn = NearestNeighbors(n_neighbors=80)
        self.nn.fit(np.vstack([r.emb_std for r in self.races]))
        self.ya = np.array([r.winner_rank for r in self.races])
        conn = storage.connect()
        rp = pd.read_sql_query("SELECT rec_point FROM entries WHERE date BETWEEN ? AND ?",
                               conn, params=E.BLOCK_A)
        conn.close()
        v = rp["rec_point"].fillna(rp["rec_point"].median())
        self.rp_mean, self.rp_std = float(v.mean()), float(v.std() or 1.0)
        self._build_selector()

    def _build_selector(self):
        """Stream B用: config+records の状況クラス・セレクタ(K=150)を組む。
        records=history.py近況。選手ごとの最新近況(player_latest)をライブで使う。"""
        conn = storage.connect()
        ent = pd.read_sql_query(
            "SELECT date,jcd,rno,car,player_id,vehicle_id,trial_record,start_timing,finish,win "
            "FROM entries WHERE finish IS NOT NULL", conn)
        wo = pd.read_sql_query(
            "SELECT date,jcd,rno,car,odds FROM win_odds WHERE odds>0 AND date BETWEEN ? AND ?",
            conn, params=E.BLOCK_A)
        conn.close()
        ent["jcd"] = ent["jcd"].astype(str).str.zfill(2)
        wo["jcd"] = wo["jcd"].astype(str).str.zfill(2)
        ent = history.add_history(ent)
        # 選手ごとの最新近況(ライブ用): 最終行の履歴特徴
        self.player_latest = {}
        for pid, g in ent.groupby("player_id"):
            last = g.iloc[-1]
            self.player_latest[pid] = {c: last.get(c) for c in FAV_FEATS + ["p_r3_order"]}
        # ブロックA各レースの本命中心records
        entA = ent[(ent["date"] >= E.BLOCK_A[0]) & (ent["date"] <= E.BLOCK_A[1])]
        hmap = {(r.date, r.jcd, int(r.rno), int(r.car)): r for r in entA.itertuples()}
        womap = {(d, j, r): dict(zip(g["car"].astype(int), g["odds"].astype(float)))
                 for (d, j, r), g in wo.groupby(["date", "jcd", "rno"])}
        rmap = {}
        for key, wod in womap.items():
            fav = min(wod, key=lambda c: wod[c])
            hf = hmap.get((key[0], key[1], key[2], fav))
            if hf is None:
                continue
            rmap[key] = self._rec_vec([getattr(hf, c, np.nan) for c in FAV_FEATS],
                                      [getattr(hmap.get((key[0], key[1], key[2], c)), "p_r3_order", np.nan)
                                       for c in wod if c != fav])
        recs = np.array([rmap.get(r.key, [np.nan] * 11) for r in self.races], dtype=float)
        self.rec_med = np.nanmedian(recs, axis=0)
        inds = np.where(np.isnan(recs)); recs[inds] = np.take(self.rec_med, inds[1])
        self.rec_mu, self.rec_sd = recs.mean(0), recs.std(0); self.rec_sd[self.rec_sd == 0] = 1
        rec_std = (recs - self.rec_mu) / self.rec_sd
        cfg = np.vstack([r.emb_std for r in self.races])
        emb_cr = np.hstack([cfg, rec_std])
        km = KMeans(n_clusters=K_SELECTOR, n_init=4, random_state=0).fit(emb_cr)
        self.sel_centroids = km.cluster_centers_
        grank = np.zeros(E.MAXC + 1)
        for r in self.races:
            if 1 <= r.winner_rank <= E.MAXC:
                grank[r.winner_rank] += 1
        grank /= grank.sum()
        self.selector = SituationClassAgent(self.sel_centroids, grank)
        self.selector.counts = np.zeros((K_SELECTOR, E.MAXC + 1)); self.selector.n = np.zeros(K_SELECTOR)
        for i, r in enumerate(self.races):
            cls = int(np.argmin(((self.sel_centroids - emb_cr[i]) ** 2).sum(1)))
            self.selector.learn(cls, r.winner_rank)

    @staticmethod
    def _rec_vec(fav_feats, others_r3):
        r3 = [v for v in others_r3 if v == v]
        fb = min(r3) if r3 else np.nan
        return list(fav_feats) + [fb, (np.mean(r3) if r3 else np.nan),
                                  (fav_feats[3] - fb) if (r3 and fav_feats[3] == fav_feats[3]) else np.nan]

    def _emb(self, card_entries, jcd, meta, upset):
        """生カード(fetch_race_card の entries) → env と同じ45次元布置。"""
        g = sorted(card_entries, key=lambda e: (e.get("trial_record") is None,
                                                e.get("trial_record") or 9.9))
        emb = np.zeros(E.EMB_DIM)
        cars = []
        for slot, e in enumerate(g[:E.MAXC]):
            base = slot * E.PER_SLOT
            emb[base + 0] = (e.get("handicap") or 0.0) / 100.0
            rp = e.get("rec_point")
            emb[base + 1] = ((rp - self.rp_mean) / self.rp_std) if rp is not None else 0.0
            emb[base + 2] = (e.get("ranking") or 200.0) / 300.0
            emb[base + 3] = 0.0                       # 旧ST(結果)=env と揃えて未使用
            emb[base + 4] = float(e.get("home") or 0)
            cars.append(e.get("car"))
        rb = E.MAXC * E.PER_SLOT
        emb[rb + 0] = (int(jcd) - 2) / 4.0
        emb[rb + 1] = (meta.get("distance") or 3000.0) / 5000.0
        emb[rb + 2] = _TRACK.get(meta.get("track_cond") or meta.get("track_condition"), 0.0) / 5.0
        emb[rb + 3] = float(upset)
        emb[rb + 4] = (meta.get("rno") or 6) / 12.0
        cars = cars + [None] * (E.MAXC - len(cars))
        return emb, cars

    def predict(self, card_entries, jcd, meta, win_odds: dict, upset=0.5):
        """→ dict(mem_probs, roughness, cars_by_rank, fav, high_rough)。"""
        emb, cars = self._emb(card_entries, jcd, meta, upset)
        emb_std = (emb - self.mu) / self.sd
        d, idx = self.nn.kneighbors(emb_std.reshape(1, -1))
        mp = _memory_car_probs(d[0], idx[0], self.ya, cars, win_odds)
        if not mp:
            return None
        fav = min(win_odds, key=lambda c: win_odds[c])
        rough = 1.0 - mp.get(fav, 0.0)
        return dict(mem_probs=mp, roughness=rough, cars_by_rank=cars, fav=fav,
                   high_rough=rough >= ROUGH_THR)

    def select_car(self, card_entries, jcd, meta, win_odds: dict, upset=0.5):
        """Stream B: config+records状況クラスの class-top 車(鋭化セレクタ)。→ (car, prob) or None。"""
        emb, cars = self._emb(card_entries, jcd, meta, upset)
        emb_std = (emb - self.mu) / self.sd
        fav = min(win_odds, key=lambda c: win_odds[c])
        pid = {e.get("car"): e.get("player_id") for e in card_entries}
        favp = self.player_latest.get(pid.get(fav), {})
        fav_feats = [favp.get(c, np.nan) for c in FAV_FEATS]
        others = [self.player_latest.get(pid.get(c), {}).get("p_r3_order", np.nan)
                  for c in win_odds if c != fav]
        rec = np.array(self._rec_vec(fav_feats, others), dtype=float)
        m = np.isnan(rec); rec[m] = self.rec_med[m]
        rec_std = (rec - self.rec_mu) / self.rec_sd
        emb_cr = np.concatenate([emb_std, rec_std])
        cls = int(np.argmin(((self.sel_centroids - emb_cr) ** 2).sum(1)))
        cp = self.selector.car_probs(cls, cars, win_odds)
        if not cp:
            return None
        car = max(cp, key=lambda c: cp[c])
        return car, cp[car]

    def trio_ev_bets(self, mem_probs, fav, trio_odds: dict, min_ev=1.0):
        """本命抜き・記憶EV>1 の3連複買い目 [(combo, odds, prob, ev)]。"""
        probs = _harville(mem_probs).get("trio", {})
        out = []
        for combo, p in probs.items():
            cs = set(int(x) for x in combo.split("-"))
            if fav in cs:
                continue
            o = trio_odds.get(combo)
            if o and o > 0 and p * o >= min_ev:
                out.append((combo, o, p, p * o))
        return sorted(out, key=lambda x: -x[2])
