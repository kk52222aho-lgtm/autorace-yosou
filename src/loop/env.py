"""レース環境：ブロックAを time-order で供給し、1レース分の状態(布置)を organize。

状態 = 番組表(半構造・試走順スロット) + その日ここまでの流れ(荒れ具合) +
       バンクロール残・月末までの残レース数。
オッズは決定に入れない(リーク回避)。payoff 算出用に確定単勝オッズだけ別途持つ。

布置の埋め込み(embedding)は「試走タイム昇順スロット r=1..MAXC」に各車を並べ、
スロットごとの [ハンデ, 得点z, ランク, ST, 地元] を連結。これで「速い車が重ハンデ/
本命がインに居る」等のレース形がベクトルの同じ位置に乗り、近傍が意味を持つ。
ラベル = 勝った車の試走順位(1..MAXC)。→ policy は「この布置では試走何番手が勝つか」を学ぶ。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import storage

MAXC = 8                       # 最大車数
PER_SLOT = 5                   # スロットごとの特徴数 [handicap, recpt_z, ranking, ST, home]
RACE_FEATS = 5                 # 場, 距離, 走路, 当日荒れ, レース番号
EMB_DIM = MAXC * PER_SLOT + RACE_FEATS

BLOCK_A = ("20230101", "20241231")
TRAIN_END_YM = "202408"        # 学習月は 2023-01..2024-08
HELDOUT_YMS = ["202409", "202410", "202411", "202412"]  # 未触judge(絶対に触らない)

_TRACK = {"良走路": 0.0, "湿走路": 1.0, "斑走路": 2.0, "風": 3.0, "オイル": 4.0, "荒": 5.0}


@dataclass
class Race:
    ym: str
    date: str
    key: tuple
    emb: np.ndarray            # 布置ベクトル(標準化前)
    cars_by_rank: list         # 試走順スロット→car番号(欠けは None)
    win_odds: dict             # car→確定単勝オッズ(payoff専用・リーク)
    winner_car: int
    winner_rank: int           # 勝者の試走順位(1..MAXC)
    emb_std: np.ndarray = None  # 標準化後の布置(train側で充填)


def _z(s: pd.Series) -> pd.Series:
    m, sd = s.mean(), s.std()
    return (s - m) / sd if sd and sd > 0 else s * 0.0


def load_block_a() -> list[Race]:
    """ブロックAの全レースを time-order で Race のリストに。"""
    conn = storage.connect()
    ent = pd.read_sql_query(
        "SELECT date,jcd,rno,car,trial_record,handicap,rec_point,ranking,"
        "start_timing,home,finish,distance,track_cond "
        "FROM entries WHERE finish IS NOT NULL AND date BETWEEN ? AND ?",
        conn, params=BLOCK_A)
    od = pd.read_sql_query(
        "SELECT date,jcd,rno,car,odds FROM win_odds WHERE odds>0 AND date BETWEEN ? AND ?",
        conn, params=BLOCK_A)
    conn.close()
    ent["jcd"] = ent["jcd"].astype(str).str.zfill(2)
    od["jcd"] = od["jcd"].astype(str).str.zfill(2)
    ent["recpt_z"] = _z(ent["rec_point"].fillna(ent["rec_point"].median()))
    odmap = {(d, j, r): dict(zip(g["car"].astype(int), g["odds"].astype(float)))
             for (d, j, r), g in od.groupby(["date", "jcd", "rno"])}

    # 当日の流れ: その日ここまでに消化したレースで「試走1番手が勝てなかった率」
    races_raw = []
    for (d, j, r), g in ent.groupby(["date", "jcd", "rno"]):
        g = g.sort_values("trial_record", na_position="last")
        cars = g["car"].astype(int).tolist()
        winner = int(g.loc[g["finish"] == 1, "car"].iloc[0]) if (g["finish"] == 1).any() else None
        if winner is None:
            continue
        wrank = cars.index(winner) + 1 if winner in cars else MAXC
        races_raw.append((d, j, r, g, cars, winner, wrank))
    # time-order: 日付→レース番号→場
    races_raw.sort(key=lambda x: (x[0], x[2], x[1]))

    # 当日荒れ(その日ここまでの upset 率)を逐次算出
    day_hist: dict[str, list[int]] = {}
    races: list[Race] = []
    for d, j, r, g, cars, winner, wrank in races_raw:
        past = day_hist.get(d, [])
        upset = np.mean(past) if past else 0.5   # まだ無ければ中立0.5
        # 布置ベクトル
        emb = np.zeros(EMB_DIM, dtype=float)
        for slot, (_, row) in enumerate(g.iterrows()):
            if slot >= MAXC:
                break
            base = slot * PER_SLOT
            emb[base + 0] = (row["handicap"] or 0.0) / 100.0
            emb[base + 1] = row["recpt_z"] if pd.notna(row["recpt_z"]) else 0.0
            emb[base + 2] = (row["ranking"] or 200.0) / 300.0
            # slot+3 は旧 start_timing(=結果ST。予想時に存在せず精度も下げるため未使用=0)
            emb[base + 3] = 0.0
            emb[base + 4] = float(row["home"] or 0)
        rb = MAXC * PER_SLOT
        emb[rb + 0] = (int(j) - 2) / 4.0
        emb[rb + 1] = (g["distance"].iloc[0] or 3000.0) / 5000.0
        emb[rb + 2] = _TRACK.get(g["track_cond"].iloc[0], 0.0) / 5.0
        emb[rb + 3] = float(upset)
        emb[rb + 4] = r / 12.0
        races.append(Race(
            ym=d[:6], date=d, key=(d, j, r), emb=emb,
            cars_by_rank=cars + [None] * (MAXC - len(cars)),
            win_odds=odmap.get((d, j, r), {}),
            winner_car=winner, winner_rank=wrank))
        day_hist.setdefault(d, []).append(1 if winner != cars[0] else 0)
    return races


def standardizer(races: list[Race]):
    """ブロックA全布置の列標準化パラメータ(mean/std)。"""
    X = np.vstack([r.emb for r in races])
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return mu, sd


def split_months(races: list[Race]):
    """学習月リスト(time-order) と 未触月リスト。"""
    train = [r for r in races if r.ym <= TRAIN_END_YM]
    heldout = [r for r in races if r.ym in HELDOUT_YMS]
    train_month_order = sorted({r.ym for r in train})
    return train, heldout, train_month_order
