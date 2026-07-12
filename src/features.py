"""特徴量エンジニアリング。

学習・予測で同じ関数を使い特徴量のズレを防ぐ。オートレースの肝：

- 想定タイム = 試走タイム + ハンデ換算(10m≒0.01秒)。ただしハンデは設計上これを
  ほぼ均等化するため、**絶対値でなくレース内の相対順位・残差**に信号が乗る。
  → レース内 z-score / rank を主特徴に据える。
- rec_point(得点相当) が選手の地力。ranking(総合順位)は低いほど強い。
- 天候・走路(良走路/湿走路) は試走タイムの効き方を変えるためカテゴリで持つ。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .history import HISTORY_COLS
from .venues import track_length

# ハンデ換算係数：10m ≒ 0.01秒 → 1m = 0.001秒
SOTEI_K = 0.001

CATEGORICAL = ["venue_code", "track_cond_code", "weather_code",
               "blood_code", "constellation_code"]

# 走路条件・天候を数値コード化（欠損は -1）
TRACK_MAP = {"良走路": 0, "湿走路": 1, "": -1}
WEATHER_MAP = {"晴": 0, "曇": 1, "雨": 2, "雪": 3, "": -1}
# “性格”枠：ノイズなら GBM が無視。PI で判定させるため入れる
BLOOD_MAP = {"A": 0, "B": 1, "O": 2, "AB": 3}
_SIGNS = ["牡羊座", "牡牛座", "双子座", "蟹座", "獅子座", "乙女座",
          "天秤座", "蠍座", "射手座", "山羊座", "水瓶座", "魚座"]
CONSTELLATION_MAP = {s: i for i, s in enumerate(_SIGNS)}

FEATURES = [
    "venue_code",        # 場コード（場ごとのクセを学習）
    "track_length",      # 走路周長(伊勢崎400/他500)
    "track_cond_code",   # 良走路/湿走路
    "weather_code",
    "wind_speed",
    "car",               # 車番
    "handicap",          # ハンデ(m)
    "trial_record",      # 試走タイム(秒)
    "sotei_time",        # 想定タイム = 試走 + ハンデ換算
    "starting_speed",
    "rec_point",         # 得点相当（地力）
    "rec_class",         # 級(数値)
    "ranking",           # 総合順位(低いほど強い)
    "last_ranking",
    "age", "weight", "height", "term",
    "sunny_order",       # 晴天時の予想着順（事前プライヤ。小さいほど上位想定）
    "rainy_order",       # 雨天時の予想着順
    "home",              # 地元(ロッカー場一致)
    "retrial",
    "blood_code", "constellation_code",   # “性格”枠（PIで判定）
    # ↓ レース内相対（本命の核）。z=フィールド平均からの偏差、rk=昇順順位
    "trial_z", "trial_rk",
    "sotei_z", "sotei_rk",
    "recpoint_z", "recpoint_rk",
    "handicap_rk",
] + HISTORY_COLS   # 直近フォーム・マシン履歴（history.add_history で付与）

CAT_INDEX = [FEATURES.index(c) for c in CATEGORICAL]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

_RACE_KEYS = ["date", "jcd", "rno"]


def _zrank(df: pd.DataFrame, src: str, z: str, rk: str, ascending: bool) -> None:
    """レース内で src を z-score 化(z)・順位化(rk)。ascending=True は小さいほど1位。"""
    if not set(_RACE_KEYS).issubset(df.columns):
        df[z] = df[src] - df[src].mean()
        df[rk] = df[src].rank(ascending=ascending)
        return
    g = df.groupby(_RACE_KEYS)[src]
    mean, std = g.transform("mean"), g.transform("std").replace(0, np.nan)
    df[z] = (df[src] - mean) / std
    df[rk] = g.rank(ascending=ascending)


def build_frame(rows, impute: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()

    for col in ("trial_record", "handicap", "rec_point", "wind_speed"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sotei_time"] = df["trial_record"] + df["handicap"] * SOTEI_K

    if "jcd" in df.columns:
        df["venue_code"] = pd.to_numeric(df["jcd"], errors="coerce")
        df["track_length"] = df["jcd"].map(
            lambda j: track_length(str(int(j)).zfill(2)) if pd.notna(j) else np.nan)
    df["track_cond_code"] = df.get("track_cond", pd.Series(index=df.index)).map(
        lambda v: TRACK_MAP.get(str(v).strip(), -1))
    df["weather_code"] = df.get("weather", pd.Series(index=df.index)).map(
        lambda v: WEATHER_MAP.get(str(v).strip(), -1))
    df["blood_code"] = df.get("blood", pd.Series(index=df.index)).map(
        lambda v: BLOOD_MAP.get(str(v).strip(), -1))
    df["constellation_code"] = df.get("constellation", pd.Series(index=df.index)).map(
        lambda v: CONSTELLATION_MAP.get(str(v).strip(), -1))

    # レース内相対（試走・想定は小さいほど速い＝1位、得点は大きいほど上位＝1位）
    _zrank(df, "trial_record", "trial_z", "trial_rk", ascending=True)
    _zrank(df, "sotei_time", "sotei_z", "sotei_rk", ascending=True)
    _zrank(df, "rec_point", "recpoint_z", "recpoint_rk", ascending=False)
    # ハンデは順位のみ（前ハンデ=小さいほど前=1位）
    if set(_RACE_KEYS).issubset(df.columns):
        df["handicap_rk"] = df.groupby(_RACE_KEYS)["handicap"].rank(ascending=True)
    else:
        df["handicap_rk"] = df["handicap"].rank(ascending=True)

    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if impute:
        df[NUMERIC] = df[NUMERIC].fillna(df[NUMERIC].median(numeric_only=True)).fillna(0)
    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype(int)
    return df


def matrix(rows) -> np.ndarray:
    return build_frame(rows)[FEATURES].to_numpy(dtype=float)
