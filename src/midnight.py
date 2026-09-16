"""ミッドナイト断面スキャン：開催区分(昼/ナイター/ミッドナイト等)別に単勝効率を割る。

DBに開催区分は無いが、Winticket の月別カレンダー(cups?date=)の cup 名に
「ミッドナイト」「オーバーミッドナイト」「ナイトレース」「アーリー」等が入るため、
(date,jcd) → cup 名でマッピングを復元できる（cupId = 初日+場コード）。

②嗜好の歪み仮説：ミッドナイトは場外・ネット民 casual マネー比率が最大の断面。
cherry-pick せず全区分を CI 付きで列挙する。
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import requests

from . import storage

RNG = np.random.RandomState(0)
BASE = "https://api.winticket.jp/v1/autorace"
HEADERS = {"User-Agent": "Mozilla/5.0 (autorace-yosou research; personal use)",
           "Accept": "application/json"}
SLEEP_SEC = 0.4


def _classify(name: str) -> str:
    if "オーバーミッドナイト" in name:
        return "over_midnight"
    if "ミッドナイト" in name:
        return "midnight"
    if "ナイトレース" in name or "ナイター" in name:
        return "night"
    if "アーリー" in name:
        return "early"
    if "アフター５" in name or "アフター5" in name:
        return "after5"
    return "day"


def build_cup_map(force: bool = False) -> pd.DataFrame:
    """月別カレンダーを全期間なめて (date,jcd) → cup名/区分 を data/cup_map.csv に保存。"""
    import os
    path = "data/cup_map.csv"
    if os.path.exists(path) and not force:
        return pd.read_csv(path, dtype={"date": str, "jcd": str})
    conn = storage.connect()
    lo, hi = conn.execute("SELECT min(date), max(date) FROM entries").fetchone()
    conn.close()
    months = pd.period_range(lo[:6], hi[:6], freq="M").strftime("%Y%m").tolist()
    sess = requests.Session()
    sess.headers.update(HEADERS)
    rows = []
    for ym in months:
        try:
            r = sess.get(f"{BASE}/cups?date={ym}01", timeout=20).json()
        except requests.RequestException:
            time.sleep(2)
            continue
        cups = {c["id"]: c for k in ("monthlyCups", "weeklyCups")
                for c in (r.get(k) or []) if c.get("id")}
        for c in cups.values():
            name = c.get("name", "")
            for d in pd.date_range(c["startDate"], c["endDate"]).strftime("%Y%m%d"):
                rows.append({"date": d, "jcd": str(c["venueId"]).zfill(2),
                             "cup_id": c["id"], "cup_name": name,
                             "grade": c.get("grade"),
                             "slot": _classify(name)})
        print(f"{ym}: {len(cups)} cups", file=sys.stderr)
        time.sleep(SLEEP_SEC)
    df = pd.DataFrame(rows).drop_duplicates(["date", "jcd"])
    df.to_csv(path, index=False)
    return df


def _load() -> pd.DataFrame:
    conn = storage.connect()
    ent = pd.read_sql_query(
        "SELECT date,jcd,rno,car,win,race_class FROM entries WHERE win IS NOT NULL", conn)
    od = pd.read_sql_query(
        "SELECT date,jcd,rno,car,odds FROM win_odds WHERE odds IS NOT NULL AND odds>0", conn)
    conn.close()
    for d in (ent, od):
        d["jcd"] = d["jcd"].astype(str).str.zfill(2)
        d["date"] = d["date"].astype(str)
    df = ent.merge(od, on=["date", "jcd", "rno", "car"], how="inner")
    cmap = build_cup_map()
    df = df.merge(cmap[["date", "jcd", "slot", "grade"]], on=["date", "jcd"], how="left")
    df["slot"] = df["slot"].fillna("unknown")
    df["fav_rank"] = df.groupby(["date", "jcd", "rno"])["odds"].rank(method="first")
    return df


def _roi_ci(odds, win, B=8000):
    payoff = odds * win
    n = len(payoff)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    roi = payoff.mean() * 100
    means = np.array([payoff[RNG.randint(0, n, n)].mean() for _ in range(B)]) * 100
    p5, p95 = np.percentile(means, [5, 95])
    return roi, p5, p95, n


def main():
    df = _load()
    races = df.groupby(["date", "jcd", "rno"]).ngroups
    print(f"データ: {races:,}レース")
    print("\n=== 開催区分別レース数 ===")
    rc = df.drop_duplicates(["date", "jcd", "rno"])["slot"].value_counts()
    print(rc.to_string())

    for rank in (1, 2, 3):
        print(f"\n=== 開催区分別：市場{rank}番人気の単勝フラットROI ===")
        print(f"{'区分':>16s}{'n':>8s}{'勝率':>8s}{'ROI':>7s}{'CI(5-95%)':>15s}")
        fav = df[df["fav_rank"] == rank]
        for val in rc.index:
            sub = fav[fav["slot"] == val]
            if len(sub) < 150:
                continue
            roi, p5, p95, n = _roi_ci(sub["odds"].to_numpy(), sub["win"].to_numpy())
            flag = " <-" if p5 >= 100 else ""
            print(f"{str(val):>16s}{n:8,d}{sub['win'].mean()*100:7.1f}%"
                  f"{roi:6.0f}%{f'[{p5:.0f},{p95:.0f}]':>15s}{flag}")

    # オッズ帯別（区分×人気帯の歪みの居所）
    print("\n=== 区分×オッズ帯：全出走の単勝フラットROI ===")
    df["band"] = pd.cut(df["odds"], [1, 2, 3, 5, 10, 30, 1000],
                        labels=["1-2", "2-3", "3-5", "5-10", "10-30", "30+"])
    print(f"{'区分':>16s}{'帯':>6s}{'n':>9s}{'ROI':>7s}{'CI(5-95%)':>15s}")
    for val in rc.index:
        for band in ["1-2", "2-3", "3-5", "5-10", "10-30", "30+"]:
            sub = df[(df["slot"] == val) & (df["band"] == band)]
            if len(sub) < 300:
                continue
            roi, p5, p95, n = _roi_ci(sub["odds"].to_numpy(), sub["win"].to_numpy())
            flag = " <-" if p5 >= 100 else ""
            print(f"{str(val):>16s}{band:>6s}{n:9,d}{roi:6.0f}%"
                  f"{f'[{p5:.0f},{p95:.0f}]':>15s}{flag}")


if __name__ == "__main__":
    main()
