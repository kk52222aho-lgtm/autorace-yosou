"""データ収集 CLI。

Winticket API を日付→開催場→レースと巡回し、事前情報＋着順＋オッズを DB 保存。
過去日も開催カレンダー(monthlyCups/weeklyCups)から場を発見できる。

例:
  python -m src.collect --start 20260601 --end 20260630
  python -m src.collect --start 20260601 --end 20260630 --venues 02 03
"""
from __future__ import annotations

import argparse
import datetime as dt

from . import storage, winticket as wt
from .venues import venue_name


FIRST_RESULT_HOUR = 11   # これより前に当日の結果を聞いても在り得ん


def _dates(start: str, end: str):
    d0 = dt.datetime.strptime(start, "%Y%m%d").date()
    d1 = dt.datetime.strptime(end, "%Y%m%d").date()
    d = d0
    while d <= d1:
        yield d.strftime("%Y%m%d")
        d += dt.timedelta(days=1)


def collect(start: str, end: str, venues: list[str] | None = None,
            skip_existing: bool = True) -> None:
    conn = storage.connect()
    total_races = total_rows = 0
    for date in _dates(start, end):
        # 2026-09-30 に競艇から移植。**まだ走っとらんレースの結果は在るわけない。**
        # この常駐は 09:40 に回るが、実測(63日)で**その日の最初の締切は51日が10時台**
        # (残りは14〜21時)。つまり観測した全ての日で**最初のレースが締まる前**に
        # 走っとって、ログに `20260929 伊勢崎(03): 0/8R` と出て1日20本ほど空打ちしとった。
        # 窓は -14日 なので翌朝の回で必ず拾う。午後に回した時は同日分も取る余地を残す。
        # → [[insight_window_includes_the_unhappened]]
        _now = dt.datetime.now()
        if date > _now.strftime("%Y%m%d"):
            continue
        if date == _now.strftime("%Y%m%d") and _now.hour < FIRST_RESULT_HOUR:
            print(f"  {date}: まだ {FIRST_RESULT_HOUR}時前。"
                  f"その日の結果は在り得んので飛ばす(最初の締切は10時台が51/63日)")
            continue
        held = venues or wt.held_venues(date)
        if not held:
            continue
        for jcd in held:
            n = wt.race_count(date, jcd)
            if not n:
                continue
            got = 0
            for rno in range(1, n + 1):
                if skip_existing and storage.has_race(conn, date, jcd, rno):
                    got += 1
                    continue
                try:
                    race = wt.fetch_race(date, jcd, rno)
                    if not race or not race.get("results"):
                        continue  # 未確定/欠損はスキップ（後日再収集）
                    total_rows += storage.save_race(conn, date, jcd, rno, race)
                    total_races += 1
                    got += 1
                except Exception as e:  # 1レースの失敗で全収集を殺さない
                    print(f"  !! {date} {jcd} {rno}R skip: {type(e).__name__}: {e}")
            print(f"  {date} {venue_name(jcd)}({jcd}): {got}/{n}R")
    conn.close()
    print(f"\n完了: {total_races} レース新規保存 / {total_rows} 行")


def main() -> None:
    ap = argparse.ArgumentParser(description="オートレース データ収集 (Winticket)")
    ap.add_argument("--start", required=True, help="開始日 YYYYMMDD")
    ap.add_argument("--end", required=True, help="終了日 YYYYMMDD")
    ap.add_argument("--venues", nargs="*", help="場コード限定 (例: 02 03)")
    ap.add_argument("--no-skip", action="store_true", help="既存も再取得")
    args = ap.parse_args()
    venues = [v.zfill(2) for v in args.venues] if args.venues else None
    collect(args.start, args.end, venues, skip_existing=not args.no_skip)


if __name__ == "__main__":
    main()
