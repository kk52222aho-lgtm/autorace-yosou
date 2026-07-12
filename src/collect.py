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
