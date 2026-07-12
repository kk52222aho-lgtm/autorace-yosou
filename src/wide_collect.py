"""ワイド(quinella_place)の minOdds/maxOdds を永続収集（専用/oddsエンドポイント）。

race詳細の odds フィールドは0でレンジ返しのため未収集だった。ここでは専用 /odds の
quinellaPlace の minOdds/maxOdds を wide_odds テーブルに保存する。realized(的中ペア)は
entries.finish の上位3着から後で導出できるので、オッズだけ集めれば良い。

  python -m src.wide_collect --n 3000    # 直近3000レース
"""
from __future__ import annotations

import argparse

from . import storage, winticket

WIDE_SCHEMA = """
CREATE TABLE IF NOT EXISTS wide_odds (
    date TEXT NOT NULL, jcd TEXT NOT NULL, rno INTEGER NOT NULL,
    combo TEXT NOT NULL, min_odds REAL, max_odds REAL,
    PRIMARY KEY (date, jcd, rno, combo)
);
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    conn = storage.connect()
    conn.execute(WIDE_SCHEMA)
    conn.commit()
    # 未収集の直近レースを対象（wide_odds に無いもの優先）
    rows = conn.execute(
        "SELECT DISTINCT e.date,e.jcd,e.rno FROM entries e "
        "WHERE e.win IS NOT NULL AND NOT EXISTS "
        "(SELECT 1 FROM wide_odds w WHERE w.date=e.date AND w.jcd=e.jcd AND w.rno=e.rno) "
        "ORDER BY e.date DESC, e.jcd, e.rno LIMIT ?", (args.n,)).fetchall()
    print(f"対象 {len(rows)} レース（wide_odds 未収集の直近分）")

    got = skipped = 0
    for i, (d, j, rno) in enumerate(rows):
        cup = winticket.resolve_cup(d, j)
        det = winticket.cup_detail(cup) if cup else None
        sch = winticket._schedule(det, d) if det else None
        if not sch:
            skipped += 1
            continue
        od = winticket._get(
            f"{winticket.BASE}/cups/{cup}/schedules/{sch['index']}/races/{rno}/odds")
        if not od:
            skipped += 1
            continue
        n = 0
        for e in od.get("quinellaPlace", []) or []:
            key = "-".join(str(x) for x in e.get("key", []))
            mn, mx = e.get("minOdds") or 0, e.get("maxOdds") or 0
            if key and mn > 0:
                conn.execute(
                    "INSERT OR REPLACE INTO wide_odds VALUES (?,?,?,?,?,?)",
                    (d, j, rno, key, float(mn), float(mx) if mx else float(mn)))
                n += 1
        if n:
            got += 1
        else:
            skipped += 1
        if (i + 1) % 100 == 0:
            conn.commit()
            print(f"  ...{i+1}/{len(rows)} 取得{got} 空{skipped}")
    conn.commit()
    conn.close()
    print(f"完了: 取得{got} / 空{skipped}")


if __name__ == "__main__":
    main()
