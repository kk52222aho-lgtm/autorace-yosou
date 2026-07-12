"""予想屋(tipster)の暗黙知を構造特徴として収集：地の文コメント + 推奨買い目。

Winticket race詳細の predictions[] は過去レースでも返る。予想屋の勝率・推奨3連単の
車・地の文コメントが入る。NLP無しで「市場本命が専門家の推奨から外れているか(=専門家が
本命の飛びを読む)」を作れる。これが config+オッズを超える市場外信号か次段でテスト。

expert_picks(date,jcd,rno, winrate, n_picks, head_car, cars_csv, comment_len)
  head_car : 推奨3連単で最頻の1着車(専門家の本命)
  cars_csv : 推奨に登場する全車(カンマ区切り)
  comment_len : 地の文の長さ(後でLLM特徴化する布石)

  python -m src.loop.expert_collect --n 2500
"""
from __future__ import annotations

import argparse
from collections import Counter

from .. import storage, winticket
from . import env as E

EXPERT_SCHEMA = """
CREATE TABLE IF NOT EXISTS expert_picks (
    date TEXT NOT NULL, jcd TEXT NOT NULL, rno INTEGER NOT NULL,
    winrate REAL, n_picks INTEGER, head_car INTEGER,
    cars_csv TEXT, comment_len INTEGER, comment TEXT,
    PRIMARY KEY (date,jcd,rno)
);
"""


def _parse_picks(pred: dict):
    ids = (pred.get("sunny") or {}).get("trifectaOddsIds") or []
    heads, allcars = [], set()
    for i in ids:
        combo = i.split(":")[-1].split(".")
        if len(combo) == 3:
            heads.append(combo[0])
            allcars.update(combo)
    head_car = int(Counter(heads).most_common(1)[0][0]) if heads else None
    return len(ids), head_car, ",".join(sorted(allcars, key=int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    args = ap.parse_args()

    conn = storage.connect()
    conn.execute(EXPERT_SCHEMA)
    # 既存DBに comment 列が無ければ足す
    if "comment" not in {r[1] for r in conn.execute("PRAGMA table_info(expert_picks)")}:
        conn.execute("ALTER TABLE expert_picks ADD COLUMN comment TEXT")
    conn.commit()
    # コメント本文がまだ無いレースを対象(本文の再収集)
    rows = conn.execute(
        "SELECT DISTINCT e.date,e.jcd,e.rno FROM entries e "
        "WHERE e.finish IS NOT NULL AND e.date BETWEEN ? AND ? AND NOT EXISTS "
        "(SELECT 1 FROM expert_picks x WHERE x.date=e.date AND x.jcd=e.jcd AND x.rno=e.rno "
        " AND x.comment IS NOT NULL) "
        "ORDER BY e.date DESC, e.jcd, e.rno LIMIT ?",
        (*E.BLOCK_A, args.n)).fetchall()
    print(f"対象 {len(rows)} レース(ブロックA・expert_picks未収集の新しい順)")

    got = empty = 0
    for i, (d, j, rno) in enumerate(rows):
        cup = winticket.resolve_cup(d, j)
        det = winticket.cup_detail(cup) if cup else None
        sch = winticket._schedule(det, d) if det else None
        if not sch:
            empty += 1
            continue
        data = winticket._get(
            f"{winticket.BASE}/cups/{cup}/schedules/{sch['index']}/races/{rno}")
        preds = (data or {}).get("predictions") or []
        if not preds:
            empty += 1
            continue
        p = preds[0]
        n_picks, head_car, cars_csv = _parse_picks(p)
        cm = p.get("comment") or ""
        conn.execute(
            "INSERT OR REPLACE INTO expert_picks VALUES (?,?,?,?,?,?,?,?,?)",
            (d, j, rno, p.get("winningRate"), n_picks, head_car, cars_csv,
             len(cm), cm))
        got += 1
        if (i + 1) % 100 == 0:
            conn.commit()
            print(f"  ...{i+1}/{len(rows)} 取得{got} 空{empty}")
    conn.commit()
    conn.close()
    print(f"完了: 取得{got} / 空{empty}")


if __name__ == "__main__":
    main()
