"""タスクA：締切前オッズのライブ収集器(前向きに溜める唯一のleak-free判定土台)。

過去分は取れない(履歴API無し・確定オッズが上書き=Step0.5で確定)。ゆえにこれから開催
されるレースで、closeAt 直前に win/wide のオッズスナップショットを時刻付きで保存する。
溜まった期間だけを将来の金額judge(月次収支・破産率)に使う。

運用: レース開催日の日中に起動して放置。closeAt-now が SNAP_WINDOW 秒以内の未締切
レースを見つけたらスナップショットを1枚保存(締切までに複数枚まで)。全レース締切で終了。
番組は当日朝に公開されるため、毎朝スケジュール起動する想定。

  python -m src.loop.live_odds            # 本日を監視して放置
  python -m src.loop.live_odds --dry      # 取得せず対象レースだけ表示
"""
from __future__ import annotations

import argparse
import time

from .. import storage
from .. import winticket
from .brain import Brain
from .live_predict import PRED_SCHEMA, log_streams

SNAP_WINDOW = 90          # closeAt の何秒前からスナップを狙うか
MAX_SNAPS = 3             # 1レース最大何枚(締切に近づくほど価値大)
POLL_SEC = 20            # 監視ループの間隔

PRECLOSE_SCHEMA = """
CREATE TABLE IF NOT EXISTS pre_close_odds (
    date TEXT NOT NULL, jcd TEXT NOT NULL, rno INTEGER NOT NULL,
    bet TEXT NOT NULL, combo TEXT NOT NULL,
    odds REAL, min_odds REAL, max_odds REAL,
    snapshot_at INTEGER NOT NULL,     -- 取得時刻(UNIX)
    close_at INTEGER,                 -- そのレースのcloseAt
    secs_to_close INTEGER,            -- close_at - snapshot_at(正=締切前)
    final INTEGER,                    -- finalOdds フラグ(締切前なら0のはず)
    PRIMARY KEY (date,jcd,rno,bet,combo,snapshot_at)
);
"""


def _today() -> str:
    return time.strftime("%Y%m%d")


def _upcoming(date: str):
    """(jcd, cup, idx, rno, closeAt) の未締切レース一覧。"""
    now = int(time.time())
    out = []
    for j in winticket.held_venues(date):
        cup = winticket.resolve_cup(date, j)
        if not cup:
            continue
        det = winticket.cup_detail(cup)
        sch = winticket._schedule(det, date) if det else None
        if not sch:
            continue
        sid = sch.get("id")
        for r in det.get("races", []):
            if r.get("scheduleId") == sid and r.get("closeAt") and r["closeAt"] > now:
                out.append((j, cup, sch["index"], r["number"], r["closeAt"]))
    return out


def _snapshot(conn, date, j, cup, idx, rno, close_at) -> int:
    """1レースのwin/wideスナップを保存。保存件数を返す。"""
    now = int(time.time())
    od = winticket._get(f"{winticket.BASE}/cups/{cup}/schedules/{idx}/races/{rno}/odds")
    if not od:
        return 0
    final = 1 if od.get("finalOdds") else 0
    s2c = close_at - now
    n = 0
    for e in od.get("win", []) or []:
        combo = "-".join(str(x) for x in e.get("key", []))
        o = e.get("odds")
        if combo and o:
            conn.execute(
                "INSERT OR REPLACE INTO pre_close_odds VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (date, j, rno, "win", combo, float(o), None, None, now, close_at, s2c, final))
            n += 1
    for e in od.get("quinellaPlace", []) or []:
        combo = "-".join(str(x) for x in e.get("key", []))
        mn, mx = e.get("minOdds") or 0, e.get("maxOdds") or 0
        if combo and mn > 0:
            conn.execute(
                "INSERT OR REPLACE INTO pre_close_odds VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (date, j, rno, "wide", combo, None, float(mn), float(mx), now, close_at, s2c, final))
            n += 1
    # 3連複(trio)＝3-C受け皿が賭ける券種。締切前を撮らないと(b)が回せない
    for e in od.get("trio", []) or []:
        combo = "-".join(str(x) for x in e.get("key", []))
        o = e.get("odds")
        if combo and o:
            conn.execute(
                "INSERT OR REPLACE INTO pre_close_odds VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (date, j, rno, "trio", combo, float(o), None, None, now, close_at, s2c, final))
            n += 1
    conn.commit()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="監視日(YYYYMMDD)。既定=本日")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    date = args.date or _today()

    conn = storage.connect()
    conn.execute(PRECLOSE_SCHEMA)
    conn.execute(PRED_SCHEMA)
    conn.commit()

    brain = None if args.dry else Brain()      # 予想の頭脳(締切直前に走らせる)
    snaps_taken: dict[tuple, int] = {}
    # 一発物のタスク。_upcoming の一時的な空(API瞬断・場間の空き・カード公開前)で
    # 永久終了すると2ヶ月の前向き収集が初日に死ぬ。空はカウントし、一度でもレースを
    # 見た後に連続 EMPTY_LIMIT 回続いたときだけ「本当に終わり」と判定する。
    EMPTY_LIMIT = 30                            # 30×POLL_SEC ≈ 10分の連続空で終了
    empty_streak = 0
    seen_any = False
    print(f"[live_odds] {date} を監視。SNAP_WINDOW={SNAP_WINDOW}s MAX_SNAPS={MAX_SNAPS} 予想=ON")
    while True:
        up = _upcoming(date)
        if not up:
            empty_streak += 1
            if seen_any and empty_streak >= EMPTY_LIMIT:
                print(f"[live_odds] 未締切レース無し{empty_streak}回連続→終了")
                break
            time.sleep(POLL_SEC)
            continue
        empty_streak = 0
        seen_any = True
        now = int(time.time())
        if args.dry:
            for j, cup, idx, rno, close in up[:20]:
                print(f"  場{j} {rno}R closeまで{close-now:5d}s")
            break
        for j, cup, idx, rno, close in up:
            key = (date, j, rno)
            if 0 < close - now <= SNAP_WINDOW and snaps_taken.get(key, 0) < MAX_SNAPS:
                got = _snapshot(conn, date, j, cup, idx, rno, close)
                if got:
                    snaps_taken[key] = snaps_taken.get(key, 0) + 1
                    print(f"  [snap] 場{j} {rno}R 締切{close-now}s前 {got}目 "
                          f"(通算{snaps_taken[key]}枚)")
                # 同じ瞬間に予想を記録(試走+締切前オッズが揃う締切直前)
                od = winticket._get(f"{winticket.BASE}/cups/{cup}/schedules/{idx}/races/{rno}/odds")
                win = {int(e["key"][0]): float(e["odds"]) for e in (od or {}).get("win", []) or [] if e.get("odds")}
                trio = {"-".join(str(x) for x in e["key"]): float(e["odds"])
                        for e in (od or {}).get("trio", []) or [] if e.get("odds")}
                card = winticket.fetch_race_card(date, j, rno) if win else None
                if card:
                    meta = dict(card["meta"]); meta["rno"] = rno
                    res = log_streams(conn, brain, date, j, rno, meta, card["entries"], win, trio, close, now)
                    if res:
                        print(f"  [予想] 場{j} {rno}R roughness={res['roughness']:.2f}"
                              f"{' ★受け皿ON' if res['high_rough'] else ''} B=車{res['mem_top']} C={res['n_c']}点")
        time.sleep(POLL_SEC)
    conn.close()


if __name__ == "__main__":
    main()
