"""タスクA：締切前オッズのライブ収集器(前向きに溜める唯一のleak-free判定土台)。

過去分は取れない(履歴API無し・確定オッズが上書き=Step0.5で確定)。ゆえにこれから開催
されるレースで、closeAt 直前に win/wide のオッズスナップショットを時刻付きで保存する。
溜まった期間だけを将来の金額judge(月次収支・破産率)に使う。

早撮りパス(試走ショック測量): verify_trial のレトロで「市場は試走を過小反応
(g=+0.12全年正)、バラけたレースほど織り込み損ね(g=0.29)」が出た。反応ダイナミクスの
精密化のため、試走前後のオッズを挟み撃ちで撮る:
  - pre_trial: 試走がまだ出とらん間、60秒毎に撮り直して最新1枚だけ残す
    (=試走開始の直前に最も近い1枚)。朝イチの薄いプールは撮らない。
  - post_trial: 試走タイムがAPIに現れた直後に1枚(pre がある場合のみ=挟み完成)。
  - close: 従来どおり締切前90秒窓で最大3枚。
  プール総額はWinticket APIに無い(2026-07確認)ため撮れない。代理として
  oddsUpdatedAt(オッズ最終更新時刻)を各スナップに保存する。

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
EARLY_HORIZON = 900       # 早撮りパスの監視開始(締切15分前から)
EARLY_REFRESH = 60        # pre_trial の撮り直し間隔(秒)
EARLY_BETS = ("win", "trio")   # 早撮りは単勝+3連複(受け皿券種)のみ。wideは close 窓だけ

PRECLOSE_SCHEMA = """
CREATE TABLE IF NOT EXISTS pre_close_odds (
    date TEXT NOT NULL, jcd TEXT NOT NULL, rno INTEGER NOT NULL,
    bet TEXT NOT NULL, combo TEXT NOT NULL,
    odds REAL, min_odds REAL, max_odds REAL,
    snapshot_at INTEGER NOT NULL,     -- 取得時刻(UNIX)
    close_at INTEGER,                 -- そのレースのcloseAt
    secs_to_close INTEGER,            -- close_at - snapshot_at(正=締切前)
    final INTEGER,                    -- finalOdds フラグ(締切前なら0のはず)
    phase TEXT DEFAULT 'close',       -- pre_trial / post_trial / close
    trial_seen INTEGER,               -- スナップ時点で試走タイムが出とったか(不明=NULL)
    odds_updated_at INTEGER,          -- APIのoddsUpdatedAt(プール活動の粗い代理)
    PRIMARY KEY (date,jcd,rno,bet,combo,snapshot_at)
);
"""

# 後付けカラム(既存DBは ALTER で補完)
_OPTIONAL_COLS = [("phase", "TEXT DEFAULT 'close'"),
                  ("trial_seen", "INTEGER"),
                  ("odds_updated_at", "INTEGER")]

_INSERT_COLS = ("date,jcd,rno,bet,combo,odds,min_odds,max_odds,"
                "snapshot_at,close_at,secs_to_close,final,phase,trial_seen,odds_updated_at")


def _migrate(conn) -> None:
    existing = {r[1] for r in conn.execute("PRAGMA table_info(pre_close_odds)")}
    for col, typ in _OPTIONAL_COLS:
        if existing and col not in existing:
            conn.execute(f"ALTER TABLE pre_close_odds ADD COLUMN {col} {typ}")
    conn.commit()


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


def _trial_done(cup, idx, rno):
    """レース詳細を直接GETして試走タイムが出とるか。True/False/None(取得失敗)。
    cup_detail はキャッシュされるため使えない(試走の出現が見えない)。"""
    d = winticket._get(f"{winticket.BASE}/cups/{cup}/schedules/{idx}/races/{rno}")
    if not d or "entries" not in d:
        return None
    ents = [e for e in d.get("entries", []) if not e.get("absent")]
    if not ents:
        return None
    n_trial = sum(1 for e in ents if e.get("trialRecord"))
    return n_trial >= max(1, len(ents) - 1)      # 欠車以外ほぼ全車に出たら試走済み


def _snapshot(conn, date, j, cup, idx, rno, close_at,
              phase="close", bets=("win", "wide", "trio"),
              trial_seen=None, replace_phase=False) -> int:
    """1レースのオッズスナップを保存。保存件数を返す。
    replace_phase=True なら同レース同phaseの旧スナップを消して最新1枚だけ残す。"""
    now = int(time.time())
    od = winticket._get(f"{winticket.BASE}/cups/{cup}/schedules/{idx}/races/{rno}/odds")
    if not od:
        return 0
    final = 1 if od.get("finalOdds") else 0
    upd = od.get("oddsUpdatedAt")
    s2c = close_at - now
    if replace_phase:
        conn.execute("DELETE FROM pre_close_odds WHERE date=? AND jcd=? AND rno=? AND phase=?",
                     (date, j, rno, phase))
    ph = ",".join(["?"] * 15)
    n = 0
    if "win" in bets:
        for e in od.get("win", []) or []:
            combo = "-".join(str(x) for x in e.get("key", []))
            o = e.get("odds")
            if combo and o:
                conn.execute(f"INSERT OR REPLACE INTO pre_close_odds ({_INSERT_COLS}) VALUES ({ph})",
                             (date, j, rno, "win", combo, float(o), None, None,
                              now, close_at, s2c, final, phase, trial_seen, upd))
                n += 1
    if "wide" in bets:
        for e in od.get("quinellaPlace", []) or []:
            combo = "-".join(str(x) for x in e.get("key", []))
            mn, mx = e.get("minOdds") or 0, e.get("maxOdds") or 0
            if combo and mn > 0:
                conn.execute(f"INSERT OR REPLACE INTO pre_close_odds ({_INSERT_COLS}) VALUES ({ph})",
                             (date, j, rno, "wide", combo, None, float(mn), float(mx),
                              now, close_at, s2c, final, phase, trial_seen, upd))
                n += 1
    # 3連複(trio)＝3-C受け皿が賭ける券種。締切前を撮らないと(b)が回せない
    if "trio" in bets:
        for e in od.get("trio", []) or []:
            combo = "-".join(str(x) for x in e.get("key", []))
            o = e.get("odds")
            if combo and o:
                conn.execute(f"INSERT OR REPLACE INTO pre_close_odds ({_INSERT_COLS}) VALUES ({ph})",
                             (date, j, rno, "trio", combo, float(o), None, None,
                              now, close_at, s2c, final, phase, trial_seen, upd))
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
    _migrate(conn)

    brain = None if args.dry else Brain()      # 予想の頭脳(締切直前に走らせる)
    snaps_taken: dict[tuple, int] = {}
    # 早撮りパスの状態: last_check=次の詳細GETまでの間引き / trial_state=試走出現の記憶
    early_last_check: dict[tuple, int] = {}
    trial_state: dict[tuple, bool] = {}
    pre_snapped: set[tuple] = set()
    post_snapped: set[tuple] = set()
    # 一発物のタスク。_upcoming の一時的な空(API瞬断・場間の空き・カード公開前)で
    # 永久終了すると2ヶ月の前向き収集が初日に死ぬ。空はカウントし、一度でもレースを
    # 見た後に連続 EMPTY_LIMIT 回続いたときだけ「本当に終わり」と判定する。
    EMPTY_LIMIT = 30                            # 30×POLL_SEC ≈ 10分の連続空で終了
    empty_streak = 0
    seen_any = False
    print(f"[live_odds] {date} を監視。SNAP_WINDOW={SNAP_WINDOW}s MAX_SNAPS={MAX_SNAPS} "
          f"早撮り={EARLY_HORIZON}s前〜 予想=ON")
    while True:
        up = _upcoming(date)
        if not up:
            if args.dry:
                print("  未締切レース無し")
                break
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

            # --- 早撮りパス(試走前後の挟み撃ち)。close窓の外側だけで動く ---
            if (SNAP_WINDOW < close - now <= EARLY_HORIZON
                    and not trial_state.get(key, False)
                    and now - early_last_check.get(key, 0) >= EARLY_REFRESH):
                early_last_check[key] = now
                done = _trial_done(cup, idx, rno)
                if done is False:
                    # 試走前: 最新1枚だけ残す(=試走開始直前に最も近い断面)
                    got = _snapshot(conn, date, j, cup, idx, rno, close,
                                    phase="pre_trial", bets=EARLY_BETS,
                                    trial_seen=0, replace_phase=True)
                    if got:
                        pre_snapped.add(key)
                        print(f"  [早撮pre] 場{j} {rno}R 締切{close-now}s前")
                elif done is True:
                    trial_state[key] = True
                    if key in pre_snapped and key not in post_snapped:
                        # 試走出現直後: 挟みの後側を1枚
                        got = _snapshot(conn, date, j, cup, idx, rno, close,
                                        phase="post_trial", bets=EARLY_BETS,
                                        trial_seen=1)
                        if got:
                            post_snapped.add(key)
                            print(f"  [早撮post] 場{j} {rno}R 締切{close-now}s前 (挟み完成)")

            # --- 従来の締切前スナップ ---
            if 0 < close - now <= SNAP_WINDOW and snaps_taken.get(key, 0) < MAX_SNAPS:
                got = _snapshot(conn, date, j, cup, idx, rno, close,
                                phase="close",
                                trial_seen=1 if trial_state.get(key) else None)
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
