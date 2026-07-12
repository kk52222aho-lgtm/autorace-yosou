"""ライブ予想ループ：締切前オッズの瞬間に判断を出し、3ストリームで immutable に記録。
確定後に自動突合してストリーム別の累積ROIを出す。前向き=リーク原理的にゼロ=割引不要の本物。

Stream A: 全レースの較正確率 vs 市場含意(ベースライン記録・本命参照)
Stream B: 記憶top(較正確率最大)の単勝(RL器78%到達の単勝方策の代表)
Stream C: 3-C受け皿(高荒れ×本命抜き3連複EV>1) ← +21ptが本物か幻かの直接判定

  python -m src.loop.live_predict            # 本日の未締切レースを予想・記録
  python -m src.loop.live_predict --reconcile  # 確定分を突合してROI更新
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .. import storage, winticket
from .brain import Brain, ROUGH_THR

PRED_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    date TEXT NOT NULL, jcd TEXT NOT NULL, rno INTEGER NOT NULL,
    stream TEXT NOT NULL, bet_type TEXT, combo TEXT NOT NULL,
    odds_at_decision REAL, mem_prob REAL, implied REAL, ev REAL,
    roughness REAL, logged_at INTEGER, secs_to_close INTEGER,
    settled INTEGER DEFAULT 0, hit INTEGER, payoff REAL,
    PRIMARY KEY (date, jcd, rno, stream, combo)
);
"""


def _odds(cup, idx, rno):
    d = winticket._get(f"{winticket.BASE}/cups/{cup}/schedules/{idx}/races/{rno}/odds")
    if not d:
        return None, None, None
    win = {int(e["key"][0]): float(e["odds"]) for e in d.get("win", []) or [] if e.get("odds")}
    trio = {"-".join(str(x) for x in e["key"]): float(e["odds"])
            for e in d.get("trio", []) or [] if e.get("odds")}
    return win, trio, bool(d.get("finalOdds"))


def log_streams(conn, brain, date, j, rno, meta, card_entries, win, trio, close_at, now):
    """締切直前(試走+締切前オッズ有り)の1レースを3ストリームで immutable 記録。
    戻り値: dict(roughness, high_rough, mem_top, n_c) / 既記録や試走無なら None。"""
    if conn.execute("SELECT 1 FROM predictions WHERE date=? AND jcd=? AND rno=? LIMIT 1",
                    (date, j, rno)).fetchone():
        return None
    if not any((e.get("trial_record") or 0) > 0 for e in card_entries):
        return None                       # 試走未計測=まだ予想しない
    p = brain.predict(card_entries, j, meta, win)
    if not p:
        return None
    mp = p["mem_probs"]; s = sum(1 / o for o in win.values() if o > 0)
    imp = {c: (1 / win[c]) / s for c in win if win[c] > 0}
    s2c = close_at - now
    fav = p["fav"]
    # Stream B = 鋭化セレクタ(config+records状況クラス)。失敗時はkNN mem_topへ縮退
    sel = brain.select_car(card_entries, j, meta, win)
    mt, mt_p = sel if sel else (max(mp, key=lambda c: mp[c]), mp[max(mp, key=lambda c: mp[c])])
    rows = [("A", "ref", str(fav), win[fav], mp.get(fav), imp.get(fav), None),
            ("B", "win", str(mt), win[mt], mt_p, imp.get(mt), mt_p * win[mt])]
    if p["high_rough"] and trio:
        for combo, o, pr, ev in brain.trio_ev_bets(mp, fav, trio):
            rows.append(("C", "trio", combo, o, pr, None, ev))
    for stream, bt, combo, o, prob, impd, ev in rows:
        conn.execute(
            "INSERT OR IGNORE INTO predictions "
            "(date,jcd,rno,stream,bet_type,combo,odds_at_decision,mem_prob,implied,ev,"
            "roughness,logged_at,secs_to_close) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (date, j, rno, stream, bt, combo, o, prob, impd, ev, p["roughness"], now, s2c))
    conn.commit()
    return dict(roughness=p["roughness"], high_rough=p["high_rough"], mem_top=mt,
                n_c=sum(1 for r in rows if r[0] == "C"))


def predict_today(date=None):
    date = date or time.strftime("%Y%m%d")
    brain = Brain()
    conn = storage.connect()
    conn.execute(PRED_SCHEMA); conn.commit()
    now = int(time.time())
    logged = 0
    for j in winticket.held_venues(date):
        cup = winticket.resolve_cup(date, j)
        det = winticket.cup_detail(cup) if cup else None
        sch = winticket._schedule(det, date) if det else None
        if not sch:
            continue
        sid = sch.get("id"); idx = sch["index"]
        for rc in det.get("races", []):
            if rc.get("scheduleId") != sid:
                continue
            close = rc.get("closeAt")
            if not close or close <= now:
                continue                              # 未締切のみ
            rno = rc["number"]
            done = conn.execute("SELECT 1 FROM predictions WHERE date=? AND jcd=? AND rno=? LIMIT 1",
                                (date, j, rno)).fetchone()
            if done:
                continue                              # 既記録は上書きしない(immutable)
            win, trio, final = _odds(cup, idx, rno)
            if not win:
                continue                              # オッズ未公開はスキップ
            card = winticket.fetch_race_card(date, j, rno)
            if not card:
                continue
            meta = dict(card["meta"]); meta["rno"] = rno
            p = brain.predict(card["entries"], j, meta, win)
            if not p:
                continue
            mp = p["mem_probs"]; s = sum(1 / o for o in win.values() if o > 0)
            imp = {c: (1 / win[c]) / s for c in win if win[c] > 0}
            s2c = close - now
            rows = []
            # Stream A: 本命の記憶確率 vs 市場含意(ベースライン)
            fav = p["fav"]
            rows.append(("A", "ref", str(fav), win[fav], mp.get(fav), imp.get(fav), None))
            # Stream B: 記憶top の単勝
            mt = max(mp, key=lambda c: mp[c])
            rows.append(("B", "win", str(mt), win[mt], mp[mt], imp.get(mt), mp[mt] * win[mt]))
            # Stream C: 高荒れ×本命抜き3連複EV>1
            if p["high_rough"] and trio:
                for combo, o, pr, ev in brain.trio_ev_bets(mp, fav, trio):
                    rows.append(("C", "trio", combo, o, pr, None, ev))
            for stream, bt, combo, o, prob, impd, ev in rows:
                conn.execute(
                    "INSERT OR IGNORE INTO predictions "
                    "(date,jcd,rno,stream,bet_type,combo,odds_at_decision,mem_prob,implied,ev,"
                    "roughness,logged_at,secs_to_close) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (date, j, rno, stream, bt, combo, o, prob, impd, ev, p["roughness"], now, s2c))
            logged += 1
            print(f"  記録 場{j} {rno}R (締切{s2c//60}分前) roughness={p['roughness']:.2f}"
                  f"{' ★高荒れ受け皿ON' if p['high_rough'] else ''} B=車{mt}({win[mt]}) "
                  f"C={sum(1 for r in rows if r[0]=='C')}点")
    conn.commit()
    _show_today(conn, date)
    conn.close()


def _show_today(conn, date):
    print(f"\n=== {date} 判断ログ(締切前・immutable) ===")
    for stream in ["A", "B", "C"]:
        n = conn.execute("SELECT COUNT(*) FROM predictions WHERE date=? AND stream=?",
                         (date, stream)).fetchone()[0]
        r = conn.execute("SELECT COUNT(DISTINCT jcd||rno) FROM predictions WHERE date=? AND stream=?",
                        (date, stream)).fetchone()[0]
        label = {"A": "較正vs市場(基準)", "B": "記憶top単勝", "C": "受け皿(高荒れ×本命抜き3連複)"}[stream]
        print(f"  Stream {stream} {label}: {r}レース / {n}記録")


RESULTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS race_results (
    date TEXT NOT NULL, jcd TEXT NOT NULL, rno INTEGER NOT NULL,
    winner INTEGER, winner_odds REAL, second INTEGER, third INTEGER,
    PRIMARY KEY (date, jcd, rno)
);
"""


def reconcile(date=None):
    """確定分を突合: 的中・払戻を記録し、結果(1着車等)を保存、ストリーム別累積ROIを出す。"""
    conn = storage.connect(); conn.execute(PRED_SCHEMA); conn.execute(RESULTS_SCHEMA)
    rows = conn.execute("SELECT DISTINCT date,jcd,rno FROM predictions WHERE settled=0").fetchall()
    for d, j, rno in rows:
        race = winticket.fetch_race(d, j, rno, with_odds=True)
        if not race or not race.get("results"):
            continue
        order = {int(c): int(o) for c, o in race["results"].items()}
        if not order:
            continue
        ordered = sorted(order, key=lambda c: order[c])
        winner = ordered[0]
        top3 = set(ordered[:3])
        wodds = (race.get("odds", {}).get("win", {}) or {}).get(str(winner))
        conn.execute("INSERT OR REPLACE INTO race_results VALUES (?,?,?,?,?,?,?)",
                     (d, j, rno, winner, float(wodds) if wodds else None,
                      ordered[1] if len(ordered) > 1 else None,
                      ordered[2] if len(ordered) > 2 else None))
        for pid, stream, bt, combo, o in conn.execute(
                "SELECT rowid,stream,bet_type,combo,odds_at_decision FROM predictions "
                "WHERE date=? AND jcd=? AND rno=? AND settled=0", (d, j, rno)).fetchall():
            hit = 0
            if bt == "win":
                hit = 1 if int(combo) == winner else 0
            elif bt == "trio":
                hit = 1 if set(int(x) for x in combo.split("-")) == top3 else 0
            elif bt == "ref":
                hit = 1 if int(combo) == winner else 0
            payoff = (o * 100) if (hit and o) else 0.0
            conn.execute("UPDATE predictions SET settled=1,hit=?,payoff=? WHERE rowid=?",
                        (hit, payoff, pid))
    conn.commit()
    print("=== ストリーム別 前向き累積ROI(締切前判断・割引不要) ===")
    for stream in ["B", "C"]:
        r = conn.execute(
            "SELECT COUNT(*),SUM(hit),SUM(payoff) FROM predictions WHERE stream=? AND settled=1 AND bet_type IN ('win','trio')",
            (stream,)).fetchone()
        n, h, pay = r[0] or 0, r[1] or 0, r[2] or 0.0
        if n:
            roi = pay / (n * 100) * 100
            print(f"  Stream {stream}: {n}ベット 的中{h}({h/n*100:.0f}%) 累積ROI {roi:.0f}% "
                  f"(前向きN目標300)")
        else:
            print(f"  Stream {stream}: 確定ベットまだ無し")
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--reconcile", action="store_true")
    args = ap.parse_args()
    if args.reconcile:
        reconcile(args.date)
    else:
        predict_today(args.date)


if __name__ == "__main__":
    main()
