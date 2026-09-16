"""外部オーケストレータ用アダプタ：本日の最良EVピックを1行JSONで返す。

ライブループ(src/loop/live_odds.py)が predictions テーブルに記録済みの
ベット(Stream B=単勝 / C=3連複)から ev 最大の1行を選ぶだけの passthrough。
Brain の再構築(710MB DB + KMeans で数分)は行わない。

  python daily_best.py pick [--date YYYYMMDD]        # 既定=今日(JST)
  python daily_best.py settle --date YYYYMMDD --jcd 02 --race 8 \
      --selection 3 [--bet 単勝|3連複]

stdout は常に JSON 1行のみ(ensure_ascii=False)。進捗は stderr へ。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

from src import storage, winticket
from src.venues import venue_name

JST = timezone(timedelta(hours=9))

# stream → 表示用ベット種 / predictions.bet_type
STREAM_LABEL = {"B": "単勝", "C": "3連複"}
LABEL_TO_BT = {"単勝": "win", "3連複": "trio"}

# 選定基準: EV >= MIN_EV の中で的中確率最大の1件
MIN_EV = 0.90
# 長穴の未較正EV対策(万車券の罠)。0で無効
MIN_PROB = 0.03
MAX_ODDS = 50.0


def _log(msg: str):
    print(msg, file=sys.stderr)


def _emit(obj: dict):
    print(json.dumps(obj, ensure_ascii=False))


def _today() -> str:
    return datetime.now(JST).strftime("%Y%m%d")


def _iso(date: str) -> str:
    return f"{date[:4]}-{date[4:6]}-{date[6:8]}"


def pick(date: str | None):
    date = date or _today()
    _log(f"pick: date={date} の predictions(Stream B/C, ev有り)から最大EVを検索")
    conn = storage.connect()
    row = conn.execute(
        "SELECT jcd, rno, stream, combo, odds_at_decision, mem_prob, ev "
        "FROM predictions WHERE date=? AND stream IN ('B','C') AND ev IS NOT NULL "
        "AND ev >= ? "
        "AND (? = 0 OR mem_prob >= ?) AND (? = 0 OR odds_at_decision <= ?) "
        "ORDER BY mem_prob DESC LIMIT 1",
        (date, MIN_EV, MIN_PROB, MIN_PROB, MAX_ODDS, MAX_ODDS)).fetchone()
    conn.close()
    if not row:
        _log("pick: 該当日の記録なし(ライブループ未稼働の可能性)")
        _emit({"pick": None, "reason": "no predictions logged today"})
        return
    jcd, rno, stream, combo, odds, mp, ev = row
    _log(f"pick: {jcd} {rno}R stream={stream} combo={combo} ev={ev:.3f}")
    _emit({
        "genre": "オートレース",
        "date": _iso(date),
        "jcd": jcd,
        "venue": venue_name(jcd),
        "race_no": int(rno),
        "bet_type": STREAM_LABEL.get(stream, stream),
        "selection": combo,
        "model_p": round(float(mp), 4) if mp is not None else None,
        "odds": float(odds) if odds is not None else None,
        "ev": round(float(ev), 4),
    })


def settle(date: str, jcd: str, rno: int, selection: str, bet: str | None):
    jcd = str(jcd).zfill(2)
    _log(f"settle: {date} {jcd} {rno}R selection={selection} bet={bet}")
    conn = storage.connect()
    q = ("SELECT bet_type, odds_at_decision, settled, hit, payoff FROM predictions "
         "WHERE date=? AND jcd=? AND rno=? AND combo=?")
    args = [date, jcd, rno, selection]
    if bet:
        q += " AND bet_type=?"
        args.append(LABEL_TO_BT.get(bet, bet))
    row = conn.execute(q, args).fetchone()
    conn.close()

    odds_at_decision = None
    bt = LABEL_TO_BT.get(bet) if bet else None
    if row:
        bt, odds_at_decision, settled, hit, payoff = row
        if settled:
            _log(f"settle: predictions で確定済 hit={hit} payoff={payoff}")
            _emit({"result": "win" if hit else "lose", "payout": int(round(payoff or 0))})
            return
        _log("settle: 記録はあるが未突合 → Winticket から結果取得")
    else:
        _log("settle: predictions に該当行なし → Winticket から結果取得")
    if bt is None:
        bt = "trio" if "-" in selection else "win"  # 車番形式から推定

    race = winticket.fetch_race(date, jcd, rno, with_odds=True)
    if not race or not race.get("results"):
        _log("settle: 結果未確定または取得不可")
        _emit({"result": "unknown", "payout": 0})
        return
    order = {int(c): int(o) for c, o in race["results"].items()}
    ordered = sorted(order, key=lambda c: order[c])
    winner = ordered[0]
    top3 = set(ordered[:3])
    _log(f"settle: 着順確定 1着={winner} top3={sorted(top3)}")
    if bt == "trio":
        hit = set(int(x) for x in selection.split("-")) == top3
    else:
        hit = int(selection) == winner
    # 払戻は reconcile と同じ規約: 判断時オッズ×100(単勝)。的中でオッズ不明なら0。
    payout = int(odds_at_decision * 100) if (hit and odds_at_decision) else 0
    _emit({"result": "win" if hit else "lose", "payout": payout,
           "note": "payout=odds_at_decision*100 (reconcile と同一規約・確定払戻ではない)"})


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="本日の最良EVピック(passthrough)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick", help="最大EVの1件をJSONで出力")
    p.add_argument("--date", default=None, help="YYYYMMDD (既定=今日JST)")
    s = sub.add_parser("settle", help="指定ベットの結果照会")
    s.add_argument("--date", required=True, help="YYYYMMDD")
    s.add_argument("--jcd", required=True)
    s.add_argument("--race", required=True, type=int)
    s.add_argument("--selection", required=True)
    s.add_argument("--bet", default=None, choices=["単勝", "3連複"])
    args = ap.parse_args()
    if args.cmd == "pick":
        pick(args.date)
    else:
        settle(args.date, args.jcd, args.race, args.selection, args.bet)


if __name__ == "__main__":
    main()
