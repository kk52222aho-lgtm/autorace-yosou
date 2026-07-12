"""Winticket 公開API クライアント（オートレースの事前情報＋オッズ）。

競輪版では netkeirin をスクレイピングして出走表を組み立てたが、オートレースは
Winticket(api.winticket.jp) の公開JSON APIが**出走表・試走タイム・ハンデ・
選手級・天候/走路条件・全買い目オッズ**を一括で返すため、これ一本で事前情報が
揃う（DOMスクレイピング不要）。着順・払戻だけは別途 results.py で取得する。

エンドポイント（keirin と同型、パスの keirin → autorace だけ違う）:
  開催カレンダー:  /v1/autorace/cups?date=YYYYMMDD          → month.cups[]
  開催詳細:        /v1/autorace/cups/{cupId}                 → schedules/races/entries/players
  オッズ:          /v1/autorace/cups/{cupId}/schedules/{index}/races/{rno}/odds
                   → win(単勝)/exacta(2連単)/trifecta(3連単)/trio/quinella/quinellaPlace

ID体系: cupId = 開催初日(YYYYMMDD) + 場コード(2桁, 川口02..山陽06)。
schedules[].index が開催日目(1..)。entries[].raceId == races[].id で突合。
"""
from __future__ import annotations

import time

import requests

BASE = "https://api.winticket.jp/v1/autorace"
HEADERS = {"User-Agent": "Mozilla/5.0 (autorace-yosou research; personal use)",
           "Accept": "application/json"}
SLEEP_SEC = 0.4

_session = requests.Session()
_session.headers.update(HEADERS)

_cal_cache: dict[str, list[dict]] = {}
_cup_cache: dict[str, dict] = {}


def _get(url: str, retries: int = 3):
    for i in range(retries):
        try:
            r = _session.get(url, timeout=20)
            if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(SLEEP_SEC * (i + 1))
    return None


def _month_cups(date: str) -> list[dict]:
    """その月/週の開催 cup 一覧（キャッシュ）。API は monthlyCups と weeklyCups の
    2系統で返すため両者を union する（id で重複排除）。"""
    ym = date[:6]
    if ym not in _cal_cache:
        d = _get(f"{BASE}/cups?date={date}") or {}
        merged: dict[str, dict] = {}
        for key in ("monthlyCups", "weeklyCups"):
            for cp in d.get(key, []) or []:
                if cp.get("id"):
                    merged[cp["id"]] = cp
        _cal_cache[ym] = list(merged.values())
        time.sleep(SLEEP_SEC)
    return _cal_cache[ym]


def _prev_month_first(date: str) -> str:
    y, m = int(date[:4]), int(date[4:6])
    y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    return f"{y:04d}{m:02d}01"


def resolve_cup(date: str, jcd: str) -> str | None:
    """(開催日, 場コード2桁) から cupId を解決。月またぎ開催に備え前月も探す。"""
    jcd = str(jcd).zfill(2)
    for anchor in (date, _prev_month_first(date)):
        for cp in _month_cups(anchor):
            vid = str(cp.get("venueId", "")).zfill(2)
            if (vid == jcd
                    and cp.get("startDate", "") <= date <= cp.get("endDate", "")):
                return cp["id"]
    return None


def cup_detail(cup_id: str) -> dict | None:
    if cup_id not in _cup_cache:
        d = _get(f"{BASE}/cups/{cup_id}")
        _cup_cache[cup_id] = d or {}
        time.sleep(SLEEP_SEC)
    return _cup_cache[cup_id] or None


def _schedule(det: dict, date: str) -> dict | None:
    for s in det.get("schedules", []):
        if s.get("date") == date:
            return s
    return None


def held_venues(date: str) -> list[str]:
    """その日に開催のある場コード(2桁)一覧。過去日も可（カレンダーAPI由来）。"""
    out = []
    for cp in _month_cups(date):
        if cp.get("startDate", "") <= date <= cp.get("endDate", ""):
            out.append(str(cp.get("venueId", "")).zfill(2))
    return sorted(set(out))


def race_count(date: str, jcd: str) -> int:
    """その日その場のレース数（開催日目のスケジュールから）。無ければ0。"""
    cup_id = resolve_cup(date, jcd)
    if not cup_id:
        return 0
    det = cup_detail(cup_id)
    sch = _schedule(det, date) if det else None
    if not sch:
        return 0
    sid = sch.get("id")
    return sum(1 for r in det.get("races", []) if r.get("scheduleId") == sid)


def _parse_entry(e: dict, p: dict, jcd: str = "") -> dict:
    """entries[] の1件 + players[] の該当選手 → 事前情報の正規化 dict。"""
    locker = str(p.get("lockerGroundVenueId", "")).zfill(2)
    return {
        "car": e.get("number"),
        "player_id": e.get("playerId"),
        "reg": p.get("registrationNumber"),
        "vehicle_id": e.get("vehicleId"),
        "trial_record": e.get("trialRecord"),      # 試走タイム(100m秒)
        "handicap": e.get("handicap"),             # ハンデ(m)
        "starting_speed": e.get("startingSpeed"),
        "absent": 1 if e.get("absent") else 0,
        "retrial": 1 if e.get("retrial") else 0,
        # 天候別の予想着順（Winticket提供の事前プライヤ。小さいほど上位想定）
        "sunny_order": e.get("sunnyOrder"),
        "rainy_order": e.get("rainyOrder"),
        # 強さ指標: recommendationPoint(得点相当) / class(S..) / ranking(総合順位:低いほど強い)
        "rec_point": p.get("recommendationPoint"),
        "rec_class": p.get("recommendationClass"),
        "ranking": p.get("ranking"),
        "last_rec_class": p.get("lastRecommendationClass"),
        "last_ranking": p.get("lastRanking"),
        "age": p.get("age"),
        "weight": p.get("weight"),
        "height": p.get("height"),
        "term": p.get("term"),
        "pref": p.get("prefecture"),
        "name": p.get("name"),
        "home": 1 if (jcd and locker == str(jcd).zfill(2)) else 0,  # 地元(ロッカー場一致)
        "blood": p.get("blood"),
        "constellation": p.get("constellation"),
    }


def _race_meta(race: dict) -> dict:
    return {
        "race_class": race.get("class"),
        "distance": race.get("distance"),
        "laps": race.get("laps"),
        "weather": race.get("weather"),
        "wind_dir": race.get("windDirection"),
        "wind_speed": race.get("windSpeed"),
        "track_cond": race.get("trackCondition"),
        "temperature": race.get("temperature"),
        "track_temp": race.get("trackTemperature"),
        "humidity": race.get("humidity"),
        "status": race.get("status"),
    }


# 賭式キー → Winticket JSONのキー
_BET_KEYS = {
    "win": "win", "show": "show", "exacta": "exacta", "trifecta": "trifecta",
    "trio": "trio", "quinella": "quinella", "quinella_place": "quinellaPlace",
}


def _parse_odds(d: dict, bet: str) -> dict[str, float]:
    key = _BET_KEYS.get(bet)
    out: dict[str, float] = {}
    for e in d.get(key, []) or []:
        k, o = e.get("key"), e.get("odds")
        if k and o:
            out["-".join(str(x) for x in k)] = float(o)
    return out


def fetch_race(date: str, jcd: str, rno: int,
               with_odds: bool = True) -> dict | None:
    """1レースの完全データを race 詳細エンドポイントから取得。

    このエンドポイント一つに **出走表(entries)・選手・着順(results)・確定オッズ・
    払戻(winningOddsIds)** が全部入る。事前(live)でも results が空なだけで使える。

    戻り値: {
      "meta": {...},
      "entries": [{車単位・事前情報}...],
      "results": {car: order},          # 着順（未確定なら空 {}）
      "record":  {car: 走破タイム(秒)},   # totalRecord
      "start_timing": {car: ST},
      "odds": {"win": {"1":o,...}, "exacta": {...}, "trifecta": {...}},  # with_odds時
      "winners": {"win": ["4"], "exacta": ["4-7"], "trifecta": [...]},
    }
    取得不可なら None。
    """
    cup_id = resolve_cup(date, jcd)
    if not cup_id:
        return None
    det = cup_detail(cup_id)
    sch = _schedule(det, date) if det else None
    if not sch:
        return None
    idx = sch.get("index")
    d = _get(f"{BASE}/cups/{cup_id}/schedules/{idx}/races/{rno}")
    if not d or "race" not in d:
        return None
    players = {p.get("id"): p for p in d.get("players", [])}
    ents = [_parse_entry(e, players.get(e.get("playerId"), {}), jcd)
            for e in d.get("entries", [])]
    ents.sort(key=lambda x: (x["car"] is None, x["car"]))
    # results[] を車番でひく（playerId → car）
    pid_to_car = {e.get("playerId"): e.get("number") for e in d.get("entries", [])}
    results, record, st = {}, {}, {}
    for r in d.get("results", []) or []:
        car = pid_to_car.get(r.get("playerId"))
        if car is None or not r.get("order"):
            continue
        results[car] = r.get("order")
        record[car] = r.get("totalRecord")
        st[car] = r.get("startTiming")
    out = {"meta": _race_meta(d["race"]), "entries": ents,
           "results": results, "record": record, "start_timing": st}
    if with_odds:
        out["odds"] = {b: _parse_odds(d, b) for b in _BET_KEYS}
        def _win(field):
            return [i.split(":")[-1].replace(".", "-") for i in d.get(field, []) or []]
        out["winners"] = {
            "win": _win("winWinningOddsIds"),
            "show": _win("showWinningOddsIds"),
            "exacta": _win("exactaWinningOddsIds"),
            "quinella": _win("quinellaWinningOddsIds"),
            "quinella_place": _win("quinellaPlaceWinningOddsIds"),
            "trio": _win("trioWinningOddsIds"),
            "trifecta": _win("trifectaWinningOddsIds"),
        }
    return out


def fetch_race_card(date: str, jcd: str, rno: int) -> dict | None:
    """事前情報のみ（cup 詳細のバルクから）。ライブ予測用。結果は含まない。"""
    r = fetch_race(date, jcd, rno, with_odds=False)
    if not r:
        return None
    return {"meta": r["meta"], "entries": r["entries"]}


def fetch_odds(date: str, jcd: str, rno: int,
               bet: str = "win") -> dict[str, float] | None:
    r = fetch_race(date, jcd, rno, with_odds=True)
    if not r:
        return None
    return r["odds"].get(bet) or None
