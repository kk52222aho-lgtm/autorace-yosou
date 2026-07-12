"""SQLite による収集データの保存。

1車1行の非正規化テーブル `entries` に出走表(事前)＋着順(結果)＋レース条件を
まとめる。オートレースは1レース最大8車（開催により7車もある＝可変）。
Winticket 由来で全項目が揃うため、そのまま pandas に読み込める。

EV バックテスト用に全車の確定単勝オッズ(`win_odds`)と、代表賭式の払戻(`payouts`)
も保存する。オートレースは**単勝が発売される**ため、競艇型の単勝EV検証が可能。
"""
from __future__ import annotations

import os
import sqlite3

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "autorace.db")

# 出走表(事前情報)由来のカラム。fetch_race()['entries'] のキーと一致。
ENTRY_COLS = [
    "car", "player_id", "reg", "vehicle_id",
    "trial_record", "handicap", "starting_speed", "absent",
    "rec_point", "rec_class", "ranking", "last_rec_class", "last_ranking",
    "age", "weight", "height", "term", "pref", "name",
    # 取りこぼしていた事前信号
    "sunny_order", "rainy_order",   # 天候別の予想着順（Winticket提供の事前プライヤ）
    "home", "retrial",              # 地元(ロッカー場一致) / 再試走
    "blood", "constellation",       # “性格”枠：ノイズなら GBM が無視、PI に判定させる
]

# 後付けした任意カラム（既存DBは ALTER で補完）
_OPTIONAL_ENTRY_COLS = [
    ("height", "REAL"), ("sunny_order", "REAL"), ("rainy_order", "REAL"),
    ("home", "INTEGER"), ("retrial", "INTEGER"),
    ("blood", "TEXT"), ("constellation", "TEXT"),
]

# レース単位(事前に確定する条件)カラム。全車で同値だが行に持たせる。
META_COLS = [
    "race_class", "distance", "laps",
    "weather", "wind_dir", "wind_speed", "track_cond",
    "temperature", "track_temp", "humidity",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    date         TEXT NOT NULL,
    jcd          TEXT NOT NULL,
    rno          INTEGER NOT NULL,
    car          INTEGER NOT NULL,
    player_id    TEXT,
    reg          TEXT,
    vehicle_id   TEXT,
    trial_record REAL,
    handicap     REAL,
    starting_speed REAL,
    absent       INTEGER,
    rec_point    REAL,
    rec_class    REAL,
    ranking      REAL,
    last_rec_class REAL,
    last_ranking REAL,
    age          REAL,
    weight       REAL,
    height       REAL,
    term         REAL,
    pref         TEXT,
    name         TEXT,
    sunny_order  REAL,
    rainy_order  REAL,
    home         INTEGER,
    retrial      INTEGER,
    blood        TEXT,
    constellation TEXT,
    race_class   TEXT,
    distance     REAL,
    laps         REAL,
    weather      TEXT,
    wind_dir     REAL,
    wind_speed   REAL,
    track_cond   TEXT,
    temperature  REAL,
    track_temp   REAL,
    humidity     REAL,
    finish       INTEGER,
    win          INTEGER,
    record       REAL,
    start_timing REAL,
    PRIMARY KEY (date, jcd, rno, car)
);
"""

# 全車の確定単勝オッズ（EVバックテスト用）
WIN_ODDS_SCHEMA = """
CREATE TABLE IF NOT EXISTS win_odds (
    date TEXT NOT NULL, jcd TEXT NOT NULL, rno INTEGER NOT NULL,
    car INTEGER NOT NULL, odds REAL,
    PRIMARY KEY (date, jcd, rno, car)
);
"""

# 代表賭式の払戻（回収率の即算用）。payoff = odds*100 円。
PAYOUT_SCHEMA = """
CREATE TABLE IF NOT EXISTS payouts (
    date TEXT NOT NULL, jcd TEXT NOT NULL, rno INTEGER NOT NULL,
    win_car INTEGER, win_yen INTEGER,
    exacta_combo TEXT, exacta_yen INTEGER,
    trifecta_combo TEXT, trifecta_yen INTEGER,
    PRIMARY KEY (date, jcd, rno)
);
"""

# 全賭式の確定オッズ台帳（マルチプールEVバックテスト用）。
# bet ∈ win/show/exacta/quinella/quinella_place/trio/trifecta。combo は "1" や "1-2-3"。
ODDS_SCHEMA = """
CREATE TABLE IF NOT EXISTS odds (
    date TEXT NOT NULL, jcd TEXT NOT NULL, rno INTEGER NOT NULL,
    bet TEXT NOT NULL, combo TEXT NOT NULL, odds REAL,
    PRIMARY KEY (date, jcd, rno, bet, combo)
);
"""


def _migrate(conn) -> None:
    existing = {r[1] for r in conn.execute("PRAGMA table_info(entries)")}
    for col, typ in _OPTIONAL_ENTRY_COLS:
        if col not in existing:
            conn.execute(f"ALTER TABLE entries ADD COLUMN {col} {typ}")
    conn.commit()


def connect() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    conn.execute(WIN_ODDS_SCHEMA)
    conn.execute(PAYOUT_SCHEMA)
    conn.execute(ODDS_SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def has_race(conn, date: str, jcd: str, rno: int) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM entries WHERE date=? AND jcd=? AND rno=? AND finish IS NOT NULL LIMIT 1",
        (date, str(jcd).zfill(2), rno))
    return cur.fetchone() is not None


def save_race(conn, date: str, jcd: str, rno: int, race: dict) -> int:
    """fetch_race() の戻り値をそのまま保存。着順があれば finish/win を付与。"""
    jcd = str(jcd).zfill(2)
    meta = race.get("meta", {})
    results = race.get("results", {})
    record = race.get("record", {})
    st = race.get("start_timing", {})
    n = 0
    for e in race["entries"]:
        car = e["car"]
        f = results.get(car)
        win = 1 if f == 1 else (0 if f is not None else None)
        row = {**{c: e.get(c) for c in ENTRY_COLS},
               **{c: meta.get(c) for c in META_COLS},
               "date": date, "jcd": jcd, "rno": rno,
               "finish": f, "win": win,
               "record": record.get(car), "start_timing": st.get(car)}
        cols = list(row.keys())
        ph = ",".join(["?"] * len(cols))
        conn.execute(f"INSERT OR REPLACE INTO entries ({','.join(cols)}) VALUES ({ph})",
                     [row[c] for c in cols])
        n += 1
    all_odds = race.get("odds", {}) or {}
    # 単勝オッズ全車（既存 backtest 互換のため win_odds も維持）
    for car_s, o in (all_odds.get("win", {}) or {}).items():
        conn.execute("INSERT OR REPLACE INTO win_odds (date,jcd,rno,car,odds) VALUES (?,?,?,?,?)",
                     (date, jcd, rno, int(car_s), o))
    # 全賭式の台帳
    for bet, ladder in all_odds.items():
        for combo, o in (ladder or {}).items():
            conn.execute(
                "INSERT OR REPLACE INTO odds (date,jcd,rno,bet,combo,odds) VALUES (?,?,?,?,?,?)",
                (date, jcd, rno, bet, combo, o))
    # 払戻
    _save_payout(conn, date, jcd, rno, race)
    conn.commit()
    return n


def _save_payout(conn, date, jcd, rno, race) -> None:
    winners = race.get("winners", {}) or {}
    odds = race.get("odds", {}) or {}

    def payoff(bet, combo):
        o = (odds.get(bet, {}) or {}).get(combo)
        return int(round(o * 100)) if o else None

    # キーが存在しても空リストのことがある（当り目未確定/返還）→ or [None] で保護
    win_car = (winners.get("win") or [None])[0]
    ex = (winners.get("exacta") or [None])[0]
    tf = (winners.get("trifecta") or [None])[0]
    conn.execute(
        """INSERT OR REPLACE INTO payouts
           (date,jcd,rno,win_car,win_yen,exacta_combo,exacta_yen,trifecta_combo,trifecta_yen)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (date, jcd, rno,
         int(win_car) if win_car else None, payoff("win", win_car),
         ex, payoff("exacta", ex), tf, payoff("trifecta", tf)))
