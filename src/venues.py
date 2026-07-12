"""オートレース場コード ↔ 場名、走路周長。

現存5場のみ（船橋・大井は廃止）。Winticket の venueId は1桁だが、
本DBの jcd は競輪版に合わせて2桁ゼロ詰め（"02".."06"）で統一する。

走路周長は重要特徴量：伊勢崎のみ 400m、他4場は 500m。周長が短いと
コーナー比率が上がりハンデ/試走タイムの効き方が変わる。
"""
from __future__ import annotations

# jcd(2桁) → 場名
VENUES: dict[str, str] = {
    "02": "川口",
    "03": "伊勢崎",
    "04": "浜松",
    "05": "飯塚",
    "06": "山陽",
}

# Winticket venueId(int/str) → jcd(2桁)。実質ゼロ詰めだが明示しておく。
WT_VENUE_TO_JCD: dict[str, str] = {str(i): f"{i:02d}" for i in range(2, 7)}

# 走路周長(m)。伊勢崎だけ 400m。
TRACK_LENGTH: dict[str, int] = {
    "02": 500,  # 川口
    "03": 400,  # 伊勢崎
    "04": 500,  # 浜松
    "05": 500,  # 飯塚
    "06": 500,  # 山陽
}


def venue_name(jcd: str) -> str:
    return VENUES.get(str(jcd).zfill(2), str(jcd))


def track_length(jcd: str) -> int:
    return TRACK_LENGTH.get(str(jcd).zfill(2), 500)


def all_jcd() -> list[str]:
    return list(VENUES.keys())
