"""(b)カウントダウン：締切前オッズが何レース溜まったら 3-C受け皿をリーク無しで
再評価できるか。(b)を「いつか」でなく「あとNレース」にして飛ばせなくする。

高荒れレース(記憶がroughness上位1/3と読むレース)にpre_close trioが要る。現状は
roughness算出用の番組表が未取得なので、高荒れ≒総取得の1/3(閾値の構成上)で概算する。
"""
from __future__ import annotations

import sqlite3

from .verify3c_discount import ROUGH_Q

DB = "data/autorace.db"
TARGET_FIRST = 300        # 初回の向き読み(粗い)
TARGET_CONCLUSIVE = 1500  # 決着(CIが締まる)


def main():
    c = sqlite3.connect(DB)
    total = c.execute("SELECT COUNT(DISTINCT date||jcd||rno) FROM pre_close_odds").fetchone()[0]
    nights = c.execute("SELECT COUNT(DISTINCT date) FROM pre_close_odds").fetchone()[0]
    has_trio = c.execute(
        "SELECT COUNT(DISTINCT date||jcd||rno) FROM pre_close_odds WHERE bet='wide' OR bet='win'").fetchone()[0]
    c.close()

    est_high = total * (1 - ROUGH_Q)     # 高荒れ概算(1/3)
    per_night = total / nights if nights else 0
    high_per_night = per_night * (1 - ROUGH_Q)

    print("=== (b)カウントダウン: 締切前オッズで3-C受け皿をリーク無し再評価 ===")
    print(f"  取得済み: {total}レース / {nights}晩 (1晩あたり{per_night:.0f}レース)")
    print(f"  高荒れ概算(≈1/3): {est_high:.0f}レース")
    print(f"  ※現状の取得は単勝+ワイド。3連複pre_closeの収集も要拡張(次の器更新で追加)。\n")
    for label, tgt in [("初回の向き読み", TARGET_FIRST), ("決着(CI締まる)", TARGET_CONCLUSIVE)]:
        remain = max(0, tgt - est_high)
        nights_left = remain / high_per_night if high_per_night > 0 else float("inf")
        bar = int(min(1.0, est_high / tgt) * 20)
        print(f"  {label:>12s} 目標{tgt}高荒れ: [{'#'*bar}{'.'*(20-bar)}] "
              f"あと{remain:.0f}レース ≈ {nights_left:.0f}晩")
    print("\n  毎晩10:00に autorace_preclose が自動起動(登録済)。溜まるほどこの数字が減る。")
    print("  高荒れが初回目標に届いたら verify3c を pre_close_odds で回してリーク無し絶対値を出す。")


if __name__ == "__main__":
    main()
