"""試走過小反応(g)の開催区分別スライス。

verify_trial の条件付きロジットを、cup_map.csv(ミッドナイト等の開催区分)で
切って再推定。②仮説: casual マネー最厚のミッドナイトで g が増幅されとるか。
併せて区分別の想定タイム1位×オッズ帯フラットROI（円が見える形）。

  python -m src.loop.verify_trial_slot
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import storage
from .verify_trial import (HANDI_SEC_PER_M, MAXC, N_BOOT, RNG,
                           _fit_g, fit_clogit)


def load_races_with_slot():
    cmap = pd.read_csv("data/cup_map.csv", dtype={"date": str, "jcd": str})
    slot_of = {(r.date, r.jcd): r.slot for r in cmap.itertuples()}

    conn = storage.connect()
    rows = conn.execute("""
        SELECT e.date, e.jcd, e.rno, e.car, e.trial_record, e.handicap,
               e.win, e.track_cond, w.odds
        FROM entries e JOIN win_odds w
          ON w.date=e.date AND w.jcd=e.jcd AND w.rno=e.rno AND w.car=e.car
        WHERE e.absent=0 AND e.finish IS NOT NULL
          AND e.trial_record > 3.0 AND e.trial_record < 5.0
          AND w.odds > 0
        ORDER BY e.date, e.jcd, e.rno, e.car""").fetchall()
    conn.close()

    races: dict[tuple, list] = {}
    for r in rows:
        races.setdefault((r[0], r[1], r[2]), []).append(r)

    LI, GD, MK, WIN = [], [], [], []
    meta = {"slot": [], "odds": [], "grank": []}
    for key, cars in races.items():
        if len(cars) < 5 or sum(c[6] == 1 for c in cars) != 1:
            continue
        yoso = np.array([c[4] + c[5] * HANDI_SEC_PER_M for c in cars])
        if yoso.std() < 1e-9:
            continue
        odds = np.array([c[8] for c in cars], dtype=float)
        imp = (1.0 / odds) / (1.0 / odds).sum()
        good = -(yoso - yoso.mean()) / yoso.std()
        n = len(cars)
        li = np.full(MAXC, 0.0); gd = np.full(MAXC, 0.0)
        mk = np.zeros(MAXC); mk[:n] = 1.0
        li[:n] = np.log(imp); gd[:n] = good
        LI.append(li); GD.append(gd); MK.append(mk)
        WIN.append(int(np.argmax([c[6] for c in cars])))
        meta["slot"].append(slot_of.get((key[0], str(key[1]).zfill(2)), "unknown"))
        meta["odds"].append(odds.tolist() + [np.nan] * (MAXC - n))
        meta["grank"].append((np.argsort(np.argsort(-good[:n])) + 1).tolist()
                             + [0] * (MAXC - n))
    return (np.array(LI), np.array(GD), np.array(MK), np.array(WIN),
            {k: np.array(v) for k, v in meta.items()})


def boot_ci_idx(idx, LI, GD, MK, WIN, n_boot=N_BOOT):
    gs = np.empty(n_boot)
    for b in range(n_boot):
        sub = idx[RNG.randint(0, len(idx), len(idx))]
        gs[b] = _fit_g(sub, LI, GD, MK, WIN)[1]
    return np.percentile(gs, [5, 95]), (gs > 0).mean()


def main():
    print("ロード中...")
    LI, GD, MK, WIN, meta = load_races_with_slot()
    print(f"対象 {len(WIN):,}R\n")

    print("== 開催区分別: g(試走残差係数) ブートCI付き ==")
    order = ["day", "early", "after5", "night", "midnight", "over_midnight"]
    counts = pd.Series(meta["slot"]).value_counts()
    for slot in order:
        if counts.get(slot, 0) < 300:
            continue
        idx = np.where(meta["slot"] == slot)[0]
        _, g = _fit_g(idx, LI, GD, MK, WIN)
        (lo, hi), ppos = boot_ci_idx(idx, LI, GD, MK, WIN)
        flag = " <-" if lo > 0 else ""
        print(f"  {slot:>14s}: g={g:+.4f}  CI[{lo:+.4f},{hi:+.4f}]"
              f"  P(g>0)={ppos:.1%}  ({len(idx):,}R){flag}")

    # 夜間まとめ(midnight+over_midnight) vs 昼系(day+early+after5)
    print("\n== 統合: 夜間(midnight系) vs 昼系 ==")
    night_set = {"midnight", "over_midnight"}
    for lab, sel in [("midnight系", np.isin(meta["slot"], list(night_set))),
                     ("それ以外", ~np.isin(meta["slot"], list(night_set)))]:
        idx = np.where(sel)[0]
        _, g = _fit_g(idx, LI, GD, MK, WIN)
        (lo, hi), ppos = boot_ci_idx(idx, LI, GD, MK, WIN)
        print(f"  {lab:>10s}: g={g:+.4f}  CI[{lo:+.4f},{hi:+.4f}]"
              f"  P(g>0)={ppos:.1%}  ({len(idx):,}R)")

    # 円が見える形: 区分×想定タイム1位のオッズ帯フラットROI
    print("\n== 区分別: 想定タイム1位のフラットROI(実オッズ, オッズ帯別) ==")
    odds = meta["odds"]; grank = meta["grank"]
    R = len(WIN)
    won = np.zeros_like(grank)
    won[np.arange(R), WIN] = 1
    bands = [("<3倍", 0, 3), ("3-10倍", 3, 10), (">10倍", 10, 9999), ("全体", 0, 9999)]
    print(f"  {'区分':>14s}" + "".join(f"{b[0]:>16s}" for b in bands))
    for slot in order:
        if counts.get(slot, 0) < 300:
            continue
        sel_slot = (meta["slot"] == slot)[:, None] & (grank == 1)
        row = f"  {slot:>14s}"
        for _, lo_, hi_ in bands:
            sel = sel_slot & (odds >= lo_) & (odds < hi_) & ~np.isnan(odds)
            n = int(sel.sum())
            if n < 50:
                row += f"{'—':>16s}"
                continue
            ret = (odds[sel] * won[sel]).sum() / n * 100
            row += f"{ret:>9.1f}% n={n:<5d}"[:16].rjust(16)
        print(row)

    print("\n ※確定オッズ基準。多重比較6断面につき単発CI逸脱は割引いて読むこと。")


if __name__ == "__main__":
    main()
