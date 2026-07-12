"""単勝 EV バックテスト（Winticket確定オッズ × モデル確率）。

競艇で効いた「割安(EV>1)だけ買う」を単勝で検証する。オートレースは単勝が
常時発売されるため、競輪(2車単に妥協)より素直に競艇型の検証ができる。

  1) 時系列ウォークフォワードで各車の1着確率を予測（リークなし）
  2) レース内で正規化し P(win) を得る
  3) EV = P(win) × 単勝オッズ。EV≥閾値の車だけ購入
  4) 実結果(1着)と突合して回収率を算出

例:
  python -m src.backtest_ev                # best: レース毎に最良EVを1点
  python -m src.backtest_ev --mode value   # value: EV≥閾値を全点
  python -m src.backtest_ev --min-ev 1.1 --max-odds 30
"""
from __future__ import annotations

import argparse

import pandas as pd

from .backtest import load, _walk_predict


def backtest(mode: str = "best", min_ev: float = 1.0,
             max_odds: float = 50.0, min_prob: float = 0.02) -> None:
    df = load()
    if len(df) < 200:
        print(f"データ不足（{len(df)}行）。src.collect で収集してください。")
        return
    df = df.reset_index(drop=True)
    df["proba"] = _walk_predict(df)
    df = df[df["proba"].notna()]

    staked = returned = bets = hits = 0
    races = ev_races = 0
    for _, g in df.groupby(["date", "jcd", "rno"]):
        if g["proba"].isna().any() or g["win_odds"].isna().all():
            continue
        races += 1
        s = g["proba"].sum()
        if s <= 0:
            continue
        cand = []
        for row in g.itertuples():
            o = getattr(row, "win_odds")
            if pd.isna(o) or o <= 0 or o > max_odds:
                continue
            p = row.proba / s
            if p < min_prob:
                continue
            ev = p * o
            if ev >= min_ev:
                cand.append((row.car, ev, o, p, row.win))
        if not cand:
            continue
        ev_races += 1
        if mode == "best":
            cand = [max(cand, key=lambda x: x[3])]  # 割安な中で最も当たりやすい1点
        for car, ev, o, p, wln in cand:
            bets += 1
            staked += 100
            if wln == 1:
                hits += 1
                returned += o * 100

    if bets == 0:
        print("対象ベットなし（EV≥閾値の単勝が無い）。")
        return
    roi = returned / staked * 100
    print(f"=== 単勝EVバックテスト（mode={mode} / EV≥{min_ev} / oddsキャップ{max_odds:.0f}）===")
    print(f"  予測済みレース : {races}  （うちEV該当 {ev_races}）")
    print(f"  ベット数       : {bets}")
    print(f"  的中率         : {hits / bets:.1%}")
    print(f"  回収率         : {roi:.1f}%   ({'✓ プラス' if roi >= 100 else '× マイナス'})")
    print("  ※確定オッズ前提。実弾は自分の投票でオッズが動く/締切乖離あり。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["best", "value"], default="best")
    ap.add_argument("--min-ev", type=float, default=1.0)
    ap.add_argument("--max-odds", type=float, default=50.0)
    ap.add_argument("--min-prob", type=float, default=0.02)
    args = ap.parse_args()
    backtest(args.mode, args.min_ev, args.max_odds, args.min_prob)


if __name__ == "__main__":
    main()
