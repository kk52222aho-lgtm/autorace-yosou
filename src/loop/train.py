"""転生ループ + 未触judge。

エピソード = ブロックA内の1ヶ月。初期バンクロールで月初から1レースずつ消化し、
月末に収支確定。破産(月内枯渇)も収支に織り込む。エピソード後、経験メモリは持ち越す
(転生: 同じ過去を再び生きるが前周の経験を保持)。ブロックAを複数周。

judge: 未触Nヶ月(2024-09..12)を、その周までに貯めた経験メモリで走らせ、月次収支の
分布(中央値・破産率)を出す。未触月のレースはメモリに書かない。学習月の収支は判定に使わない。

オッズは決定に使わず payoff だけに使う(確定オッズ=リーク。この心臓では相対順位学習が主で、
月次収支は骨が回る事の確認用。黒赤の judge は締切前オッズを足す次段)。
"""
from __future__ import annotations

import numpy as np

from . import env as E
from .policy import Agent, Config


def _play_month(agent: Agent, races_m: list, cfg: Config):
    """1ヶ月を1レースずつ消化。(月次損益, 破産flag, ベット数, 参加率) を返す。"""
    bank = cfg.init_bankroll
    bets = 0
    seen = 0
    bankrupt = False
    for r in races_m:
        seen += 1
        if bank < cfg.min_bet:
            bankrupt = True
            break
        action, _diag = agent.decide(r)
        if action is None:
            continue
        _kind, car = action
        stake = max(cfg.min_bet, bank * cfg.stake_frac)
        stake = min(stake, bank)
        bets += 1
        if car == r.winner_car:
            odds = r.win_odds.get(car, 0.0)
            bank += stake * (odds - 1.0)      # 単勝払戻(確定オッズ)
        else:
            bank -= stake
    pnl = bank - cfg.init_bankroll
    part = bets / seen if seen else 0.0
    return pnl, bankrupt, bets, part


def _judge(agent: Agent, heldout: list, cfg: Config):
    """未触月ごとの月次収支分布(メモリは書かない)。"""
    agent.mem.build()   # 現メモリで近傍を固定
    by_month: dict[str, list] = {}
    for r in heldout:
        by_month.setdefault(r.ym, []).append(r)
    rows = []
    for ym in sorted(by_month):
        pnl, bankrupt, bets, part = _play_month(agent, by_month[ym], cfg)
        rows.append((ym, pnl, bankrupt, bets, part))
    return rows


def run(n_laps: int = 3, cfg: Config = None):
    cfg = cfg or Config()
    print("ブロックA読み込み中...")
    races = E.load_block_a()
    mu, sd = E.standardizer(races)
    for r in races:
        r.emb_std = (r.emb - mu) / sd
    train, heldout, train_months = E.split_months(races)
    train_by_month = {ym: [r for r in train if r.ym == ym] for ym in train_months}
    print(f"ブロックA: 全{len(races):,}R / 学習{len(train):,}R({train_months[0]}..{train_months[-1]}, "
          f"{len(train_months)}ヶ月) / 未触{len(heldout):,}R({'/'.join(E.HELDOUT_YMS)})")
    print(f"CONFIG(探索対象/今回固定): k={cfg.k} radius={cfg.radius} min_support={cfg.min_support} "
          f"bet_thr={cfg.bet_threshold} frac={cfg.stake_frac} bank0={cfg.init_bankroll:,.0f}\n")

    agent = Agent(cfg)
    for lap in range(1, n_laps + 1):
        # --- 学習フェーズ: 月順に転生。各月開始時の(持ち越し)メモリで走り、月末に追記 ---
        train_pnls, train_bankrupt = [], 0
        for ym in train_months:
            agent.mem.build(exclude_ym=ym)          # 同月の答えは自己一致で引かせない
            pnl, bankrupt, _bets, _part = _play_month(agent, train_by_month[ym], cfg)
            train_pnls.append(pnl)
            train_bankrupt += int(bankrupt)
            agent.remember(train_by_month[ym])       # 月末にこの月をメモリへ(転生持ち越し)
        # --- θ恒常性: 学習中の破産率から閾値を微調整(報酬由来・ハードコードでない) ---
        br = train_bankrupt / len(train_months)
        if br > 0.3:
            cfg.bet_threshold = min(0.6, cfg.bet_threshold + 0.02)
        elif br < 0.1:
            cfg.bet_threshold = max(0.2, cfg.bet_threshold - 0.01)

        # --- judge: 未触月(メモリは書かない) ---
        rows = _judge(agent, heldout, cfg)
        pnls = [x[1] for x in rows]
        brate = np.mean([x[2] for x in rows])
        med = np.median(pnls)
        parts = np.mean([x[4] for x in rows])
        print(f"===== 周回 {lap} =====  経験メモリ件数={len(agent.mem):,}  "
              f"(学習中: 平均月次{np.mean(train_pnls):+,.0f}円 破産率{br:.0%} → bet_thr={cfg.bet_threshold:.2f})")
        print(f"  未触4ヶ月の月次収支:")
        for ym, pnl, bankrupt, bets, part in rows:
            tag = " ★破産" if bankrupt else ""
            print(f"    {ym}: {pnl:+11,.0f}円  (bets={bets:4d} 参加率{part:5.1%}){tag}")
        print(f"  → 中央値 {med:+,.0f}円 / 破産率 {brate:.0%} / 平均参加率 {parts:.1%}\n")


if __name__ == "__main__":
    run(3)
