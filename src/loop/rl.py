"""リゼロ強化学習の器：状態→行動(見送り/人気k番に配分)の方策を月次収支報酬で学習。

- 1エピソード=ブロックA学習月の1ヶ月。方策(パラメータW)だけ持ち越し何万回も生き直す。
- 行動 = {0:見送り, k:人気k番(オッズ昇順)の車を単勝}。局面の重み付け・遡り・似の定義は
  方策(softmax)内部に創発させる(明示的に定義しない)。
- 報酬 = そのレースの利益率(勝ち:odds-1 / 負け:-1 / 見送り:0)。REINFORCE(episode毎に
  advantage標準化して分散安定)。オッズはリークのまま使う(純度は後段で締切前オッズで殴る)。
- 未触4ヶ月の月次収支中央値・破産率をチェックポイントで刻む。

  python -m src.loop.rl --episodes 10000
"""
from __future__ import annotations

import argparse

import numpy as np

from . import env as E

RNG = np.random.RandomState(0)
NA = E.MAXC + 1                 # 行動数(見送り + 人気1..8)
INIT_BANK = 100_000.0
STAKE_FRAC = 0.05
MIN_BET = 100.0


def _features(races):
    """各レースの (状態ベクトル, 人気別odds, 人気別勝敗, 有効行動マスク) を precompute。"""
    S, OD, WON, MASK, months = [], [], [], [], []
    for r in races:
        if not r.win_odds:
            continue
        ranked = sorted(r.win_odds.items(), key=lambda kv: kv[1])   # (car,odds)昇順
        s = sum(1.0 / o for _, o in ranked if o > 0)
        imp = [(1.0 / o) / s for _, o in ranked]
        imp8 = (imp + [0.0] * E.MAXC)[:E.MAXC]
        gap = imp8[0] - (imp8[1] if len(imp) > 1 else 0.0)
        upset = r.emb[E.MAXC * E.PER_SLOT + 3]                       # 当日荒れ
        state = np.array([1.0] + imp8 + [gap, upset], dtype=float)   # dim = 1+8+2=11
        odds = np.zeros(NA); won = np.zeros(NA); mask = np.zeros(NA)
        mask[0] = 1.0                                                # 見送りは常に有効
        for k, (car, o) in enumerate(ranked[:E.MAXC], start=1):
            odds[k] = o
            won[k] = 1.0 if car == r.winner_car else 0.0
            mask[k] = 1.0
        S.append(state); OD.append(odds); WON.append(won); MASK.append(mask)
        months.append(r.ym)
    return (np.array(S), np.array(OD), np.array(WON), np.array(MASK), np.array(months))


def _softmax(logits, mask):
    logits = np.where(mask > 0, logits, -1e9)
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z) * (mask > 0)
    return e / e.sum(1, keepdims=True)


def _reward(action, odds, won):
    """行動の利益率(見送り0 / 勝ちodds-1 / 負け-1)。ベクトル。"""
    r = np.zeros(len(action))
    bet = action > 0
    idx = np.arange(len(action))
    r[bet] = np.where(won[idx[bet], action[bet]] > 0,
                      odds[idx[bet], action[bet]] - 1.0, -1.0)
    return r


FLAT = 1000.0                  # フラット掛け(純edge測定・複利破産を排す)


def _flat_pnl(act, OD, WON):
    """フラット掛けの月次PnL和(見送り0/勝ち(odds-1)*FLAT/負け-FLAT)。"""
    bet = act > 0
    idx = np.arange(len(act))
    pnl = 0.0
    for i in idx[bet]:
        a = act[i]
        pnl += FLAT * (OD[i, a] - 1.0) if WON[i, a] > 0 else -FLAT
    return pnl, int(bet.sum())


def _bankruptcy(act, OD, WON):
    """現実掛け(bank5%)で破産したか(生存判定)。"""
    bank = INIT_BANK
    for i in range(len(act)):
        if bank < MIN_BET:
            return 1
        a = act[i]
        if a == 0:
            continue
        stake = min(bank, max(MIN_BET, bank * STAKE_FRAC))
        bank += stake * (OD[i, a] - 1.0) if WON[i, a] > 0 else -stake
    return 0


def _judge(W, Sh, ODh, WONh, MASKh, mh):
    """未触月次で フラットPnL中央値: pass許容 / 強制ベット / 本命ベタ / ランダム。破産率も。"""
    p_pass, p_force, p_fav, p_rand, brk, betcnt, total = [], [], [], [], 0, 0, 0
    for ym in sorted(set(mh)):
        m = mh == ym
        S, OD, WON, MASK = Sh[m], ODh[m], WONh[m], MASKh[m]
        probs = _softmax(S @ W.T, MASK)
        act = probs.argmax(1)
        pnl, nb = _flat_pnl(act, OD, WON)
        p_pass.append(pnl); betcnt += nb; total += len(act)
        brk += _bankruptcy(act, OD, WON)
        pf = probs.copy(); pf[:, 0] = -1
        p_force.append(_flat_pnl(pf.argmax(1), OD, WON)[0])
        fav = np.where(MASK[:, 1] > 0, 1, 0)
        p_fav.append(_flat_pnl(fav, OD, WON)[0])
        # ランダム: 有効な人気からランダム1車
        ra = np.array([RNG.choice(np.where(MASK[i, 1:] > 0)[0] + 1) for i in range(len(act))])
        p_rand.append(_flat_pnl(ra, OD, WON)[0])
    return (np.median(p_pass), np.median(p_force), np.median(p_fav), np.median(p_rand),
            brk / len(p_pass), betcnt / max(total, 1))


def train(n_episodes=10000, lr=0.2):
    print("ブロックA読み込み...")
    races = E.load_block_a()
    train_r = [r for r in races if r.ym <= E.TRAIN_END_YM]
    held_r = [r for r in races if r.ym in E.HELDOUT_YMS]
    S, OD, WON, MASK, M = _features(train_r)
    Sh, ODh, WONh, MASKh, Mh = _features(held_r)
    months = sorted(set(M))
    by = {ym: np.where(M == ym)[0] for ym in months}
    dim = S.shape[1]
    W = np.zeros((NA, dim))
    baseline_mu, baseline_sd = 0.0, 1.0
    checkpoints = [100, 500, 1000, 3000, 10000]

    print(f"学習月 {len(train_r):,}R({months[0]}..{months[-1]}) / 未触 {len(held_r):,}R "
          f"/ 状態次元{dim} 行動{NA}\n")
    print("フラット掛け¥1000/ベットの未触月次PnL中央値(純edge)。学習=pass許容, 強制=必ず賭ける\n")
    print(f"{'episode':>8s}{'pass許容':>11s}{'強制ベット':>11s}{'本命ベタ':>11s}"
          f"{'ランダム':>11s}{'破産率':>7s}{'ベット率':>8s}")
    for ep in range(1, n_episodes + 1):
        ym = months[(ep - 1) % len(months)]
        idx = by[ym]
        s, od, won, mask = S[idx], OD[idx], WON[idx], MASK[idx]
        probs = _softmax(s @ W.T, mask)
        # サンプリング
        cum = probs.cumsum(1)
        u = RNG.rand(len(s), 1)
        act = (u < cum).argmax(1)
        rew = _reward(act, od, won)
        # advantage 標準化(分散安定)
        adv = (rew - rew.mean()) / (rew.std() + 1e-6)
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(act)), act] = 1.0
        dlogits = adv[:, None] * (onehot - probs)
        dW = dlogits.T @ s / len(s)
        W += lr * dW * (1.0 - ep / (n_episodes * 1.5))    # lr減衰
        if ep in checkpoints:
            mp, mf, mfav, mr, brate, betrate = _judge(W, Sh, ODh, WONh, MASKh, Mh)
            print(f"{ep:8d}{mp:+11,.0f}{mf:+11,.0f}{mfav:+11,.0f}{mr:+11,.0f}"
                  f"{brate:7.0%}{betrate:8.1%}")
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=0.2)
    args = ap.parse_args()
    train(args.episodes, args.lr)
    print("\n ※オッズはリーク(確定)版。純度は締切前オッズ蓄積後に後追いで殴る。")


if __name__ == "__main__":
    main()
