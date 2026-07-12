"""リゼロRL v2：生きた較正確率を状態に食わせ、見送りを最下位に罰する。

前回(rl.py)の全面見送り収束の真因は、argmaxで潰れた平らな入力＋pass=0(罰なし)。
今回:
  - 状態 = 記憶の較正済み確率(荒れ/堅い・変動する)＋市場含意＋roughness。平らでない。
  - 報酬序列: 勝ち(odds-1) > 負け(-1) > 見送り(PASS_PEN=-1.5)。張らないこと自体にコスト。
    「やって負け > 何もしない」を明示的に刻む。全面見送りは最下位。
  - リゼロ: ブロックA1ヶ月=1エピソード、方策W持ち越し、負けエピソードも方策に効かせる。

見るもの(絶対額でなく向き): 前回の全面見送りが覆り賭けに行くか / 周回で賭けたレースの
選び方(bet ROI)が育つか。

  python -m src.loop.rl2 --episodes 3000
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import env as E
from .verify2 import _memory_car_probs

RNG = np.random.RandomState(0)
NA = E.MAXC + 1
PASS_PEN = -1.5          # 見送りの罰(負け-1より下=最下位)
K = 80


def _precompute(races):
    """WFで各レースの (状態, 人気別odds/won/mask, month) を作る。状態に記憶較正確率を入れる。"""
    mu, sd = E.standardizer(races)
    for r in races:
        r.emb_std = (r.emb - mu) / sd
    months = sorted({r.ym for r in races})
    by = {m: [r for r in races if r.ym == m] for m in months}
    memX, memY = [], []
    S, OD, WON, MASK, MM = [], [], [], [], []
    for mi, m in enumerate(months):
        if mi > 0 and len(memX) >= K:
            nn = NearestNeighbors(n_neighbors=min(K, len(memX)))
            nn.fit(np.vstack(memX)); ya = np.array(memY)
            Q = np.vstack([r.emb_std for r in by[m]])
            D, I = nn.kneighbors(Q)
            for r, dd, ii in zip(by[m], D, I):
                if len(r.win_odds) < 2:
                    continue
                mp = _memory_car_probs(dd, ii, ya, r.cars_by_rank, r.win_odds)
                if not mp:
                    continue
                s = sum(1.0 / o for o in r.win_odds.values() if o > 0)
                imp = {c: (1.0 / o) / s for c, o in r.win_odds.items() if o > 0}
                ranked = sorted(r.win_odds.items(), key=lambda kv: kv[1])  # 人気順
                memp = np.zeros(E.MAXC); impv = np.zeros(E.MAXC)
                odds = np.zeros(NA); won = np.zeros(NA); mask = np.zeros(NA)
                mask[0] = 1.0
                for k, (car, o) in enumerate(ranked[:E.MAXC], start=1):
                    memp[k - 1] = mp.get(car, 0.0)
                    impv[k - 1] = imp.get(car, 0.0)
                    odds[k] = o; won[k] = 1.0 if car == r.winner_car else 0.0
                    mask[k] = 1.0
                rough = 1.0 - mp.get(ranked[0][0], 0.0)
                state = np.concatenate([[1.0], memp, impv, [rough]])  # dim=1+8+8+1=18
                S.append(state); OD.append(odds); WON.append(won); MASK.append(mask)
                MM.append(m)
        for r in by[m]:
            memX.append(r.emb_std); memY.append(r.winner_rank)
    return (np.array(S), np.array(OD), np.array(WON), np.array(MASK), np.array(MM))


def _softmax(logits, mask):
    logits = np.where(mask > 0, logits, -1e9)
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z) * (mask > 0)
    return e / e.sum(1, keepdims=True)


def _reward(act, OD, WON):
    r = np.full(len(act), PASS_PEN)          # 既定=見送りの罰
    bet = act > 0
    idx = np.arange(len(act))
    r[bet] = np.where(WON[idx[bet], act[bet]] > 0, OD[idx[bet], act[bet]] - 1.0, -1.0)
    return r


def _judge(W, S, OD, WON, MASK, MM):
    """未触月: greedy方策の ベット率 / 賭けたレースのROI / フラットPnL中央値。"""
    bet_r, rois, pnls = [], [], []
    for ym in sorted(set(MM)):
        m = MM == ym
        s, od, won, mask = S[m], OD[m], WON[m], MASK[m]
        act = _softmax(s @ W.T, mask).argmax(1)
        bet = act > 0
        idx = np.arange(len(act))
        staked = bet.sum() * 100
        ret = 0.0
        for i in idx[bet]:
            if won[i, act[i]] > 0:
                ret += od[i, act[i]] * 100
        bet_r.append(bet.mean())
        rois.append(ret / staked * 100 if staked > 0 else np.nan)
        pnls.append(ret - staked)
    return np.mean(bet_r), np.nanmedian(rois), np.median(pnls)


def train(n_episodes, lr=0.3):
    print("ブロックA・WF較正確率を前計算中...")
    races = E.load_block_a()
    S, OD, WON, MASK, MM = _precompute(races)
    tr = np.array([m <= E.TRAIN_END_YM for m in MM])
    hd = np.array([m in E.HELDOUT_YMS for m in MM])
    St, ODt, WONt, MASKt, MMt = S[tr], OD[tr], WON[tr], MASK[tr], MM[tr]
    Sh, ODh, WONh, MASKh, MMh = S[hd], OD[hd], WON[hd], MASK[hd], MM[hd]
    months = sorted(set(MMt))
    by = {m: np.where(MMt == m)[0] for m in months}
    W = np.zeros((NA, S.shape[1]))
    ckpt = [100, 300, 1000, 3000, 10000]
    print(f"学習{tr.sum():,}R / 未触{hd.sum():,}R / 状態{S.shape[1]}次元 / PASS_PEN={PASS_PEN}")
    print("報酬序列: 勝ち > 負け(-1) > 見送り(-1.5=最下位)\n")
    print(f"{'episode':>8s}{'未触ベット率':>13s}{'賭けたROI':>10s}{'フラットPnL中央':>15s}")
    for ep in range(1, n_episodes + 1):
        ym = months[(ep - 1) % len(months)]
        idx = by[ym]
        s, od, won, mask = St[idx], ODt[idx], WONt[idx], MASKt[idx]
        probs = _softmax(s @ W.T, mask)
        cum = probs.cumsum(1); u = RNG.rand(len(s), 1)
        act = (u < cum).argmax(1)
        rew = _reward(act, od, won)
        adv = (rew - rew.mean()) / (rew.std() + 1e-6)
        onehot = np.zeros_like(probs); onehot[np.arange(len(act)), act] = 1.0
        W += lr * ((adv[:, None] * (onehot - probs)).T @ s) / len(s) * (1 - ep / (n_episodes * 1.5))
        if ep in ckpt and ep <= n_episodes:
            br, roi, pnl = _judge(W, Sh, ODh, WONh, MASKh, MMh)
            print(f"{ep:8d}{br*100:12.1f}%{roi:9.0f}%{pnl:+14,.0f}円")
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    args = ap.parse_args()
    train(args.episodes)
    print("\n 見るもの=向き: 全面見送り(前回)が覆りベット率>0か / 周回で賭けたROIが育つか。")
    print(" ※確定オッズ=リーク。絶対ROIの黒字は名乗らない。")


if __name__ == "__main__":
    main()
