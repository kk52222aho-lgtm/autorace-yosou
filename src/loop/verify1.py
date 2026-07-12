"""検証1：記憶の素の予測力。賭け/報酬/見送り/支持ゲートを全部外す。

WF(各月を過去月だけの記憶で予測・リーク無)で、記憶kNNに「勝者の試走順位(1..8)」を
予測させ、的中率を3つ並べる:
  - サイコロ            : レース内ランダム1車 = 平均 1/頭数
  - 母集団最頻          : 常に試走1番手 = P(勝者が試走1位)
  - 記憶ベース(kNN)     : k近傍の勝者順位の距離重み最頻

記憶が母集団最頻(≈38%)を超えないなら記憶は死んでる(実装が活かせてない)。
超えるなら記憶は生きてる=問題は報酬/賭け側。支持ゲートは掛けない(純予測力を見る)。

  python -m src.loop.verify1
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import env as E


def main():
    races = E.load_block_a()
    mu, sd = E.standardizer(races)
    for r in races:
        r.emb_std = (r.emb - mu) / sd
    months = sorted({r.ym for r in races})
    by = {m: [r for r in races if r.ym == m] for m in months}

    K = 80
    memX, memY = [], []
    dice, mode_hit, mem_hit, n = 0.0, 0, 0, 0
    # 記憶ベースが実際に予測した順位の分布も見る
    pred_rank_dist = np.zeros(E.MAXC + 1)

    for mi, m in enumerate(months):
        if mi > 0 and len(memX) >= K:
            nn = NearestNeighbors(n_neighbors=min(K, len(memX)))
            Xa = np.vstack(memX)
            nn.fit(Xa)
            ya = np.array(memY)
            batch = by[m]
            Q = np.vstack([r.emb_std for r in batch])
            dist, idx = nn.kneighbors(Q)
            for r, dd, ii in zip(batch, dist, idx):
                ncar = sum(1 for c in r.cars_by_rank if c is not None)
                dice += 1.0 / ncar
                if r.winner_rank == 1:
                    mode_hit += 1
                # 記憶: k近傍の勝者順位を距離重みで集計→最頻
                w = 1.0 / (dd + 1e-6)
                score = np.zeros(E.MAXC + 1)
                for lab, wi in zip(ya[ii], w):
                    if 1 <= lab <= E.MAXC:
                        score[lab] += wi
                pred = int(np.argmax(score))
                pred_rank_dist[pred] += 1
                if pred == r.winner_rank:
                    mem_hit += 1
                n += 1
        for r in by[m]:
            memX.append(r.emb_std)
            memY.append(r.winner_rank)

    print(f"WF予測レース {n:,} / 記憶件数(最終) {len(memX):,}\n")
    print("=== 検証1: 勝者の試走順位を当てる的中率 ===")
    print(f"  サイコロ(レース内ランダム)   : {dice/n*100:5.1f}%")
    print(f"  母集団最頻(常に試走1番手)     : {mode_hit/n*100:5.1f}%")
    print(f"  記憶ベース(kNN k={K})        : {mem_hit/n*100:5.1f}%")
    diff = (mem_hit - mode_hit) / n * 100
    print(f"\n  記憶 − 母集団最頻 = {diff:+.1f}pt  "
          f"→ {'記憶は生きてる(母集団最頻を超えた)' if diff > 0.5 else '記憶は死んでる(最頻に潰れ=実装が活かせてない)'}")
    print("\n  記憶が予測した順位の分布(1に偏るほど最頻に潰れ):")
    for rk in range(1, E.MAXC + 1):
        if pred_rank_dist[rk] > 0:
            print(f"    順位{rk}: {pred_rank_dist[rk]/n*100:5.1f}%")


if __name__ == "__main__":
    main()
