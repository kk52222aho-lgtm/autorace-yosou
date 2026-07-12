"""状況クラス・エージェント：布置を離散クラスに切り、クラスごとに「教訓」(勝者試走順位の
分布)を貯めて、次の同クラスのレースに持ち越す。過去レースを時系列に流して積み上げる。

連続kNN(似た布置を距離で引く)の離散版＝解釈可能で、各クラスの教訓がオンライン更新される。
point-in-time: 各レースは「自分より前の同クラスの結果」だけで判断→結果を教訓に追記。

  クラス = KMeans(布置) で K個。学習月だけでfit(未来でクラスを定義しない)。
  教訓[クラス] = 勝者試走順位の累積カウント。判断は (クラス教訓 + 全体事前) の平滑化確率。
  クラスが埋まるほど教訓が鋭くなる = 経験の持ち越しが効く。

  python -m src.loop.situation_class
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

from . import env as E

RNG = np.random.RandomState(0)
K_CLASSES = 100
ALPHA = 25.0                 # 事前(全体分布)の強さ=クラスが薄い間は全体に寄る


class SituationClassAgent:
    def __init__(self, centroids, global_rank):
        self.centroids = centroids                 # (K, dim)
        self.global_rank = global_rank             # 全体の勝者試走順位分布(事前)
        self.counts = np.zeros((len(centroids), E.MAXC + 1))  # 教訓: クラス×順位カウント
        self.n = np.zeros(len(centroids))

    def classify(self, emb_std):
        return int(np.argmin(((self.centroids - emb_std) ** 2).sum(1)))

    def rank_probs(self, cls):
        """クラスの教訓(+全体事前で平滑化) → 勝者試走順位の確率分布。"""
        c = self.counts[cls, 1:] + ALPHA * self.global_rank[1:]
        return c / c.sum()

    def car_probs(self, cls, cars_by_rank, win_odds):
        rp = self.rank_probs(cls)
        probs = {}
        for r in range(1, E.MAXC + 1):
            car = cars_by_rank[r - 1]
            if car is not None and car in win_odds:
                probs[car] = rp[r - 1]
        s = sum(probs.values())
        return {c: p / s for c, p in probs.items()} if s > 0 else {}

    def learn(self, cls, winner_rank):
        if 1 <= winner_rank <= E.MAXC:
            self.counts[cls, winner_rank] += 1
            self.n[cls] += 1


def _roi_boot(bets, B=8000):
    pay = np.array([o if h else 0.0 for o, h in bets])
    n = len(pay)
    roi = pay.mean() * 100
    boot = np.array([pay[RNG.randint(0, n, n)].mean() for _ in range(B)]) * 100
    return roi, np.percentile(boot, 5), np.percentile(boot, 95)


def main():
    races = E.load_block_a()
    mu, sd = E.standardizer(races)
    for r in races:
        r.emb_std = (r.emb - mu) / sd
    races.sort(key=lambda r: (r.date, r.key[2], r.key[1]))   # time-order
    train = [r for r in races if r.ym <= E.TRAIN_END_YM]

    # クラス定義=学習月の布置でKMeans(未来でクラスを作らない)
    km = KMeans(n_clusters=K_CLASSES, n_init=4, random_state=0)
    km.fit(np.vstack([r.emb_std for r in train]))
    grank = np.zeros(E.MAXC + 1)
    for r in train:
        if 1 <= r.winner_rank <= E.MAXC:
            grank[r.winner_rank] += 1
    grank /= grank.sum()
    agent = SituationClassAgent(km.cluster_centers_, grank)

    # 時系列に流して predict-then-learn。未触月で評価。
    p_fav, y_fav, mt_bets = [], [], []
    for r in races:
        cls = agent.classify(r.emb_std)
        heldout = r.ym in E.HELDOUT_YMS
        if r.win_odds and agent.n[cls] >= 1:
            cp = agent.car_probs(cls, r.cars_by_rank, r.win_odds)
            if cp and heldout:
                fav = min(r.win_odds, key=lambda c: r.win_odds[c])
                p_fav.append(cp.get(fav, 0.0)); y_fav.append(1 if r.winner_car == fav else 0)
                mt = max(cp, key=lambda c: cp[c])
                mt_bets.append((r.win_odds[mt], mt == r.winner_car))
        agent.learn(cls, r.winner_rank)     # 結果を教訓に追記(次の同クラスへ持ち越し)

    auc = roc_auc_score(y_fav, p_fav)
    roi, p5, p95 = _roi_boot(mt_bets)
    print(f"状況クラス・エージェント (K={K_CLASSES}クラス, 未触{len(y_fav)}R)")
    print(f"  荒れ較正 P(本命勝ち) 未触AUC = {auc:.3f}  (連続kNN比: 0.604)")
    print(f"  クラスtop単勝ROI = {roi:.0f}% [{p5:.0f},{p95:.0f}]  (連続kNN mem_top比: 74%)")
    print(f"  平均クラス経験数 = {agent.n.mean():.0f}R/クラス (埋まるほど教訓が鋭化)\n")

    # 教訓の中身(解釈可能性): 荒れクラス上位3と堅いクラス上位3
    peak = agent.counts[:, 1] / np.maximum(agent.n, 1)     # 各クラスで試走1位が勝つ率
    order = np.argsort(peak)
    print("  状況クラスの教訓例(試走1位が勝つ率で両端):")
    for tag, idxs in [("荒れる型(試走1位が勝ちにくい)", order[:3]),
                      ("堅い型(試走1位が勝ちやすい)", order[::-1][:3])]:
        print(f"    【{tag}】")
        for ci in idxs:
            if agent.n[ci] < 20:
                continue
            rp = agent.counts[ci, 1:] / agent.n[ci]
            top = np.argsort(rp)[::-1][:3] + 1
            print(f"      class{ci}(n={agent.n[ci]:.0f}): 勝者試走順位 "
                  f"{', '.join(f'{r}位{rp[r-1]*100:.0f}%' for r in top)}")


if __name__ == "__main__":
    main()
