"""エージェント：経験メモリ(布置→勝者試走順位)のk近傍で行動を決める。

行動 = {見送り} or {試走順位 r の車の単勝に bankroll の frac を張る}。
「似た布置」= 標準化布置空間のユークリッド近傍。近傍数k・半径・支持閾値・張る閾値・
配分fracは CONFIG に出し(探索対象)、真値としては埋めない。

転生で経験が効く機構:
  メモリは周回で持ち越し密度が上がる → ある布置の周りに十分近い近傍が集まる確率が
  上がる → 「支持が足りて勝負できる」布置が増える(参加率↑) → 経験が行動を変える。
  未触月には決してメモリを書かない(judge汚染防止)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .env import Race, MAXC


@dataclass
class Config:
    # ▼探索対象(今回の試走では固定)。人間が真値として決める杭ではない。
    k: int = 80                 # 近傍数
    radius: float = 6.0         # 近傍として認める最大距離(標準化空間・45次元で校正)
    min_support: int = 20       # これ未満の近傍数なら「経験不足」で見送り
    bet_threshold: float = 0.32 # 最頻勝者順位の近傍支持率がこれ未満なら見送り
    stake_frac: float = 0.03    # 1ベットで張る bankroll 比
    init_bankroll: float = 100_000.0
    min_bet: float = 100.0


@dataclass
class Memory:
    """布置ベクトル(標準化済) と ラベル(勝者試走順位) の追記型ストア。"""
    X: list = field(default_factory=list)
    y: list = field(default_factory=list)
    ym: list = field(default_factory=list)   # 各記憶の年月(同月除外用)
    _nn: object = None
    _n_built: int = 0

    def add_many(self, embs: np.ndarray, labels: list[int], yms: list[str]) -> None:
        for e, l, m in zip(embs, labels, yms):
            self.X.append(e)
            self.y.append(l)
            self.ym.append(m)

    def __len__(self):
        return len(self.X)

    def build(self, exclude_ym: str = None) -> None:
        """現メモリで近傍を組む(月初に1回)。exclude_ym を指定するとその月の記憶を除く
        (転生で同じ過去を生き直す際、今まさに再生中の月の答えを自己一致で引くのを防ぐ)。"""
        if len(self.X) == 0:
            self._nn = None
            return
        Xa = np.vstack(self.X)
        ya = np.array(self.y)
        if exclude_ym is not None:
            mask = np.array(self.ym) != exclude_ym
            Xa, ya = Xa[mask], ya[mask]
        if len(Xa) == 0:
            self._nn = None
            return
        self._nn = NearestNeighbors(n_neighbors=min(len(Xa), 200))
        self._nn.fit(Xa)
        self._ya = ya
        self._n_built = len(Xa)

    def estimate(self, emb: np.ndarray, cfg: Config):
        """布置 emb に対し (best_rank, support_ratio, n_support) を返す。
        n_support < min_support なら経験不足。"""
        if self._nn is None:
            return None, 0.0, 0
        d, idx = self._nn.kneighbors(emb.reshape(1, -1),
                                     n_neighbors=min(cfg.k, self._n_built))
        d, idx = d[0], idx[0]
        m = d <= cfg.radius
        d, idx = d[m], idx[m]
        n_sup = len(idx)
        if n_sup < cfg.min_support:
            return None, 0.0, n_sup
        w = 1.0 / (d + 1e-6)
        labels = self._ya[idx]
        # 距離重み付きで最頻の勝者試走順位
        score = np.zeros(MAXC + 1)
        for lab, wi in zip(labels, w):
            if 1 <= lab <= MAXC:
                score[lab] += wi
        best_rank = int(np.argmax(score))
        support = score[best_rank] / (w.sum() + 1e-12)
        return best_rank, support, n_sup


class Agent:
    """状態→行動。内部に転生を跨ぐ経験メモリを保持。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mem = Memory()

    def decide(self, race: Race):
        """(行動, 診断) を返す。行動 = None(見送り) or ('win', car)。"""
        best_rank, support, n_sup = self.mem.estimate(race.emb_std, self.cfg)
        if best_rank is None or best_rank < 1:
            return None, dict(reason="経験不足" if n_sup < self.cfg.min_support else "無効",
                              n_sup=n_sup)
        if support < self.cfg.bet_threshold:
            return None, dict(reason="支持薄", n_sup=n_sup, support=support)
        car = race.cars_by_rank[best_rank - 1]
        if car is None or car not in race.win_odds:
            return None, dict(reason="対象車オッズ無", n_sup=n_sup)
        return ("win", car), dict(reason="勝負", n_sup=n_sup, support=support,
                                  best_rank=best_rank)

    def remember(self, races: list[Race]) -> None:
        """学習月のレース群を経験メモリへ追記(転生で持ち越される)。"""
        self.mem.add_many([r.emb_std for r in races],
                          [r.winner_rank for r in races],
                          [r.ym for r in races])
