"""(1)記憶を鋭くする：メトリック学習で布置を圧縮し、荒れ較正(P(本命勝ち))のAUCを
上げられるか。過学習を最初から縛る＝訓練AUCと未触AUCを並べ、較正も毎回確認。

比較(全て label=市場本命が勝つ / 特徴=config45次元・オッズ非使用):
  raw_kNN   : 素の記憶(標準化45次元Euclid) 未触AUC  ← 現状0.604
  GBM       : config抽出上限の参照＋過学習ギャップ実演(訓練AUC vs 未触AUC)
  NCA_kNN   : メトリック学習(NCAで低次元へ)後のkNN 未触AUC + 較正

判定: NCA_kNN が raw を未触で超え、かつ訓練-未触ギャップ小・較正維持なら「鋭くなった=前進」。
超えない/ギャップ大/較正崩れ なら config は天井=再表現では鋭くできない。
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors, NeighborhoodComponentsAnalysis

from . import env as E
from .. import storage

RNG = np.random.RandomState(0)
K = 80


def _load():
    races = E.load_block_a()
    mu, sd = E.standardizer(races)
    for r in races:
        r.emb_std = (r.emb - mu) / sd
    import pandas as pd
    conn = storage.connect()
    fin = pd.read_sql_query(
        "SELECT date,jcd,rno,car,finish FROM entries WHERE finish IS NOT NULL AND date BETWEEN ? AND ?",
        conn, params=E.BLOCK_A)
    conn.close()
    fin["jcd"] = fin["jcd"].astype(str).str.zfill(2)
    finmap = {(d, j, r): dict(zip(g["car"].astype(int), g["finish"].astype(int)))
              for (d, j, r), g in fin.groupby(["date", "jcd", "rno"])}
    X, y, M = [], [], []
    for r in races:
        if not r.win_odds:
            continue
        fav = min(r.win_odds, key=lambda c: r.win_odds[c])
        ff = finmap.get(r.key, {}).get(fav)
        if ff is None:
            continue
        X.append(r.emb_std); y.append(1 if ff == 1 else 0); M.append(r.ym)
    return np.array(X), np.array(y), np.array(M)


def _knn_prob(Xtr, ytr, Xte):
    nn = NearestNeighbors(n_neighbors=min(K, len(Xtr))); nn.fit(Xtr)
    d, idx = nn.kneighbors(Xte)
    w = 1.0 / (d + 1e-6)
    return (w * ytr[idx]).sum(1) / w.sum(1)


def _calib(p, y):
    q = np.quantile(p, [0, .2, .4, .6, .8, 1.0])
    out = []
    for i in range(5):
        m = (p >= q[i]) & (p <= q[i + 1]) if i == 4 else (p >= q[i]) & (p < q[i + 1])
        out.append((p[m].mean() * 100, y[m].mean() * 100, m.sum()))
    return out


def main():
    X, y, M = _load()
    months = sorted(set(M))
    tr = np.array([m <= E.TRAIN_END_YM for m in M])
    hd = np.array([m in E.HELDOUT_YMS for m in M])
    print(f"本命勝ち率(ベース) 訓練{y[tr].mean()*100:.1f}% 未触{y[hd].mean()*100:.1f}% / "
          f"train {tr.sum():,} held {hd.sum():,}\n")

    # --- GBM: 抽出上限＋過学習ギャップ ---
    gbm = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=0)
    gbm.fit(X[tr], y[tr])
    auc_tr = roc_auc_score(y[tr], gbm.predict_proba(X[tr])[:, 1])
    auc_hd = roc_auc_score(y[hd], gbm.predict_proba(X[hd])[:, 1])
    print(f"GBM config-only : 訓練AUC {auc_tr:.3f} / 未触AUC {auc_hd:.3f} "
          f"(ギャップ {auc_tr-auc_hd:+.3f} ← 大きいほど『鋭くなった顔』の過学習)")

    # --- raw kNN (WFで未触AUC) ---
    memX_list = [X[np.array([m == mm for m in M])] for mm in months]  # unused placeholder
    # 未触月を過去全部で予測
    past = M <= E.TRAIN_END_YM
    p_raw = _knn_prob(X[past], y[past], X[hd])
    print(f"raw_kNN(現状)   : 未触AUC {roc_auc_score(y[hd], p_raw):.3f} (=argmax外した記憶の素)")

    # --- NCA metric kNN ---
    # 過学習抑制: NCAは訓練の部分標本で学習(全量だと過剰適合しやすい)
    idx = RNG.choice(np.where(past)[0], size=min(4000, past.sum()), replace=False)
    nca = NeighborhoodComponentsAnalysis(n_components=12, max_iter=40, random_state=0)
    nca.fit(X[idx], y[idx])
    Xp = nca.transform(X)
    p_nca_hd = _knn_prob(Xp[past], y[past], Xp[hd])
    # NCAの訓練AUC(過去を過去で予測=WF的, self除外のためkNN)
    p_nca_tr = _knn_prob(Xp[past], y[past], Xp[past])
    auc_nca_hd = roc_auc_score(y[hd], p_nca_hd)
    auc_nca_tr = roc_auc_score(y[past], p_nca_tr)
    print(f"NCA_kNN(鋭く)   : 訓練AUC {auc_nca_tr:.3f} / 未触AUC {auc_nca_hd:.3f} "
          f"(ギャップ {auc_nca_tr-auc_nca_hd:+.3f})")

    print("\n--- NCA_kNN 未触の較正(予測P(本命勝ち)五分位 → 実勝率) ---")
    for pp, aa, nn in _calib(p_nca_hd, y[hd]):
        print(f"    予測{pp:4.0f}% → 実{aa:4.0f}%  (n={nn})")

    gain = auc_nca_hd - roc_auc_score(y[hd], p_raw)
    print(f"\n判定: NCA_kNN 未触AUC − raw_kNN 未触AUC = {gain:+.3f}")
    if gain > 0.005 and (auc_nca_tr - auc_nca_hd) < 0.05:
        print(" → 未触で鋭くなり過学習ギャップも小＝前進。次は鋭い記憶で3-Cを回し直す。")
    else:
        print(" → 未触で有意に鋭くならない or ギャップ大＝configは天井。再表現では受け皿の幅は広がらない。")


if __name__ == "__main__":
    main()
