"""検証3-C(事前登録1セルのみ)：記憶の荒れ較正がexoticの受け皿を持つか。

事前登録(後から動かさない):
  roughness = 1 − 記憶が市場本命に付けた勝率。tuneで上位1/3の荒れ閾値を決め、testに固定適用。
  セル = 「記憶高荒れ × 本命抜き3連複 × 記憶EV>1」の1つだけ。総当たり禁止。
  判定 = 絞った群(高荒れ) vs 絞らない群(低荒れ/全体) の相対改善が、レース単位ブロックブートで
         有意か。確定オッズ=リークなので絶対ROIの黒字は名乗らず、相対改善のみ見る。

  python -m src.loop.verify3c
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import env as E
from .. import storage
from .verify2 import _memory_car_probs
from ..backtest_multi import _harville

RNG = np.random.RandomState(0)
K = 80
TUNE_FRAC = 0.6
ROUGH_QUANTILE = 0.667          # tuneで上位1/3を「高荒れ」と事前登録


def _trio_ladders():
    conn = storage.connect()
    import pandas as pd
    od = pd.read_sql_query(
        "SELECT date,jcd,rno,combo,odds FROM odds WHERE bet='trio' AND date BETWEEN ? AND ?",
        conn, params=E.BLOCK_A)
    conn.close()
    od["jcd"] = od["jcd"].astype(str).str.zfill(2)
    return {(d, j, r): dict(zip(g["combo"], g["odds"]))
            for (d, j, r), g in od.groupby(["date", "jcd", "rno"])}


def _group_boot(per_race, B=10000):
    """per_race: [(staked, returned)]。レース単位ブートのROI分布(%)を返す。"""
    if not per_race:
        return None
    st = np.array([x[0] for x in per_race], float)
    rt = np.array([x[1] for x in per_race], float)
    n = len(per_race)
    boot = np.array([(lambda i: rt[i].sum() / st[i].sum() * 100)(RNG.randint(0, n, n))
                     for _ in range(B)])
    return rt.sum() / st.sum() * 100, boot, n


def main():
    races = E.load_block_a()
    mu, sd = E.standardizer(races)
    for r in races:
        r.emb_std = (r.emb - mu) / sd
    trio_lad = _trio_ladders()
    months = sorted({r.ym for r in races})
    by = {m: [r for r in races if r.ym == m] for m in months}

    # WFで各レースの (date, roughness, [本命抜きEV>1の(staked,returned)集計]) を作る
    memX, memY = [], []
    rows = []   # (date, roughness, staked, returned)
    for mi, m in enumerate(months):
        if mi > 0 and len(memX) >= K:
            nn = NearestNeighbors(n_neighbors=min(K, len(memX)))
            nn.fit(np.vstack(memX)); ya = np.array(memY)
            Q = np.vstack([r.emb_std for r in by[m]])
            D, I = nn.kneighbors(Q)
            for r, dd, ii in zip(by[m], D, I):
                lad = trio_lad.get(r.key)
                if not lad or len(r.win_odds) < 4:
                    continue
                mp = _memory_car_probs(dd, ii, ya, r.cars_by_rank, r.win_odds)
                if not mp:
                    continue
                fav = min(r.win_odds, key=lambda c: r.win_odds[c])
                rough = 1.0 - mp.get(fav, 0.0)   # 記憶が本命に付けた勝率が低い=荒れ
                rows.append((r.key, rough, mp, lad, fav))
        for r in by[m]:
            memX.append(r.emb_std); memY.append(r.winner_rank)

    # realized trio(上位3着)を entries.finish から
    conn = storage.connect()
    import pandas as pd
    fin = pd.read_sql_query(
        "SELECT date,jcd,rno,car,finish FROM entries WHERE finish IS NOT NULL AND date BETWEEN ? AND ?",
        conn, params=E.BLOCK_A)
    conn.close()
    fin["jcd"] = fin["jcd"].astype(str).str.zfill(2)
    finmap = {(d, j, r): dict(zip(g["car"].astype(int), g["finish"].astype(int)))
              for (d, j, r), g in fin.groupby(["date", "jcd", "rno"])}

    recs = []   # (date, rough, staked, returned)  本命抜きEV>1・3連複
    for key, rough, mp, lad, fav in rows:
        fm = finmap.get(key, {})
        top3 = sorted([c for c in fm if fm[c] > 0], key=lambda c: fm[c])[:3]
        if len(top3) < 3:
            continue
        real = "-".join(map(str, sorted(top3)))
        probs = _harville(mp).get("trio", {})
        staked = returned = 0.0
        for combo, p in probs.items():
            cars = set(int(x) for x in combo.split("-"))
            if fav in cars:                       # 本命抜き のみ
                continue
            o = lad.get(combo)
            if not o or o <= 0:
                continue
            if p * o > 1.0:                       # 記憶EV>1 のみ
                staked += 100
                returned += o * 100 if combo == real else 0.0
        if staked > 0:
            recs.append((key[0], rough, staked, returned))

    # tune/test 分割 + tuneで荒れ閾値決定
    dates = sorted({x[0] for x in recs})
    cut = dates[int(len(dates) * TUNE_FRAC)]
    tune = [x for x in recs if x[0] < cut]
    test = [x for x in recs if x[0] >= cut]
    thr = np.quantile([x[1] for x in tune], ROUGH_QUANTILE)
    print(f"本命抜きEV>1・3連複を出せたレース {len(recs):,} / tune {len(tune):,} test {len(test):,}")
    print(f"事前登録: 高荒れ閾値(tune上位1/3) roughness≥{thr:.3f} を test に固定適用\n")

    hi = [(s, r) for d, ro, s, r in test if ro >= thr]
    lo = [(s, r) for d, ro, s, r in test if ro < thr]
    allt = [(s, r) for d, ro, s, r in test]
    rh = _group_boot(hi); rl = _group_boot(lo); ra = _group_boot(allt)
    print(f"{'群':>16s}{'races':>7s}{'ROI(確定=リーク)':>16s}{'CI(5-95)':>15s}")
    for name, g in [("高荒れ(絞る)", rh), ("低荒れ", rl), ("全体(絞らない)", ra)]:
        if g:
            roi, boot, n = g
            print(f"{name:>16s}{n:7d}{roi:14.0f}%{f'[{np.percentile(boot,5):.0f},{np.percentile(boot,95):.0f}]':>15s}")
    if rh and rl:
        diff = rh[1] - rl[1]
        print(f"\n 相対改善(高荒れ − 低荒れ): {rh[0]-rl[0]:+.0f}pt  "
              f"CI[{np.percentile(diff,5):+.0f},{np.percentile(diff,95):+.0f}]  "
              f"P(改善≤0)={ (diff<=0).mean()*100:.1f}%")
        verdict = "受け皿候補(相対改善が有意)" if np.percentile(diff, 5) > 0 else "受け皿なし(相対改善が有意でない)"
        print(f" → {verdict}")
    print("\n ※確定オッズ=リーク。絶対ROIの黒字は名乗らない。見るのは高荒れ絞りの相対改善のみ。")


if __name__ == "__main__":
    main()
