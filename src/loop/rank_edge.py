"""タスクB：布置kNNに「勝てる布置に絞る力」が在るかを順位精度で判定(オッズ隔離)。

金額でなく的中で見る。ブロックA全体をウォークフォワード(各月を厳密に過去の月だけの
記憶で予測=リーク無)。kNNが高支持(近傍が密で確信)で絞った部分集合で、本命的中率が
市場含意本命(単勝1番人気=最低オッズ)の的中率を超える布置帯が在るか。

超えないなら布置kNNにはエッジ生成力なし → タスクC(メトリック学習)へ進む根拠。
超える帯が在れば、その帯がどんなレース型かを出す。

  python -m src.loop.rank_edge
"""
from __future__ import annotations

import numpy as np

from . import env as E
from .policy import Config, Memory

RNG = np.random.RandomState(0)


def _walk_forward(races, cfg: Config):
    """月順WF。各レースで (support, n_sup, model_hit, market_hit, trial1_hit, race) を収集。"""
    months = sorted({r.ym for r in races})
    by_month = {m: [r for r in races if r.ym == m] for m in months}
    mem = Memory()
    recs = []
    for mi, m in enumerate(months):
        if mi > 0:                      # 先頭月は過去が無い→予測せず記憶だけ
            mem.build()
            for r in by_month[m]:
                best_rank, support, n_sup = mem.estimate(r.emb_std, cfg)
                if best_rank is None or best_rank < 1:
                    continue
                model_car = r.cars_by_rank[best_rank - 1]
                if model_car is None:
                    continue
                # 市場含意本命=最低オッズ車(payoffでなく比較基準・最強ベースライン)
                mk = min(r.win_odds, key=lambda c: r.win_odds[c]) if r.win_odds else None
                recs.append(dict(
                    ym=m, support=support, n_sup=n_sup,
                    model_hit=int(model_car == r.winner_car),
                    market_hit=int(mk == r.winner_car) if mk is not None else None,
                    trial1_hit=int(r.cars_by_rank[0] == r.winner_car),
                    best_rank=best_rank, race=r))
        for r in by_month[m]:
            mem.add_many([r.emb_std], [r.winner_rank], [r.ym])
    return recs


def _rate(a):
    return np.mean(a) * 100 if len(a) else float("nan")


def _boot_diff(model, market, B=10000):
    """model_hit - market_hit(pt) のブートCI と P(diff<=0)。レース対応でペア。"""
    model, market = np.array(model), np.array(market)
    n = len(model)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    diff = (model.mean() - market.mean()) * 100
    boot = np.empty(B)
    for b in range(B):
        idx = RNG.randint(0, n, n)
        boot[b] = (model[idx].mean() - market[idx].mean()) * 100
    p5, p95 = np.percentile(boot, [5, 95])
    return diff, p5, p95, float((boot <= 0).mean() * 100)


def _row(name, rs):
    m = [r["model_hit"] for r in rs]
    mk = [r["market_hit"] for r in rs if r["market_hit"] is not None]
    mm = [r["model_hit"] for r in rs if r["market_hit"] is not None]  # market対応のmodel
    t1 = [r["trial1_hit"] for r in rs]
    diff, p5, p95, ple = _boot_diff(mm, mk)
    flag = " ✓超" if p5 > 0 else (" ~" if diff > 0 else "")
    print(f"{name:>16s}{len(rs):8d}{_rate(m):8.1f}%{_rate(mk):9.1f}%{_rate(t1):8.1f}%"
          f"{diff:+7.1f}pt{f'[{p5:+.1f},{p95:+.1f}]':>15s}{ple:7.1f}%{flag}")


def main():
    print("ブロックA読み込み・WF順位精度検証(オッズ隔離)...")
    races = E.load_block_a()
    mu, sd = E.standardizer(races)
    for r in races:
        r.emb_std = (r.emb - mu) / sd
    cfg = Config()
    recs = _walk_forward(races, cfg)
    print(f"WF予測レース(有効推定) {len(recs):,}  "
          f"[市場本命ベース的中率≈47%が壁 / model=kNN本命 / trial1=試走1番手]\n")

    print(f"{'サブセット':>16s}{'n':>8s}{'model':>8s}{'market':>9s}{'trial1':>8s}"
          f"{'差':>9s}{'CI(5-95)':>15s}{'P(≤0)':>8s}")
    # 全推定
    _row("全推定", recs)
    # 賭けた部分集合(高支持&支持数十分)
    bet = [r for r in recs if r["n_sup"] >= cfg.min_support and r["support"] >= cfg.bet_threshold]
    _row("賭けた集合", bet)
    # 支持帯別(n_sup>=min_support 前提)
    print("  --- 支持率帯別(確信の強さで層化, n_sup≥min_support) ---")
    conf = [r for r in recs if r["n_sup"] >= cfg.min_support]
    for lo, hi in [(0.32, 0.40), (0.40, 0.50), (0.50, 0.65), (0.65, 1.01)]:
        band = [r for r in conf if lo <= r["support"] < hi]
        if len(band) >= 100:
            _row(f"支持{lo:.2f}-{hi:.2f}", band)
    # 近傍密度帯別(n_sup)
    print("  --- 近傍密度帯別(n_sup, support≥bet_threshold) ---")
    dens = [r for r in recs if r["support"] >= cfg.bet_threshold]
    for lo, hi in [(20, 40), (40, 60), (60, 81)]:
        band = [r for r in dens if lo <= r["n_sup"] < hi]
        if len(band) >= 100:
            _row(f"n_sup{lo}-{hi}", band)
    print("\n ✓超=model的中がmarketをCI下限>0で超過(布置エッジ候補) / ~=点で超えるがCI跨ぐ / 無印=未達")


if __name__ == "__main__":
    main()
