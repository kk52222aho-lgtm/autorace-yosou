"""試走タイムの市場織り込み検定（レトロ・時計ゼロ）。

問い: 群衆は試走タイム(想定タイム=試走+ハンデ換算)を正しく織り込んどるか。
  - 過剰反応なら: implied確率で統制した後、良試走の残差係数が「負」
    (良タイムほど過剰投票=期待値マイナス)に出る。
  - 過小反応なら: 正に出る(市場を抜く残差予測力=生存軸の芽)。
  - ゼロなら: 教科書的効率(単勝teardownの結論と整合)。

手法: レース内条件付きロジット P(i勝ち) = softmax(b*log(imp_i) + g*good_i)。
  imp=正規化implied確率(確定オッズ)、good=想定タイムのレース内z(良いほど正)。
  g の符号と レース単位ブートストラップCI が主結果。
  断面: 年別(居座り確認) / 競争密度(想定タイム散らばり)3分位 / 良走路vs湿・雨。
  補助: 想定タイム順位×オッズ帯のフラットROI表(円が見える形)。

  python -m src.loop.verify_trial
"""
from __future__ import annotations

import numpy as np

from .. import storage

RNG = np.random.RandomState(0)
MAXC = 8
HANDI_SEC_PER_M = 0.001          # 10m ≒ 0.01s
N_BOOT = 1000


def load_races():
    """(logimp, good, mask, winner_idx, meta) の padded 配列を返す。"""
    conn = storage.connect()
    rows = conn.execute("""
        SELECT e.date, e.jcd, e.rno, e.car, e.trial_record, e.handicap,
               e.win, e.track_cond, w.odds
        FROM entries e JOIN win_odds w
          ON w.date=e.date AND w.jcd=e.jcd AND w.rno=e.rno AND w.car=e.car
        WHERE e.absent=0 AND e.finish IS NOT NULL
          AND e.trial_record > 3.0 AND e.trial_record < 5.0
          AND w.odds > 0
        ORDER BY e.date, e.jcd, e.rno, e.car""").fetchall()
    conn.close()

    races: dict[tuple, list] = {}
    for r in rows:
        races.setdefault((r[0], r[1], r[2]), []).append(r)

    LI, GD, MK, WIN = [], [], [], []
    meta = {"year": [], "dens": [], "good_cond": [], "odds": [], "grank": []}
    for key, cars in races.items():
        if len(cars) < 5 or sum(c[6] == 1 for c in cars) != 1:
            continue
        yoso = np.array([c[4] + c[5] * HANDI_SEC_PER_M for c in cars])
        if yoso.std() < 1e-9:
            continue
        odds = np.array([c[8] for c in cars], dtype=float)
        imp = (1.0 / odds) / (1.0 / odds).sum()
        good = -(yoso - yoso.mean()) / yoso.std()        # 良いほど正
        n = len(cars)
        li = np.full(MAXC, 0.0); gd = np.full(MAXC, 0.0)
        mk = np.zeros(MAXC); mk[:n] = 1.0
        li[:n] = np.log(imp); gd[:n] = good
        LI.append(li); GD.append(gd); MK.append(mk)
        WIN.append(int(np.argmax([c[6] for c in cars])))
        meta["year"].append(int(key[0][:4]))
        meta["dens"].append(float(yoso.std()))
        cond = cars[0][7] or ""
        meta["good_cond"].append(1 if cond == "良走路" else 0)
        meta["odds"].append(odds.tolist() + [np.nan] * (MAXC - n))
        # 想定タイム順位(1=最良)
        meta["grank"].append((np.argsort(np.argsort(-good[:n])) + 1).tolist()
                             + [0] * (MAXC - n))
    return (np.array(LI), np.array(GD), np.array(MK), np.array(WIN),
            {k: np.array(v) for k, v in meta.items()})


def fit_clogit(LI, GD, MK, WIN, iters=200):
    """2パラメタ条件付きロジットMLE(Newton)。戻り値 (b, g)。"""
    th = np.array([1.0, 0.0])
    X = np.stack([LI, GD], axis=2)                     # (R, 8, 2)
    for _ in range(iters):
        z = X @ th
        z = np.where(MK > 0, z, -1e9)
        z -= z.max(1, keepdims=True)
        p = np.exp(z) * (MK > 0)
        p /= p.sum(1, keepdims=True)
        xw = X[np.arange(len(WIN)), WIN]               # 勝者の特徴
        ex = (p[:, :, None] * X).sum(1)                # E_p[x]
        grad = (xw - ex).sum(0)
        # Hessian: -sum_r Cov_p(x)
        xc = X - ex[:, None, :]
        H = -np.einsum("rc,rci,rcj->ij", p, xc, xc)
        step = np.linalg.solve(H, grad)
        th -= step
        if np.abs(step).max() < 1e-10:
            break
    return th


def _fit_g(idx, LI, GD, MK, WIN):
    return fit_clogit(LI[idx], GD[idx], MK[idx], WIN[idx])


def boot_ci(LI, GD, MK, WIN, n_boot=N_BOOT):
    R = len(WIN)
    gs = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.randint(0, R, R)
        gs[b] = _fit_g(idx, LI, GD, MK, WIN)[1]
    return np.percentile(gs, [5, 95]), (gs > 0).mean()


def roi_table(meta, GD, MK, WIN):
    """想定タイム順位×オッズ帯のフラット100円ROI。"""
    odds = meta["odds"]; grank = meta["grank"]
    R = len(WIN)
    won = np.zeros_like(grank)
    won[np.arange(R), WIN] = 1
    bands = [("<3倍", 0, 3), ("3-10倍", 3, 10), (">10倍", 10, 9999)]
    print(f"\n  {'想定T順位':>8s}" + "".join(f"{b[0]:>16s}" for b in bands) + f"{'全体':>16s}")
    for rk in range(1, 6):
        row = f"  {rk:>8d}"
        sel_rk = grank == rk
        for _, lo, hi in bands + [("all", 0, 9999)]:
            sel = sel_rk & (odds >= lo) & (odds < hi) & ~np.isnan(odds)
            n = sel.sum()
            if n < 50:
                row += f"{'—':>16s}"
                continue
            ret = (odds[sel] * won[sel]).sum() / n * 100
            row += f"{ret:>9.1f}% n={n:<5d}"[:16].rjust(16)
        print(row)


def main():
    print("ロード中...")
    LI, GD, MK, WIN, meta = load_races()
    R = len(WIN)
    print(f"対象 {R:,}R\n")

    b, g = fit_clogit(LI, GD, MK, WIN)
    (lo, hi), ppos = boot_ci(LI, GD, MK, WIN)
    print("== 主結果: 条件付きロジット P(勝ち)=softmax(b*log(imp) + g*good想定T) ==")
    print(f"  b(市場係数)={b:+.3f}  (1=implied通りに較正)")
    print(f"  g(試走残差)={g:+.4f}  ブートCI(5-95%)=[{lo:+.4f},{hi:+.4f}]  P(g>0)={ppos:.1%}")
    print("  解釈: g>0=過小反応(市場を抜く芽) / g<0=過剰反応(良試走は買われすぎ) / CI跨ぎ=効率")

    print("\n== 年別(居座り確認) ==")
    for y in sorted(set(meta["year"])):
        idx = np.where(meta["year"] == y)[0]
        _, gy = _fit_g(idx, LI, GD, MK, WIN)
        print(f"  {y}: g={gy:+.4f} ({len(idx):,}R)")

    print("\n== 競争密度(想定タイムのレース内std)3分位 ==")
    qs = np.percentile(meta["dens"], [33.3, 66.7])
    labels = [f"拮抗(std<{qs[0]:.3f}s)", "中", f"バラ(std>{qs[1]:.3f}s)"]
    tier = np.digitize(meta["dens"], qs)
    for t in range(3):
        idx = np.where(tier == t)[0]
        _, gt = _fit_g(idx, LI, GD, MK, WIN)
        print(f"  {labels[t]}: g={gt:+.4f} ({len(idx):,}R)")

    print("\n== 走路状態 ==")
    for lab, v in [("良走路", 1), ("湿・雨ほか", 0)]:
        idx = np.where(meta["good_cond"] == v)[0]
        if len(idx) < 500:
            continue
        _, gc = _fit_g(idx, LI, GD, MK, WIN)
        print(f"  {lab}: g={gc:+.4f} ({len(idx):,}R)")

    print("\n== 補助: 想定タイム順位×オッズ帯 フラットROI(実オッズ) ==")
    roi_table(meta, GD, MK, WIN)

    print("\n ※確定オッズ基準。g≠0でも締切前オッズでの再確認(前向き収集)が本採用条件。")


if __name__ == "__main__":
    main()
