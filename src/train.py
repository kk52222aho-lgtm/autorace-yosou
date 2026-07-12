"""モデル学習。

各車が「1着になる確率」を予測する二値分類器。予測時はレース内で相対比較する。

オートレースの強い公衆ベースラインは2つ：**想定タイム1位**（機械的予測）と
**市場本命=単勝最低オッズ**。特に市場本命(初期計測で50%)を超えられるかが本丸。

例:
  python -m src.train
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

from . import history, storage
from .features import CAT_INDEX, FEATURES, build_frame

MODEL_PATH = os.path.join(storage.DATA_DIR, "model.joblib")


def _make_estimator():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=300,
        l2_regularization=1.0, categorical_features=CAT_INDEX, random_state=42,
    )


def load_dataset() -> pd.DataFrame:
    conn = storage.connect()
    df = pd.read_sql_query("SELECT * FROM entries WHERE win IS NOT NULL", conn)
    odds = pd.read_sql_query("SELECT date,jcd,rno,car,odds AS win_odds FROM win_odds", conn)
    conn.close()
    df["jcd"] = df["jcd"].astype(str).str.zfill(2)
    odds["jcd"] = odds["jcd"].astype(str).str.zfill(2)
    df = df.merge(odds, on=["date", "jcd", "rno", "car"], how="left")
    return history.add_history(df)


def race_topk_accuracy(df: pd.DataFrame, proba: np.ndarray) -> float:
    tmp = df.copy()
    tmp["p"] = proba
    hit = total = 0
    for _, g in tmp.groupby(["date", "jcd", "rno"]):
        actual = g.loc[g["win"] == 1, "car"]
        if len(actual) == 0:
            continue
        total += 1
        if g.loc[g["p"].idxmax(), "car"] == actual.iloc[0]:
            hit += 1
    return hit / total if total else 0.0


def _col_baseline(df: pd.DataFrame, col: str, ascending: bool) -> float:
    """各レースで col が最小(ascending)/最大の車を本命にした的中率。"""
    hit = total = 0
    for _, g in df.groupby(["date", "jcd", "rno"]):
        actual = g.loc[g["win"] == 1, "car"]
        if len(actual) == 0 or g[col].isna().all():
            continue
        total += 1
        pick = g.loc[(g[col].idxmin() if ascending else g[col].idxmax()), "car"]
        if pick == actual.iloc[0]:
            hit += 1
    return hit / total if total else 0.0


def baselines(df: pd.DataFrame) -> dict[str, float]:
    feat = build_frame(df, impute=False)
    d = df.copy()
    d["sotei_time"] = feat["sotei_time"]
    return {
        "想定タイム1位": _col_baseline(d, "sotei_time", ascending=True),
        "試走1位": _col_baseline(d, "trial_record", ascending=True),
        "得点1位": _col_baseline(d, "rec_point", ascending=False),
        "市場本命(単勝最低)": _col_baseline(d, "win_odds", ascending=True),
    }


def main():
    df = load_dataset()
    if len(df) < 60:
        print(f"学習データ不足（{len(df)}行）。src.collect で収集してください。")
        return

    feat = build_frame(df)
    X = feat[FEATURES].to_numpy(dtype=float)
    y = df["win"].to_numpy(dtype=int)

    dates = np.sort(df["date"].unique())
    cut = dates[int(len(dates) * 0.8)] if len(dates) > 4 else dates[-1]
    tr = df["date"].to_numpy() < cut
    te = ~tr
    if te.sum() == 0:
        tr = np.ones(len(df), bool); tr[::5] = False; te = ~tr

    model = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
    model.fit(X[tr], y[tr])

    p_te = model.predict_proba(X[te])[:, 1]
    try:
        auc = roc_auc_score(y[te], p_te)
    except ValueError:
        auc = float("nan")
    ll = log_loss(y[te], p_te, labels=[0, 1])
    acc = race_topk_accuracy(df[te], p_te)
    base = baselines(df[te])
    market = base["市場本命(単勝最低)"]
    verdict = "✓ 市場本命ベース超え" if acc > market else "× 市場本命ベース未達"

    print("=== 評価 (test) ===")
    print(f"  レース数(test) : {df[te].groupby(['date','jcd','rno']).ngroups}")
    print(f"  AUC            : {auc:.3f}")
    print(f"  LogLoss        : {ll:.3f}")
    print(f"  本命的中率(model): {acc:.1%}   → {verdict}")
    for name, v in base.items():
        print(f"  {name:14s}: {v:.1%}")

    final = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
    final.fit(X, y)
    joblib.dump({"model": final, "features": FEATURES}, MODEL_PATH)
    print(f"\nモデルを保存しました: {MODEL_PATH}")


if __name__ == "__main__":
    main()
