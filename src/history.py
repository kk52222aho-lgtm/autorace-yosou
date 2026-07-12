"""履歴（ローリング）特徴。リークなしで「その車/選手の“今より前”の実績」を作る。

rec_point は半期固定の静的評価。市場はより鮮度の高い直近フォーム・マシン状態・
ST傾向を織り込む。ここを埋めて市場本命(50%)超えを狙う。

因果性：date→rno で時系列ソートし、各 groupby 系列を shift(1) してから
expanding/rolling 平均を取る＝各行は「自分より前のレースだけ」を見る。
ライブ予測時も、DB の過去レース全部が履歴になるので同じ関数で計算できる。
"""
from __future__ import annotations

import pandas as pd

# 追加する履歴特徴カラム（features.FEATURES に取り込む）
HISTORY_COLS = [
    "p_n",            # 選手の過去出走数（経験＝データ内）
    "p_avg_order",    # 過去平均着順
    "p_win_rate",     # 過去勝率
    "p_r3_order",     # 直近3走平均着順（フォーム）
    "p_avg_st",       # 過去平均スタート(ST)
    "p_avg_trial",    # 過去平均試走
    "trial_dev",      # 今回試走 − 個人平均試走（＜0=好調のサイン）
    "v_n",            # 車両の過去出走数
    "v_avg_order",    # 車両の過去平均着順（マシン地力）
    "v_win_rate",     # 車両の過去勝率
]


def _prior_expanding(g, col):
    return g[col].transform(lambda s: s.shift().expanding().mean())


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """entries 由来 df に HISTORY_COLS を付与して返す（元の行順は保持）。"""
    df = df.copy()
    df["_ord"] = range(len(df))
    s = df.sort_values(["date", "rno", "_ord"])

    gp = s.groupby("player_id", sort=False)
    s["p_n"] = gp.cumcount()
    s["p_avg_order"] = _prior_expanding(gp, "finish")
    s["p_win_rate"] = _prior_expanding(gp, "win")
    s["p_r3_order"] = gp["finish"].transform(
        lambda x: x.shift().rolling(3, min_periods=1).mean())
    s["p_avg_st"] = _prior_expanding(gp, "start_timing")
    s["p_avg_trial"] = _prior_expanding(gp, "trial_record")
    s["trial_dev"] = s["trial_record"] - s["p_avg_trial"]

    gv = s.groupby("vehicle_id", sort=False)
    s["v_n"] = gv.cumcount()
    s["v_avg_order"] = _prior_expanding(gv, "finish")
    s["v_win_rate"] = _prior_expanding(gv, "win")

    s = s.sort_values("_ord").drop(columns="_ord")
    return s
