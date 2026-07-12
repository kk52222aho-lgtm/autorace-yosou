"""オートレース 前向き実戦ダッシュボード（Streamlit）。

  streamlit run streamlit_app.py

過去の確定オッズ検証では「本物か幻か」を区別できない(リーク)ことが確定したため、
このサイトは締切前オッズで判断を固定した前向き実戦記録(3ストリーム)だけを見せる。
DB(predictions / pre_close_odds)を読むだけ＝軽量。数字は割引不要の本物・ただしN蓄積中。
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import streamlit as st

DATA = os.path.join(os.path.dirname(__file__), "data")
CSV = os.path.join(DATA, "predictions.csv")     # Cloud用の軽量同期ファイル(669MB DBは上げない)
DB = os.path.join(DATA, "autorace.db")          # ローカルのみ(フォールバック)
TARGET_FIRST, TARGET_CONCLUSIVE = 300, 1500

st.set_page_config(page_title="オートレース前向き実戦", page_icon="🏍️", layout="wide")


@st.cache_data(ttl=300)
def load():
    # Cloud: 軽量CSV / ローカル: CSV優先・無ければDB
    if os.path.exists(CSV):
        pred = pd.read_csv(CSV, dtype={"date": str, "jcd": str})
    elif os.path.exists(DB):
        con = sqlite3.connect(DB)
        try:
            pred = pd.read_sql_query("SELECT * FROM predictions", con)
        except Exception:
            pred = pd.DataFrame()
        con.close()
    else:
        pred = pd.DataFrame()
    return pred, None


def stream_roi(pred, stream):
    d = pred[(pred.stream == stream) & (pred.settled == 1) & (pred.bet_type.isin(["win", "trio"]))]
    if d.empty:
        return 0, 0, float("nan")
    n = len(d); h = int(d.hit.sum()); pay = d.payoff.fillna(0).sum()
    return n, h, pay / (n * 100) * 100


def main():
    st.title("🏍️ オートレース 前向き実戦")
    pred, pc = load()

    st.info(
        "**確定オッズの過去検証では『本物か幻か』を区別できない**(締切後オッズ=リーク、Step0.5で確定)。"
        "だからこのサイトは**締切前オッズで判断を固定した前向き実戦記録**だけを見せる。"
        "前向き＝リーク原理ゼロ＝割引不要の本物。ただし**N蓄積中で統計的にはまだ無意味**。"
        "統計的推定であり的中・利益を保証しない。投資は自己責任で。", icon="📋")

    n_races = pred[["date", "jcd", "rno"]].drop_duplicates().shape[0] if not pred.empty else 0
    n_days = pred["date"].nunique() if not pred.empty else 0
    bn, bh, broi = stream_roi(pred, "B")
    cn, ch, croi = stream_roi(pred, "C")
    # 高荒れ(受け皿ON)レース = Stream C を出したレース
    hi_races = pred[pred.stream == "C"][["date", "jcd", "rno"]].drop_duplicates().shape[0] if not pred.empty else 0

    c = st.columns(4)
    c[0].metric("記録レース", f"{n_races}", f"{n_days}日")
    c[1].metric("Stream B 記憶top単勝", f"ROI {broi:.0f}%" if bn else "—", f"{bn}ベット 的中{bh}")
    c[2].metric("Stream C 受け皿(本命抜き3連複)", f"ROI {croi:.0f}%" if cn else "—", f"{cn}ベット 的中{ch}")
    c[3].metric("受け皿レース(高荒れ)", f"{hi_races}", f"目標{TARGET_FIRST}で初回判定")

    st.progress(min(1.0, hi_races / TARGET_FIRST),
                text=f"(b)受け皿判定カウントダウン: 高荒れ {hi_races}/{TARGET_FIRST}R "
                     f"（決着は{TARGET_CONCLUSIVE}R）")

    st.subheader("3ストリームとは")
    st.markdown(
        "- **Stream A** 較正確率 vs 市場含意（ベースライン記録）\n"
        "- **Stream B** 記憶top（config+records 状況クラス K=150）の単勝＝鋭化セレクタ\n"
        "- **Stream C** 受け皿＝高荒れ×本命抜き3連複EV>1 ← 過去確定オッズで相対+21ptだった信号が、"
        "前向き実オッズで本物か幻かの直接判定")

    st.subheader("直近の判断ログ（締切前・immutable）")
    if not pred.empty:
        key = ["date", "jcd", "rno"]
        base = pred.groupby(key).agg(roughness=("roughness", "first"),
                                     settled=("settled", "max")).reset_index()
        cpts = (pred[pred.stream == "C"].groupby(key).size().rename("C点").reset_index())
        bpick = (pred[pred.stream == "B"][key + ["combo", "hit"]]
                 .rename(columns={"combo": "B_単勝車", "hit": "B的中"}))
        recent = (base.merge(cpts, on=key, how="left").merge(bpick, on=key, how="left")
                  .fillna({"C点": 0}))
        recent["受け皿"] = recent["roughness"].apply(lambda x: "★ON" if x and x >= 0.801 else "")
        recent = recent.sort_values(key, ascending=False).head(30)
        st.dataframe(recent[key + ["roughness", "受け皿", "B_単勝車", "B的中", "C点", "settled"]],
                     width="stretch", hide_index=True)
    else:
        st.write("まだ記録なし（毎晩 autorace_preclose が自動記録）。")

    with st.expander("これまでの地図（正直な結論）"):
        st.markdown(
            "- **全プール(単勝/2連単/2連複/3連複/3連単/ワイド)＋全セグメント＋買い目スライスで回収エッジ無し**"
            "（確定オッズ・検出力込みで確定）。市場は美しく較正、控除の壁(単勝25%/exotic30%)。\n"
            "- **記憶は生きてる**：読み出しをargmax→較正確率に直すと単勝セレクタが本命ベタ超え。"
            "状況クラス化(K=150,records込み)で更に鋭化(荒れ較正AUC 0.621)。**だが単勝ROI天井79%は不変**。\n"
            "- **唯一の候補=Stream C受け皿**：確定オッズで相対+21ptだが、最厳リーク割引で+8→消失＝**蜃気楼濃厚**。"
            "**確定オッズ上では確定不能**。だから前向き実戦で殺すか活かすかを決める。\n"
            "- 締切前オッズの実測リーク=単勝中央13%。過去に遡っての取得は原理的に不可能＝前向き蓄積が唯一の勝負場。")
    st.caption(f"DB: {DB} / 更新はキャッシュ5分。--reconcile で確定分の的中・払戻を突合。")


if __name__ == "__main__":
    main()
