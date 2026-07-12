"""オートレース 予想＆結果ボード（Streamlit）。

  streamlit run streamlit_app.py

レースの締切前に予想を記録し、結果と答え合わせした記録を「予想／結果／的中／配当」で
一覧表示する。過去の確定オッズ検証はリークで本物/幻を区別できないため、ここは前向きの
実戦記録だけ。DB(predictions/race_results)→軽量CSVを読む(669MB DBは載せない)。
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import streamlit as st

DATA = os.path.join(os.path.dirname(__file__), "data")
DB = os.path.join(DATA, "autorace.db")
VENUE = {"02": "川口", "03": "伊勢崎", "04": "浜松", "05": "飯塚", "06": "山陽"}

st.set_page_config(page_title="オートレース予想＆結果", page_icon="🏍️", layout="wide")


@st.cache_data(ttl=300)
def load():
    def rd(name):
        p = os.path.join(DATA, name + ".csv")
        if os.path.exists(p):
            return pd.read_csv(p, dtype={"date": str, "jcd": str})
        if os.path.exists(DB):
            con = sqlite3.connect(DB)
            try:
                df = pd.read_sql_query(f"SELECT * FROM {name}", con)
                df["date"] = df["date"].astype(str); df["jcd"] = df["jcd"].astype(str).str.zfill(2)
            except Exception:
                df = pd.DataFrame()
            con.close()
            return df
        return pd.DataFrame()
    return rd("predictions"), rd("race_results")


def _yen(x):
    return f"¥{int(round(x)):,}" if x and x > 0 else ""


def build_board(pred, res):
    """レース単位の 予想／結果／的中／配当 ボード。"""
    if pred.empty:
        return pd.DataFrame()
    key = ["date", "jcd", "rno"]
    rows = []
    for (d, j, r), g in pred.groupby(key):
        b = g[g.stream == "B"]
        c = g[g.stream == "C"]
        rr = res[(res.date == d) & (res.jcd == j) & (res.rno == r)] if not res.empty else pd.DataFrame()
        winner = int(rr.winner.iloc[0]) if not rr.empty and pd.notna(rr.winner.iloc[0]) else None
        settled = int(g.settled.max()) if "settled" in g else 0
        # 単勝(Stream B)
        b_car = int(b.combo.iloc[0]) if not b.empty else None
        b_odds = float(b.odds_at_decision.iloc[0]) if not b.empty else None
        b_hit = int(b.hit.fillna(0).iloc[0]) if not b.empty and b.hit.notna().any() else None
        b_pay = float(b.payoff.fillna(0).iloc[0]) if not b.empty else 0
        # 受け皿(Stream C)
        c_n = len(c)
        c_hit = int(c.hit.fillna(0).sum()) if c_n else 0
        c_pay = float(c.payoff.fillna(0).sum()) if c_n else 0
        tanshou = ("—" if not settled else ("○ " + _yen(b_pay) if b_hit else "✗")) if b_car else "—"
        uke = ("—" if c_n == 0 else
               (f"{c_n}点 " + ("○ " + _yen(c_pay) if c_hit else "✗") if settled else f"{c_n}点 (結果待ち)"))
        rows.append({
            "日付": f"{d[4:6]}/{d[6:8]}", "場": VENUE.get(j, j), "R": f"{r}R",
            "予想(単勝)": f"車{b_car} @{b_odds:.1f}" if b_car else "—",
            "結果(1着)": f"車{winner}" if winner else ("結果待ち" if not settled else "—"),
            "単勝": tanshou,
            "受け皿(本命抜き3連複)": uke,
            "_k": (d, j, r),
        })
    df = pd.DataFrame(rows).sort_values("_k", ascending=False).drop(columns="_k")
    return df


def roi(pred, stream):
    d = pred[(pred.stream == stream) & (pred.get("settled") == 1)
             & (pred.bet_type.isin(["win", "trio"]))] if not pred.empty else pd.DataFrame()
    if d.empty:
        return 0, 0, None
    n = len(d); h = int(d.hit.fillna(0).sum()); pay = d.payoff.fillna(0).sum()
    return n, h, pay / (n * 100) * 100


def main():
    st.title("🏍️ オートレース 予想 ＆ 結果")
    pred, res = load()

    st.warning(
        "**実験中・まだ勝ち負けは分かりません。** レースの締切前に予想を固定して結果と答え合わせしてる"
        "記録です（後からいじれない形）。まだ数が少なく、下の回収率はサンプル不足でアテになりません"
        "（数百レース貯まって初めて意味を持つ）。的中・利益を保証しません。投資は自己責任で。")

    n_races = pred[["date", "jcd", "rno"]].drop_duplicates().shape[0] if not pred.empty else 0
    bn, bh, broi = roi(pred, "B")
    cn, ch, croi = roi(pred, "C")
    m = st.columns(3)
    m[0].metric("答え合わせ済みレース", f"{n_races}")
    m[1].metric("単勝（毎レース1点）", f"的中 {bh}/{bn}",
                f"回収率 {broi:.0f}%" if broi is not None else "—")
    m[2].metric("受け皿（荒れる時だけ3連複）", f"的中 {ch}/{cn}",
                f"回収率 {croi:.0f}%" if croi is not None else "—")

    st.subheader("予想＆結果（新しい順）")
    board = build_board(pred, res)
    if board.empty:
        st.info("まだ記録がありません（毎晩、開催があれば自動で増えます）。")
    else:
        st.dataframe(board, width="stretch", hide_index=True, height=560)
        st.caption("単勝＝毎レース「記憶が一番買いと見た車」を1点。受け皿＝荒れると読んだレースだけ"
                   "本命を外した3連複を複数点。○＝的中（¥は100円あたりの払戻）。")

    with st.expander("これは何をやってるの？（仕組みと正直な現状）"):
        st.markdown(
            "- **予想はレース締切の直前に固定**して記録（後からいじれない）。だから出てる数字は"
            "「都合よく選んだ後出し」ではない本物。\n"
            "- **単勝**：状況を150クラスに分け、似た過去レースの結果から「一番勝ちそうな車」を1点。\n"
            "- **受け皿**：過去データで「本命が飛んで荒れそう」と読んだレースだけ、本命を外した3連複を買う。"
            "過去の確定オッズでは有望に見えたが、リークで本物か幻か区別できないと判明→**前向きの実戦で決着させてる最中**。\n"
            "- **これまでの結論**：オートレースの馬券は市場が効率的で、過去データ上はどの買い方も控除の壁に負けた。"
            "唯一の生き残り候補がこの『受け皿』で、それを今この記録で殺すか活かすか見てる。\n"
            "- **要は今は実験。数百レース貯まるまで勝ち負けは名乗らない。**")
    st.caption("毎晩、開催があれば締切前に自動で予想を記録→結果を突合→このページを更新。")


if __name__ == "__main__":
    main()
