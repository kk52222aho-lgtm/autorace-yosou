@echo off
cd /d C:\dev\autorace-yosou
REM 締切前オッズ収集 + 3ストリーム予想(締切直前)を一日中ポーリング
python -m src.loop.live_odds
REM 確定分を突合(的中・払戻) → 軽量CSVをGitHubへpush(Streamlit Cloud更新)
python -m src.loop.live_predict --reconcile
python -m src.loop.sync_app
