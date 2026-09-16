@echo off
cd /d C:\dev\autorace-yosou
if not exist logs mkdir logs
set LOG=logs\nightly.log
echo ===== start %date% %time% ===== >> %LOG%
REM collect pre-close odds + record 3-stream predictions (polls all day)
python -m src.loop.live_odds >> %LOG% 2>&1
echo [live_odds exit %errorlevel%] >> %LOG%
REM reconcile settled races, then push light CSVs to GitHub (Streamlit Cloud)
python -m src.loop.live_predict --reconcile >> %LOG% 2>&1
echo [reconcile exit %errorlevel%] >> %LOG%
python -m src.loop.sync_app >> %LOG% 2>&1
echo [sync_app exit %errorlevel%] >> %LOG%
echo ===== end %date% %time% ===== >> %LOG%
