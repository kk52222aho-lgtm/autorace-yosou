@echo off
REM Daily results collection for autorace. ASCII ONLY.
REM (Japanese in a .cmd is read as CP932 by cmd.exe and can break the parser, so the
REM  task exits nonzero every morning while writing nothing - feedback_silent_cron_death.)
REM
REM Why this exists (2026-09-08): autorace_preclose has been collecting pre-close odds
REM daily since 2026-07-07 (192,041 rows / 797 races), but NOTHING collected the results.
REM payouts stopped at 20260629 - eight days BEFORE the forward collection even started.
REM So 43 days of forward data were unjudgeable.
REM
REM The general shape: forward collectors get a scheduled task because they are new and
REM perishable. Results collectors do not, because "the data will still be there".
REM Then the forward experiment cannot be scored. The same hole was found in kyotei and
REM in toto on the same day. See memory insight_forward_without_settle.
REM
REM Window is the last 14 days: collect skips what it already has, so re-running is cheap
REM and it self-heals a few missed days without a full backfill.

set PY=C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\python3.12.exe
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d C:\dev\autorace-yosou

for /f %%d in ('powershell -NoProfile -Command "(Get-Date).AddDays(-14).ToString('yyyyMMdd')"') do set FROM=%%d
for /f %%d in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyyMMdd')"') do set TO=%%d

"%PY%" -m src.collect --start %FROM% --end %TO% >> data\daily_results.log 2>&1
