@echo off
REM Auto-push script - chay luc 6h sang 24/05/2026 qua Task Scheduler.
REM Chi commit + push app.py. Pull --rebase truoc de tranh ghi de teammate.

set REPO=C:\Users\diosodumb\Documents\Final_CS231
set LOG=%REPO%\auto-push.log
cd /d "%REPO%"

echo. >> "%LOG%"
echo === %DATE% %TIME% === >> "%LOG%"

REM 1. Fetch latest tu remote
echo [1/5] Fetching... >> "%LOG%"
git fetch origin main >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ABORT: fetch failed >> "%LOG%"
    exit /b 1
)

REM 2. Kiem tra co thay doi tren app.py khong
git diff --quiet HEAD -- app.py
if errorlevel 1 (
    echo [2/5] app.py co thay doi. Staging + commit... >> "%LOG%"
    git add app.py >> "%LOG%" 2>&1
    git commit -m "UI: redesign theo Claude Design - 5 pipeline, custom lightbox, dark snapshot stage, badge status, summary table" >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo ABORT: commit failed >> "%LOG%"
        exit /b 1
    )
) else (
    echo [2/5] app.py chua co thay doi. Skip commit. >> "%LOG%"
)

REM 3. Rebase len origin/main de tranh ghi de
echo [3/5] Rebasing onto origin/main... >> "%LOG%"
git pull --rebase origin main >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ABORT: rebase conflict. Aborting rebase... >> "%LOG%"
    git rebase --abort >> "%LOG%" 2>&1
    echo CO XUNG DOT - khong push. User can manual review. >> "%LOG%"
    exit /b 1
)

REM 4. Push
echo [4/5] Pushing... >> "%LOG%"
git push origin main >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ABORT: push failed >> "%LOG%"
    exit /b 1
)

REM 5. Done - tu disable task de khong chay lai o logon tiep theo
echo [5/5] SUCCESS - %DATE% %TIME% >> "%LOG%"
schtasks /change /tn "FaceRecognition_AutoPush" /disable >> "%LOG%" 2>&1
echo Task da self-disable. De chay lai: schtasks /change /tn FaceRecognition_AutoPush /enable >> "%LOG%"
exit /b 0
