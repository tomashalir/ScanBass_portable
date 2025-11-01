@echo off
cd /d "C:\Users\42072\Downloads\ScanBass_portable"
echo Activating virtual environment...
call .venv311\Scripts\activate

echo Starting backend server...
start cmd /k ".venv311\Scripts\activate && python -m uvicorn src.web_service:app --host 0.0.0.0 --port 8000 --reload"

echo Waiting for backend to start...
timeout /t 3 >nul

echo Opening ScanBass UI...
start "" "C:\Users\42072\Downloads\ScanBass_portable\frontend\index.html"

echo ScanBass is running!
exit
