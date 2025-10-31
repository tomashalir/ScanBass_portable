@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo Starting ScanBass (portable)...
".\.venv311\Scripts\python.exe" ".\src\gui_app_bootstrap.py"
if errorlevel 1 (
  echo ---
  echo Pokud se neotevre GUI, vyfot chybu a posli mi ji prosim.
  echo ---
  pause
)
