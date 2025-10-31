@echo off
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
  echo [ERROR] ISCC.exe not found at "%ISCC%". Install Inno Setup 6 or edit this path.
  pause
  exit /b 1
)
echo === Compiling installer ===
"%ISCC%" "ScanBass_exe.iss"
echo.
echo [OK] Check the "Output" folder here.
pause