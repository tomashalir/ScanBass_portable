# run_scanbass.ps1
# 🚀 Start script for ScanBass Portable

# Nastavení cesty ke složce projektu
$projectPath = "C:\Users\42072\Downloads\ScanBass_portable"
$frontendPath = "$projectPath\frontend\index.html"
$venvActivate = "$projectPath\.venv311\Scripts\activate"

# 1️⃣ Přepnutí do složky projektu
Set-Location $projectPath

# 2️⃣ Aktivace virtuálního prostředí
Write-Host "Activating virtual environment..."
& $venvActivate

# 3️⃣ Spuštění backend serveru v novém okně
Write-Host "Starting backend server..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$projectPath`"; .\.venv311\Scripts\activate; python -m uvicorn src.web_service:app --host 0.0.0.0 --port 8000 --reload"

# 4️⃣ Počkej 3 sekundy, aby se server spustil
Start-Sleep -Seconds 3

# 5️⃣ Otevření frontendu v prohlížeči
Write-Host "Opening ScanBass UI..."
Start-Process "chrome.exe" $frontendPath

# ✅ Hotovo
Write-Host ""
Write-Host "✅ ScanBass is now running!"
Write-Host "Frontend opened in Chrome."
Write-Host "Backend live at: http://localhost:8000"
