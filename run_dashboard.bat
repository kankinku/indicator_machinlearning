@echo off
cd /d "%~dp0"
echo [Vibe Launcher] Starting Dashboard Server...

:: 1. Start Streamlit Server in Background (Minimized)
:: --server.headless=true: Prevent auto-opening standard browser tab
:: --server.runOnSave=true: Auto-reload on code changes
start /min "Vibe Dashboard Server" python -m streamlit run dashboard.py --server.headless=true --server.runOnSave=true

:: 2. Wait for Server to wake up
echo Waiting for server instantiation...
timeout /t 3 /nobreak >nul

:: 3. Open in Application Mode (Windowed, No Toolbar)
:: Using Microsoft Edge as the "App Runner" (Standard on Windows)
:: If you prefer Chrome, change 'msedge' to 'chrome'
echo [Vibe Launcher] Launching UI Window...
start msedge --app=http://localhost:8501

:: 4. Exit this launcher script automatically
exit
