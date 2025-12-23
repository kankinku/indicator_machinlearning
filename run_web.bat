@echo off
echo [Vibe Launcher] Starting Dashboard...

:: 1. Start Python Launcher Script
:: We use call to run python script which handles the logic (server + browser check)
python run_web.py

echo [Vibe Launcher] Exited.
pause
