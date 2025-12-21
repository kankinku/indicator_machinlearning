@echo off
echo Starting Vibe Trading Lab Web Dashboard...
echo Open http://localhost:8000 in your browser.
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
pause
