@echo off
REM Start the CodeQuery backend server on Windows.
REM
REM Prerequisites:
REM   1. Python 3.10+ with venv (this script creates it if needed)
REM   2. Ollama running with qwen2.5-coder:7b pulled
REM   3. Git installed and in PATH
REM
REM Usage:
REM   start.bat

cd /d "%~dp0"

REM Check for virtual environment
if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -q -r requirements.txt

REM Check if Ollama is running
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo WARNING: Ollama is not running!
    echo   Start it from another terminal: ollama serve
    echo   Then pull the model: ollama pull qwen2.5-coder:7b
    echo.
    echo Starting anyway (chat will fail until Ollama is available)...
)

REM Start the server
echo.
echo Starting CodeQuery backend on http://localhost:8000
echo API docs at http://localhost:8000/docs
echo.

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
