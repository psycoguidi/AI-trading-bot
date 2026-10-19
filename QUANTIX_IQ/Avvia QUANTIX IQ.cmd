@echo off
cd /d "%~dp0"
if not exist "07_collegamento\config.json" (
  echo Prima configura il progetto come descritto in README.md.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Ambiente Python assente. Segui README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 "07_collegamento\avvio.py" 
if errorlevel 1 pause
