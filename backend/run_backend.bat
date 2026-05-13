@echo off
cd /d %~dp0
if not exist venv (
  python -m venv venv
)
REM Always use the venv interpreter — avoids "ModuleNotFoundError: cv2" when global Python has no OpenCV.
"%~dp0venv\Scripts\python.exe" -m pip install -r requirements.txt
REM 127.0.0.1:8080 avoids WinError 10013 on some Windows setups for 0.0.0.0:8000
"%~dp0venv\Scripts\python.exe" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8080
