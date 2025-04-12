@echo off
echo Installing Session Sentry Client Service...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.8 or later from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate

REM Install dependencies
pip install -r requirements.txt

REM Install the service
python install_service.py install

echo.
echo Installation complete!
echo.
echo The service will start automatically on system startup.
echo.
echo To manually start the service, run: net start SessionSentry
echo To stop the service, run: net stop SessionSentry
echo.
pause 