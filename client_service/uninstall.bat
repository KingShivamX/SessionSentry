@echo off
echo Uninstalling Session Sentry Client Service...

REM Stop the service if it's running
net stop SessionSentry >nul 2>&1

REM Uninstall the service
python install_service.py remove

REM Remove virtual environment
if exist venv (
    rmdir /s /q venv
)

echo.
echo Uninstallation complete!
echo.
echo All Session Sentry files have been removed.
echo.
pause 