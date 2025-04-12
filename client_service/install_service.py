import os
import sys
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import subprocess
import json
from pathlib import Path

class SessionSentryService(win32serviceutil.ServiceFramework):
    _svc_name_ = "SessionSentry"
    _svc_display_name_ = "Session Sentry Security Monitor"
    _svc_description_ = "Monitors and reports security events to central server"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        self.is_alive = True

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.is_alive = False

    def SvcDoRun(self):
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        self.main()

    def main(self):
        # Get the directory where the service is installed
        service_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create config file if it doesn't exist
        config_path = os.path.join(service_dir, 'config.json')
        if not os.path.exists(config_path):
            default_config = {
                "api_url": "http://localhost:3000",
                "api_key": "",
                "check_interval": 60,
                "backup_file": "security_analytics.json"
            }
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)

        # Run the main service script
        main_script = os.path.join(service_dir, 'main.py')
        while self.is_alive:
            try:
                subprocess.run([sys.executable, main_script], check=True)
            except Exception as e:
                servicemanager.LogErrorMsg(f"Error running service: {str(e)}")
            win32event.WaitForSingleObject(self.hWaitStop, 60000)  # Check every minute

def install_service():
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create necessary directories
        Path(os.path.join(current_dir, 'logs')).mkdir(exist_ok=True)
        
        # Install the service
        win32serviceutil.InstallService(
            pythonClassString=f"{__name__}.SessionSentryService",
            serviceName="SessionSentry",
            displayName="Session Sentry Security Monitor",
            description="Monitors and reports security events to central server",
            startType=win32service.SERVICE_AUTO_START
        )
        
        print("Service installed successfully!")
        print("\nTo start the service, run:")
        print("net start SessionSentry")
        print("\nTo stop the service, run:")
        print("net stop SessionSentry")
        
    except Exception as e:
        print(f"Error installing service: {str(e)}")
        sys.exit(1)

def uninstall_service():
    try:
        # Stop the service if it's running
        try:
            win32serviceutil.StopService("SessionSentry")
        except:
            pass
            
        # Uninstall the service
        win32serviceutil.RemoveService("SessionSentry")
        print("Service uninstalled successfully!")
        
    except Exception as e:
        print(f"Error uninstalling service: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'install':
            install_service()
        elif sys.argv[1] == 'uninstall':
            uninstall_service()
        else:
            win32serviceutil.HandleCommandLine(SessionSentryService)
    else:
        print("Usage:")
        print("python install_service.py install   - Install the service")
        print("python install_service.py uninstall - Uninstall the service")
        print("python install_service.py start     - Start the service")
        print("python install_service.py stop      - Stop the service") 