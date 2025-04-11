import win32evtlog

try:
    handle = win32evtlog.OpenEventLog('localhost', 'Security')
    print("✅ Successfully opened Security log.")
    win32evtlog.CloseEventLog(handle)
except Exception as e:
    print(f"❌ Failed to open Security log: {e}")