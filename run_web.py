
import os
import sys
import subprocess
import time
import webbrowser
import signal
import socket

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    try:
        # Windows specific
        result = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
        for line in result.splitlines():
            parts = line.split()
            if len(parts) > 4:
                pid = parts[-1]
                print(f"[Launcher] Killing stale process {pid} on port {port}")
                subprocess.run(f"taskkill /F /PID {pid}", shell=True)
    except:
        pass

def main():
    api_url = "http://localhost:8001"
    
    # Pre-check port
    if is_port_in_use(8001):
        print("[Launcher] Port 8001 is busy. Attempting to clear...")
        kill_process_on_port(8001)
        time.sleep(1)

    print(f"[Launcher] Starting Dashboard System...")
    print(f"[Launcher] Host Configured: 0.0.0.0 (Accessible via Local IP)")

    first_launch = True
    
    while True:
        # 1. Start Server
        print("[Launcher] Starting API Server Subprocess...")
        server_process = subprocess.Popen(
            [sys.executable, "src/api/main.py"],
            cwd=os.getcwd()
            # creationflags=subprocess.CREATE_NEW_CONSOLE  <-- Removed to stop popup
        )
        
        # 2. Wait for startup (Health Check)
        print("[Launcher] Waiting for server to initialize...")
        server_ready = False
        import urllib.request
        
        for _ in range(30):
            try:
                urllib.request.urlopen(api_url, timeout=1)
                server_ready = True
                break
            except:
                if server_process.poll() is not None:
                    # Died during startup
                    break
                time.sleep(1)
                print(".", end="", flush=True)
        print("\n")
        
        if server_ready:
            print("[Launcher] Server is running!")
            if first_launch:
                print(f"[Launcher] Opening Browser: {api_url}")
                webbrowser.open(api_url)
                first_launch = False
        else:
            print("[Launcher] Server failed to start or died immediately.")
        
        # 3. Monitor Loop
        try:
            while True:
                ret_code = server_process.poll()
                if ret_code is not None:
                    print(f"[Launcher] Server process died! (Exit Code: {ret_code})")
                    print("[Launcher] Restarting in 5 seconds...")
                    break
                time.sleep(2)
        except KeyboardInterrupt:
            print("\n[Launcher] Stopping by user request...")
            server_process.send_signal(signal.SIGTERM)
            time.sleep(1)
            if server_process.poll() is None:
                server_process.kill()
            sys.exit(0)
            
        # If we broke local loop, it means server died.
        # Cleanup just in case
        if server_process.poll() is None:
             server_process.kill()
        
        time.sleep(5)

if __name__ == "__main__":
    main()
