import os
import shutil
import sys
import subprocess
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def kill_running_processes():
    """Kills any running main.py or run_web.py processes to release file handles."""
    print(">>> [Reset] Cleaning up active processes...")
    try:
        # Use PowerShell to find and kill processes by command line content
        # This is much more reliable than window title matching on Windows
        scripts = ["main.py", "run_web.py"]
        for script in scripts:
            # We use a broad match for the script name in the command line
            ps_cmd = f"Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like '*{script}*' }} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}"
            subprocess.run(['powershell', '-Command', ps_cmd], capture_output=True)
        
        time.sleep(2)
        print("    [Process] Cleanup complete.")
    except Exception as e:
        print(f"    [Warning] Process cleanup had issues: {e}")

def delete_path(path: Path):
    """Deletes a file or directory tree."""
    if not path.exists():
        return
    
    try:
        if path.is_dir():
            # Handle Windows permission issues with a retry or simple rmtree
            shutil.rmtree(path)
            print(f"    [Deleted] Directory: {path.relative_to(PROJECT_ROOT)}")
        else:
            os.remove(path)
            print(f"    [Deleted] File: {path.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        print(f"    [Error] Could not delete {path}: {e}")

def reset_system():
    print("================================================================================")
    print("                     RL TRADING SYSTEM - FULL RESET                             ")
    print("================================================================================")
    
    # 0. Cleanup
    kill_running_processes()
    
    # 1. Define targets
    targets = [
        PROJECT_ROOT / "ledger",           # DB, Artifacts, Models
        PROJECT_ROOT / "logs",             # All logs
        PROJECT_ROOT / ".cache",           # Joblib disk cache
        PROJECT_ROOT / "data" / "features.json", # Feature Registry
        PROJECT_ROOT / "src" / "l3_meta" / "q_table.json", # Legacy state
        PROJECT_ROOT / "src" / "l3_meta" / "q_table_risk.json", # Legacy state
    ]
    
    # 2. Delete
    print(">>> [Reset] Deleting persistent state...")
    for target in targets:
        delete_path(target)
        
    # 3. Reconstruct directory structure
    print(">>> [Reset] Reconstructing directory structure...")
    (PROJECT_ROOT / "ledger" / "artifacts").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "ledger" / "diagnostics").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs" / "instrumentation").mkdir(parents=True, exist_ok=True)
    print("    [Created] ledger/artifacts/ , ledger/diagnostics/ , logs/instrumentation/")
    
    # 4. Repopulate Features
    print(">>> [Reset] Repopulating Feature Registry...")
    population_scripts = [
        PROJECT_ROOT / "tools" / "populate_features_v2.py",
        PROJECT_ROOT / "tools" / "populate_features_extended.py",
        PROJECT_ROOT / "tools" / "register_context_features.py"
    ]
    
    for script in population_scripts:
        if script.exists():
            print(f"    [Running] {script.name}...")
            res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
            if res.returncode != 0:
                print(f"    [Error] {script.name} failed:")
                print(res.stderr)
            else:
                print(f"    [Success] {script.name} completed.")
        else:
            print(f"    [Warning] {script.name} not found.")

    print("================================================================================")
    print(">>> [Reset] SYSTEM INITIALIZED SUCCESSFULLY.")
    print("    All previous experiments, logs, and models have been wiped.")
    print("    Feature Registry is restored with ~50 indicators + context features.")
    print("================================================================================")

if __name__ == "__main__":
    confirm = input("!!! WARNING !!! This will DELETE ALL DATA. Proceed? [y/N]: ")
    if confirm.lower() == 'y':
        reset_system()
    else:
        print("Reset cancelled.")
