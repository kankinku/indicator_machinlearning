import os
import shutil
import sys
import subprocess
from pathlib import Path

# Add project root to sys.path to import modules if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import time

def kill_zombie_processes():
    print(">>> [System Reset] Killing running python processes (main.py, run_web.py)...")
    try:
        # Windows process termination using WMIC
        # We suppress output to keep it clean
        subprocess.run('wmic process where "CommandLine like \'%main.py%\'" call terminate', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run('wmic process where "CommandLine like \'%run_web.py%\'" call terminate', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)  # Wait for file handles to be released
        print("    [Process] Cleanup signals sent.")
    except Exception as e:
        print(f"    [Warning] Failed to kill processes: {e}")

def reset_system():
    # 0. Kill Zombies First!
    kill_zombie_processes()

    print(">>> [System Reset] Starting system initialization...")
    
    # 1. Define Paths to Clean
    paths_to_clean = [
        PROJECT_ROOT / "ledger",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "src" / "l3_meta" / "q_table.json",
        PROJECT_ROOT / "data" / "features.json"
    ]
    
    # 2. Delete Files/Directories
    for path in paths_to_clean:
        if path.exists():
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"    [Deleted] Directory: {path}")
                else:
                    os.remove(path)
                    print(f"    [Deleted] File: {path}")
            except Exception as e:
                print(f"    [Error] Failed to delete {path}: {e}")
                print("    (Hint: Close all python windows and try again)")
        else:
            print(f"    [Skip] Not found: {path}")

    # 3. Create Necessary Directories (Empty)
    # config.py usually handles this, but good to be explicit for logs and ledger
    (PROJECT_ROOT / "ledger").mkdir(exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    print("    [Created] Empty 'ledger' and 'logs' directories.")

    # 4. Populate Feature Registry
    print(">>> [System Reset] Populating Feature Registry...")
    try:
        # Import and run the population script
        # We run it as a subprocess to ensure clean state if it relies on globals
        import subprocess
        
        script_path = PROJECT_ROOT / "src" / "features" / "extended_population.py"
        result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            print("    [Success] Feature Registry populated.")
        else:
            print("    [Error] Failed to populate registry:")
            print(result.stderr)
            
    except Exception as e:
        print(f"    [Error] Failed to run population script: {e}")

    print(">>> [System Reset] Initialization Complete. Ready to run main.py.")

if __name__ == "__main__":
    confirm = input("Are you sure you want to RESET the entire system (delete all data)? [y/N]: ")
    if confirm.lower() == 'y':
        reset_system()
    else:
        print("Cancelled.")
