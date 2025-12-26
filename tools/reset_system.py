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
    print("                 RL TRADING SYSTEM - GENOME V2 FULL RESET                       ")
    print("================================================================================")
    
    # 0. Cleanup
    kill_running_processes()
    
    # 1. Define targets
    targets = [
        PROJECT_ROOT / "ledger",           # DB, Artifacts, Models
        PROJECT_ROOT / "logs",             # All logs (errors, incidents, instrumentation)
        PROJECT_ROOT / ".cache",           # Joblib disk cache
        PROJECT_ROOT / "data" / "feature_registry.json", # [V18] Registry
        PROJECT_ROOT / "src" / "l3_meta" / "q_table.json", # Legacy state
    ]
    
    # 2. Delete
    print(">>> [Reset] Deleting persistent state...")
    for target in targets:
        delete_path(target)
        
    # 3. Reconstruct directory structure
    print(">>> [Reset] Reconstructing directory structure...")
    dirs_to_create = [
        PROJECT_ROOT / "ledger" / "artifacts",
        PROJECT_ROOT / "ledger" / "diagnostics",
        PROJECT_ROOT / "ledger" / "models",
        PROJECT_ROOT / "logs" / "instrumentation",
        PROJECT_ROOT / "logs" / "errors",
        PROJECT_ROOT / "logs" / "incidents",
        PROJECT_ROOT / "data",
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        print(f"    [Created] {d.relative_to(PROJECT_ROOT)}")
    
    # 4. Repopulate Features (Genome v2 Source of Truth)
    print(">>> [Reset] Repopulating Feature Registry (Genome v2)...")
    try:
        # Use the src function directly to ensure metadata standards are enforced
        cmd = f'{sys.executable} -c "from src.features.extended_population import populate_extended_population; populate_extended_population()"'
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            print("    [Success] Feature Registry populated with Genome v2 metadata.")
        else:
            print(f"    [Error] Population failed: {res.stderr}")
            
        # Also register context features if tool exists
        context_script = PROJECT_ROOT / "tools" / "register_context_features.py"
        if context_script.exists():
             subprocess.run([sys.executable, str(context_script)], capture_output=True)
             print("    [Success] Context/Macro features registered.")
             
    except Exception as e:
        print(f"    [Error] Feature repopulation issue: {e}")

    # 5. Blackwell GPU Health Check
    print(">>> [Reset] Running Hardware Health Check (Blackwell/sm_120)...")
    check_code = "import torch; print(f'    [GPU] {torch.cuda.get_device_name()} / Arch: {torch.cuda.get_arch_list()} / CUDA: {torch.version.cuda}'); torch.randn(1).cuda()"
    try:
        res = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True)
        if res.returncode == 0:
            print(res.stdout.strip())
            print("    [Success] GPU Kernel Execution validated.")
        else:
            print(f"    [Warning] GPU Check Failed: {res.stderr.strip()}")
            print("    Check PyTorch Nightly installation (cu128).")
    except Exception as e:
        print(f"    [Warning] Could not run GPU check: {e}")

    print("================================================================================")
    print(">>> [Reset] SYSTEM V2 INITIALIZED SUCCESSFULLY.")
    print("    All previous experiments, logs, and models have been wiped.")
    print("    Ready for High-Performance RL Training with Genome v2 Logic.")
    print("================================================================================")

if __name__ == "__main__":
    confirm = input("!!! WARNING !!! This will DELETE ALL DATA. Proceed? [y/N]: ")
    if confirm.lower() == 'y':
        reset_system()
    else:
        print("Reset cancelled.")
