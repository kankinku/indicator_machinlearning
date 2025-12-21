# === WARNING FILTERS (MUST BE FIRST) ===
# Suppress FutureWarnings from external libraries before any imports
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=r"ta\..*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Series.__setitem__.*")
warnings.filterwarnings("ignore", message="Series.fillna with 'method' is deprecated")

import sys
import os
import logging

# Ensure 'src' is importable
# Add the project root execution directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.shared.logger import get_logger
from src.orchestration.infinite_loop import infinite_loop

# Initialize Logger
logger = get_logger("main")

def main():
    logger.info(">>> [Vibe Launcher] Starting Meta-Optimization Engine...")
    
    # Execute Loop
    try:
        # Config is handled within infinite_loop via src.config
        # We do not pass hardcoded overrides to adhere to Configuration Separation rule.
        infinite_loop()
    except KeyboardInterrupt:
        logger.info(">>> [System] Shutting down gracefully. Bye.")
    except Exception as e:
        logger.critical(f"!!! [Critical Error] Unhandled exception in main loop: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    logging.captureWarnings(True) 
    main()
