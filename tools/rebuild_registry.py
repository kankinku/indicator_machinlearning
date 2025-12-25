
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import config
from src.features.extended_population import populate_extended_population

def main():
    print("Starting Registry Rebuild...")
    
    reg_path = config.FEATURE_REGISTRY_PATH
    if reg_path.exists():
        print(f"Deleting existing registry file: {reg_path}")
        try:
            os.remove(reg_path)
        except Exception as e:
            print(f"Error removing file: {e}")
            return

    try:
        populate_extended_population()
        print("Successfully rebuilt registry with new metadata.")
    except Exception as e:
        print(f"Failed to rebuild registry: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
