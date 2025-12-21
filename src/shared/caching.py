
import os
from joblib import Memory
from src.config import config

# Initialize Joblib Memory
# We use a centralized cache directory in the workspace
CACHE_DIR = config.BASE_DIR / ".cache" / "joblib"
if not CACHE_DIR.exists():
    os.makedirs(CACHE_DIR, exist_ok=True)

memory = Memory(location=str(CACHE_DIR), verbose=0)

# Re-export decorator for cleaner imports
cache = memory.cache
