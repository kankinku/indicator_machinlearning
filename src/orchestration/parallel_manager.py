
from joblib import Parallel
from typing import Optional
from src.config import config

class ParallelManager:
    """
    [V14-O] Persistent Parallel Pool Manager
    Handles worker pool reuse to avoid spawn overhead on Windows.
    """
    _instance = None
    _pool: Optional[Parallel] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ParallelManager, cls).__new__(cls)
        return cls._instance

    def get_pool(self) -> Parallel:
        """
        Returns a reusable Parallel context.
        Note: Joblib's Parallel with backend='loky' handles pool reuse internally 
        if we keep the instance alive or use the global one.
        """
        if self._pool is None:
            self._pool = Parallel(
                n_jobs=config.PARALLEL_MAX_WORKERS,
                backend=config.PARALLEL_BACKEND,
                timeout=config.PARALLEL_TIMEOUT,
                verbose=0
            )
        return self._pool

def get_parallel_pool() -> Parallel:
    return ParallelManager().get_pool()
