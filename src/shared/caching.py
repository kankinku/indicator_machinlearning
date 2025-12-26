"""
Caching Module - 성능 최적화를 위한 캐싱 시스템

이 모듈은 두 가지 캐싱 전략을 제공합니다:
1. Disk Cache (joblib Memory): 장기간 유지되는 결과 캐싱
2. Memory Cache (LRU): DataFrame 해시 기반 빠른 인메모리 캐싱

원칙:
- 동일한 입력에 대해서는 재계산 없이 캐시된 결과 반환
- DataFrame 변경 감지에는 pd.util.hash_pandas_object 사용
- 메모리 효율을 위해 LRU 방식 제한
"""
import os
import hashlib
import json
import time
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar
from collections import OrderedDict
from threading import Lock

import pandas as pd
import numpy as np

from joblib import Memory
from src.config import config

# ============================================
# 1. Disk Cache (Joblib - Persistent)
# ============================================
CACHE_DIR = config.BASE_DIR / ".cache" / "joblib"
if not CACHE_DIR.exists():
    os.makedirs(CACHE_DIR, exist_ok=True)

memory = Memory(location=str(CACHE_DIR), verbose=0)

# Re-export decorator for cleaner imports
cache = memory.cache


# ============================================
# 2. Memory Cache (LRU - Fast, In-Memory)
# ============================================

class DataFrameCache:
    """
    DataFrame 기반 결과를 위한 LRU 메모리 캐시.
    
    pd.util.hash_pandas_object를 사용하여 DataFrame의 변경을 빠르게 감지합니다.
    Thread-safe 설계입니다.
    """
    
    def __init__(self, maxsize: int = 128):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._maxsize = maxsize
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
    
    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """
        DataFrame을 빠르게 해시합니다.
        pd.util.hash_pandas_object 사용 - Series/DataFrame 전용 최적화.
        """
        try:
            # 열 이름과 인덱스까지 포함한 해시
            content_hash = pd.util.hash_pandas_object(df, index=True).sum()
            col_hash = hash(tuple(df.columns.tolist()))
            shape_hash = hash(df.shape)
            return f"{content_hash}_{col_hash}_{shape_hash}"
        except Exception:
            # 폴백: numpy 바이트 해시
            return hashlib.md5(df.values.tobytes()).hexdigest()
    
    def _hash_dict(self, d: Dict) -> str:
        """딕셔너리를 해시합니다."""
        try:
            return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(d).encode()).hexdigest()
    
    def make_key(self, df: pd.DataFrame, genome: Dict, *args) -> str:
        """
        캐시 키를 생성합니다.
        
        Args:
            df: 입력 DataFrame
            genome: 피처 genome 딕셔너리
            *args: 추가 해시에 포함할 인수들
        """
        df_hash = self._hash_dataframe(df)
        genome_hash = self._hash_dict(genome)
        args_hash = hashlib.md5(str(args).encode()).hexdigest()[:8]
        return f"{df_hash}_{genome_hash}_{args_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값을 가져옵니다."""
        with self._lock:
            if key in self._cache:
                # LRU: 최근 사용으로 이동
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """캐시에 값을 저장합니다."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    # 가장 오래된 항목 제거
                    self._cache.popitem(last=False)
            self._cache[key] = value
    
    def clear(self) -> None:
        """캐시를 비웁니다."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    @property
    def stats(self) -> Dict[str, int]:
        """캐시 통계를 반환합니다."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "maxsize": self._maxsize,
            "hit_rate_pct": round(hit_rate, 2),
        }


class ObjectCache:
    """
    Generic LRU cache with optional TTL for non-DataFrame payloads.
    """
    def __init__(self, maxsize: int = 128, ttl_sec: Optional[int] = None):
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl_sec = ttl_sec
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            value, ts = self._cache[key]
            if self._ttl_sec and (time.time() - ts) > self._ttl_sec:
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
            self._cache[key] = (value, time.time())

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> Dict[str, int]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "maxsize": self._maxsize,
            "hit_rate_pct": round(hit_rate, 2),
        }


# 전역 캐시 인스턴스
_feature_cache = DataFrameCache(maxsize=256)
_label_cache = DataFrameCache(maxsize=128)
_signal_cache = DataFrameCache(maxsize=getattr(config, "SIGNAL_CACHE_MAXSIZE", 256))
_backtest_cache = ObjectCache(
    maxsize=getattr(config, "BACKTEST_CACHE_MAXSIZE", 128),
    ttl_sec=getattr(config, "BACKTEST_CACHE_TTL_SEC", 3600),
)


def get_feature_cache() -> DataFrameCache:
    """피처 캐시 인스턴스를 반환합니다."""
    return _feature_cache


def get_label_cache() -> DataFrameCache:
    """라벨 캐시 인스턴스를 반환합니다."""
    return _label_cache


def get_signal_cache() -> DataFrameCache:
    """Signal cache for logic-tree evaluation results."""
    return _signal_cache


def get_backtest_cache() -> ObjectCache:
    """Backtest result cache keyed by policy/window/config signatures."""
    return _backtest_cache


# ============================================
# 3. Decorator for DataFrame-based Caching
# ============================================

F = TypeVar('F', bound=Callable)

def df_cache(cache_instance: DataFrameCache = None):
    """
    DataFrame 기반 함수를 위한 캐시 데코레이터.
    
    첫 번째 인수가 DataFrame이고, 두 번째 인수가 genome dict인 함수에 사용합니다.
    
    Example:
        @df_cache()
        def generate_features(df: pd.DataFrame, genome: Dict) -> pd.DataFrame:
            ...
    """
    def decorator(func: F) -> F:
        cache = cache_instance or _feature_cache
        
        @wraps(func)
        def wrapper(df: pd.DataFrame, genome: Dict, *args, **kwargs):
            # 캐시 키 생성
            key = cache.make_key(df, genome, *args, tuple(sorted(kwargs.items())))
            
            # 캐시 조회
            cached = cache.get(key)
            if cached is not None:
                return cached
            
            # 계산 수행
            result = func(df, genome, *args, **kwargs)
            
            # 결과 캐싱
            cache.set(key, result)
            return result
        
        # 캐시 접근을 위한 속성 추가
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = lambda: cache.stats
        
        return wrapper
    return decorator


def clear_all_caches() -> Dict[str, Dict]:
    """모든 캐시를 비우고 통계를 반환합니다."""
    stats = {
        "feature_cache": _feature_cache.stats,
        "label_cache": _label_cache.stats,
        "signal_cache": _signal_cache.stats,
        "backtest_cache": _backtest_cache.stats,
    }
    _feature_cache.clear()
    _label_cache.clear()
    _signal_cache.clear()
    _backtest_cache.clear()
    return stats

