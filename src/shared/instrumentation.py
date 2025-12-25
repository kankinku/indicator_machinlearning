
import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path
from src.shared.logger import get_logger

logger = get_logger("shared.instrumentation")

@dataclass
class BatchMetrics:
    batch_id: int
    timestamp: float
    seed: int
    n_policies: int
    
    # Timing
    total_time: float = 0.0
    stage1_time: float = 0.0
    stage2_time: float = 0.0
    overhead_time: float = 0.0
    
    # Pass Rates
    pass_rate_s1: float = 0.0
    pass_rate_s2: float = 0.0
    
    # Quality (Top-K)
    best_score: float = -9999.0
    mean_topk_score: float = 0.0
    median_topk_score: float = 0.0
    
    # Trades
    mean_trades_topk: float = 0.0
    zero_trade_ratio: float = 0.0
    
    # Diversity
    diversity_mean_jaccard: float = 0.0
    duplicate_ratio: float = 0.0

    # Exploration
    cold_start_ratio: float = 0.0
    
    # Stability
    exceptions: Dict[str, int] = field(default_factory=dict)
    permission_errors: int = 0
    
    def to_jsonl(self) -> str:
        return json.dumps(asdict(self))

class BatchInstrumentation:
    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.report_dir / "batch_reports.jsonl"
        self.current_metrics: Optional[BatchMetrics] = None
        self._batch_start_time = 0.0

    def start_batch(self, batch_id: int, seed: int, n_policies: int):
        self.current_metrics = BatchMetrics(
            batch_id=batch_id,
            timestamp=time.time(),
            seed=seed,
            n_policies=n_policies
        )
        self._batch_start_time = time.time()
        logger.info(f"=== [INSTRUMENTATION] Batch {batch_id} Started (Seed: {seed}, Pols: {n_policies}) ===")

    def end_batch(self):
        if not self.current_metrics:
            return
            
        self.current_metrics.total_time = time.time() - self._batch_start_time
        self.current_metrics.overhead_time = self.current_metrics.total_time - (
            self.current_metrics.stage1_time + self.current_metrics.stage2_time
        )
        
        # Save to file
        with open(self.history_file, "a") as f:
            f.write(self.current_metrics.to_jsonl() + "\n")
            
        logger.info(f"=== [INSTRUMENTATION] Batch {self.current_metrics.batch_id} End: {self.current_metrics.total_time:.2f}s (S1: {self.current_metrics.stage1_time:.2f}s, S2: {self.current_metrics.stage2_time:.2f}s, OH: {self.current_metrics.overhead_time:.2f}s) ===")
        self.current_metrics = None

    def record_stage1(self, duration: float, pass_rate: float):
        if self.current_metrics:
            self.current_metrics.stage1_time = duration
            self.current_metrics.pass_rate_s1 = pass_rate

    def record_stage2(self, duration: float, pass_rate: float):
        if self.current_metrics:
            self.current_metrics.stage2_time = duration
            self.current_metrics.pass_rate_s2 = pass_rate

    def record_quality(self, best_score: float, mean_topk: float, median_topk: float):
        if self.current_metrics:
            self.current_metrics.best_score = best_score
            self.current_metrics.mean_topk_score = mean_topk
            self.current_metrics.median_topk_score = median_topk

    def record_trades(self, mean_trades: float, zero_ratio: float):
        if self.current_metrics:
            self.current_metrics.mean_trades_topk = mean_trades
            self.current_metrics.zero_trade_ratio = zero_ratio

    def record_diversity(self, mean_jaccard: float, duplicate_ratio: float):
        if self.current_metrics:
            self.current_metrics.diversity_mean_jaccard = mean_jaccard
            self.current_metrics.duplicate_ratio = duplicate_ratio

    def record_exploration(self, cold_start_ratio: float):
        if self.current_metrics:
            self.current_metrics.cold_start_ratio = cold_start_ratio

    def record_exception(self, exc_type: str):
        if self.current_metrics:
            self.current_metrics.exceptions[exc_type] = self.current_metrics.exceptions.get(exc_type, 0) + 1
            if exc_type == "PermissionError":
                self.current_metrics.permission_errors += 1

# Singleton access
_inst = None
def get_instrumentation(report_dir: Optional[Path] = None) -> BatchInstrumentation:
    global _inst
    if _inst is None:
        from src.config import config
        r_dir = report_dir or config.LOG_DIR / "instrumentation"
        _inst = BatchInstrumentation(r_dir)
    return _inst
