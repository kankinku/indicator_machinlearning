import hashlib
import json
from typing import Optional, Dict, Tuple
from src.evaluation.contracts import ModuleResult
from src.contracts import PolicySpec
from src.shared.logger import get_logger

logger = get_logger("evaluation.result_store")

def get_policy_sig(policy: PolicySpec) -> str:
    """[V15] Returns a stable MD5 signature for a PolicySpec."""
    # We focus on the logic parts that affect backtest results
    data = {
        "genome": policy.feature_genome,
        "risk": policy.risk_budget,
        "rules": policy.decision_rules,
        "logic_trees": getattr(policy, "logic_trees", {}),
        "execution": policy.execution_assumption,
        "data_window": policy.data_window,
        "tuned": policy.tuned_params,
    }
    dump = json.dumps(data, sort_keys=True)
    return hashlib.md5(dump.encode()).hexdigest()

class ResultStore:
    """
    [V15] SSOT Result Store
    Prevents duplicate evaluation of the same policy on the same data window.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResultStore, cls).__new__(cls)
            cls._instance.cache: Dict[Tuple[str, str], ModuleResult] = {}
        return cls._instance

    def get(self, policy_sig: str, window_sig: str) -> Optional[ModuleResult]:
        key = (policy_sig, window_sig)
        return self.cache.get(key)

    def put(self, policy_sig: str, window_sig: str, result: ModuleResult):
        key = (policy_sig, window_sig)
        if key in self.cache:
            logger.warning(f"[ResultStore] 기존 결과 덮어씀: {key} (중복 저장 주의)")
        self.cache[key] = result

    def clear(self):
        self.cache.clear()

def get_result_store() -> ResultStore:
    return ResultStore()
