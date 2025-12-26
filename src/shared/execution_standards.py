"""
Execution Standards Contract (V12.3+)
Defines the Single Source of Truth for Execution, Signal, and Indicator implementations.
This ensures parity between the RL Engine and TradingView (Pine Script).
"""
from enum import Enum
from dataclasses import dataclass

class ExecutionMode(Enum):
    NEXT_OPEN = "next_open"  # Default: Enter on Next Bar Open
    CLOSE = "close"          # Not recommended for realistic backtest
    
class SignalTiming(Enum):
    ON_CLOSE = "on_close"    # Signal confirms at Bar Close
    INTRA_BAR = "intra_bar"  # Signal can trigger during bar (High repainting risk)

class TPSLPriority(Enum):
    STOP_FIRST = "stop_first" # If Low < SL, exit SL. Then check TP.
    TP_FIRST = "tp_first"     # If High > TP, exit TP. Then check SL.

@dataclass
class ExecutionContract:
    execution_mode: ExecutionMode = ExecutionMode.NEXT_OPEN
    signal_timing: SignalTiming = SignalTiming.ON_CLOSE
    tpsl_priority: TPSLPriority = TPSLPriority.STOP_FIRST
    
    # Validation Tolerances
    tolerance_price_pct: float = 0.0005 # 0.05% price mismatch allowed
    tolerance_equity_pct: float = 0.005 # 0.5% equity mismatch allowed
    
    def validate_pine_script(self, pine_code: str) -> bool:
        """
        Rudimentary check to ensure Pine Script adheres to standards.
        """
        checks = {
            "calc_on_every_tick=false": "Signals must confirm on close",
            "process_orders_on_close=false": "Orders should typically process on next open (default)",
            "calc_on_order_fills=false": "Avoid recalc on fills for stability"
        }
        # This is a placeholder for actual validation logic
        return True

# Global Standard Instance
STANDARD = ExecutionContract()
