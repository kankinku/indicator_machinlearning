import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backtest.engine import DeterministicBacktestEngine
from src.orchestration.policy_evaluator import RuleEvaluator
from src.contracts import PolicySpec

class TestV13ProCore(unittest.TestCase):
    def setUp(self):
        # Create dummy data: 10 days of prices
        # Day 0 to 9
        self.dates = pd.date_range("2023-01-01", periods=10)
        self.data = pd.DataFrame({
            "close": [100.0, 105.0, 110.0, 108.0, 102.0, 110.0, 115.0, 120.0, 118.0, 125.0],
            "feature_a": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "feature_b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }, index=self.dates)
        
        self.evaluator = RuleEvaluator()
        self.engine = DeterministicBacktestEngine()

    def test_rule_evaluator_basic(self):
        """Test basic rule evaluation and complexity."""
        spec = PolicySpec(
            spec_id="test",
            template_id="v13",
            decision_rules={
                "entry": "feature_a > 25",
                "exit": "feature_a > 75"
            }
        )
        entry_sig, exit_sig, complexity = self.evaluator.evaluate_signals(self.data, spec)
        
        # feature_a > 25 starts at index 2 (val 30)
        self.assertTrue(entry_sig.iloc[2])
        self.assertFalse(entry_sig.iloc[1])
        
        # feature_a > 75 starts at index 7 (val 80)
        self.assertTrue(exit_sig.iloc[7])
        self.assertFalse(exit_sig.iloc[6])
        
        # Token count: 3 tokens each ('feature_a', '>', '25') -> 6 total
        # Implementation uses: complexity = len(tokens) * 0.1 -> 0.6
        self.assertAlmostEqual(complexity, 0.6)

    def test_rule_evaluator_nan_safety(self):
        """Test that NaN values result in False for rules."""
        data_with_nan = self.data.copy()
        data_with_nan.loc[self.dates[2], "feature_a"] = np.nan
        
        spec = PolicySpec(
            spec_id="nan_test",
            template_id="v13",
            decision_rules={"entry": "feature_a > 0"}
        )
        entry_sig, _, _ = self.evaluator.evaluate_signals(data_with_nan, spec)
        
        self.assertFalse(entry_sig.iloc[2], "NaN should not satisfy > 0")

    def test_backtest_timing_and_execution(self):
        """
        Test timing alignment:
        Signal[i] (Close[i]) -> Position[i] -> Profit from returns[i+1]
        """
        # Manual signals: 1 for entry at index 1, 1 for exit at index 4
        entry_signals = pd.Series(False, index=self.dates)
        exit_signals = pd.Series(False, index=self.dates)
        
        entry_signals.iloc[1] = True # Signal at Close[1] (105)
        exit_signals.iloc[4] = True  # Signal at Close[4] (102)
        
        res = self.engine.run(self.data["close"], entry_signals, exit_signals)
        
        # Entry at Close[1]=105, Exit at Close[4]=102.
        # Days invested: Close[1] to Close[2], Close[2] to Close[3], Close[3] to Close[4].
        # Return = (102 / 105 - 1) * 100
        expected_ret = (102/105 - 1) * 100
        self.assertAlmostEqual(res.total_return, expected_ret, places=4)
        
        # Trade list check
        self.assertEqual(len(res.trades), 1)
        trade = res.trades[0]
        self.assertEqual(trade["entry_price"], 105.0)
        self.assertEqual(trade["exit_price"], 102.0)

    def test_backtest_simultaneous_signals(self):
        """Test that exit priority works when both entry and exit are True."""
        entry_signals = pd.Series(False, index=self.dates)
        exit_signals = pd.Series(False, index=self.dates)
        
        entry_signals.iloc[1] = True
        exit_signals.iloc[1] = True # Simultaneous
        
        res = self.engine.run(self.data["close"], entry_signals, exit_signals)
        
        # Starting FLAT, Exit has priority but we are FLAT, so we stay FLAT.
        # No entry should occur.
        self.assertEqual(len(res.trades), 0)

    def test_backtest_metrics(self):
        """Test profit factor and win rate."""
        # Create a winning trade and a losing trade
        entry = pd.Series(False, index=self.dates)
        exit = pd.Series(False, index=self.dates)
        
        # Trade 1: Entry index 0 (100), Exit 2 (110) -> Gain 10%
        entry.iloc[0] = True
        exit.iloc[2] = True
        
        # Trade 2: Entry index 7 (120), Exit 8 (118) -> Loss ~1.66%
        entry.iloc[7] = True
        exit.iloc[8] = True
        
        res = self.engine.run(self.data["close"], entry, exit)
        
        # Win Rate: 1 win, 1 loss -> 0.5 (Engine uses 0.0 to 1.0)
        self.assertEqual(res.win_rate, 0.5)
        
        # Gross profit = 10%
        # Gross loss = abs(118/120 - 1) * 100 = 1.666...
        # PF = 10 / 1.666... = 6.0
        self.assertAlmostEqual(res.profit_factor, 6.0)

if __name__ == "__main__":
    unittest.main()
