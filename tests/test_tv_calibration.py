import unittest
import pandas as pd

from src.backtest.tv_engine import TradingViewBacktestEngine
from src.testing.tv_calibration import compare_results


class TestTVCalibration(unittest.TestCase):
    def test_engine_next_open_entry(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0, 13.0],
                "high": [10.5, 11.5, 12.5, 13.5],
                "low": [9.5, 10.5, 11.5, 12.5],
                "close": [10.2, 11.2, 12.2, 13.2],
            },
            index=dates,
        )
        entry = pd.Series([False, True, False, False], index=dates)
        exit_sig = pd.Series([False, False, True, False], index=dates)

        engine = TradingViewBacktestEngine(execution_timing="next_open")
        result = engine.run(df, entry, exit_sig)

        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(trade.entry_bar_idx, 2)
        self.assertEqual(trade.exit_bar_idx, 3)
        self.assertAlmostEqual(trade.entry_price, 12.0, places=6)

    def test_compare_trade_count_mismatch(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0],
                "high": [10.5, 11.5, 12.5],
                "low": [9.5, 10.5, 11.5],
                "close": [10.2, 11.2, 12.2],
            },
            index=dates,
        )
        reference = {"trades": [{"entry_bar_idx": 1, "exit_bar_idx": 2, "entry_price": 11.0, "exit_price": 12.0}], "summary": {}}
        output = {"trades": [], "summary": {}}

        mismatches = compare_results(reference, output, df)
        self.assertTrue(any(m.code == "TRADE_COUNT_MISMATCH" for m in mismatches))


if __name__ == "__main__":
    unittest.main()
