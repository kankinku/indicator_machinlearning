
import unittest
import numpy as np
from src.config import config
from src.l3_meta.reward_shaper import RewardShaper, RewardBreakdown
from src.l3_meta.analyst import IndicatorPriorController
from src.l3_meta.replay_buffer import IntegratedTaggedReplayBuffer, MultiActionExperience
from src.shared.execution_standards import STANDARD, ExecutionMode, TPSLPriority

class TestV12_3_Upgrades(unittest.TestCase):
    
    def test_execution_standards(self):
        """Test Execution Standards Contract"""
        self.assertEqual(STANDARD.execution_mode, ExecutionMode.NEXT_OPEN)
        self.assertEqual(STANDARD.tpsl_priority, TPSLPriority.STOP_FIRST)
        self.assertTrue(STANDARD.validate_pine_script("calc_on_every_tick=false"))

    def test_reward_shaper_alpha_saturation(self):
        """Test Tanh Saturation for Alpha Return"""
        shaper = RewardShaper()
        # Scale = 100.0 (Default)
        
        # Valid metrics base
        base_metrics = {
            "n_trades": 30, 
            "trades_per_year": 30.0,
            "win_rate": 0.5, 
            "mdd_pct": 10.0,
            "exposure_ratio": 0.1,
            "oos_bars": 252
        }

        metrics_low = base_metrics.copy()
        metrics_low.update({"excess_return": 10.0, "total_return_pct": 15.0})
        bd_low = shaper.compute_breakdown(metrics_low)
        
        metrics_high = base_metrics.copy()
        metrics_high.update({"excess_return": 200.0, "total_return_pct": 200.0})
        bd_high = shaper.compute_breakdown(metrics_high)
        
        # Check if they passed
        self.assertFalse(bd_low.is_rejected, f"Low rejected: {bd_low.rejection_reason}")
        self.assertFalse(bd_high.is_rejected, f"High rejected: {bd_high.rejection_reason}")

        self.assertTrue(bd_high.return_component > bd_low.return_component)
        self.assertTrue(bd_high.return_component < 3.05) # Cap check

    def test_reward_shaper_stage1_oxygen(self):
        """Test Stage 1 Trade Activity Boost"""
        # Set Stage 1
        config.CURRICULUM_CURRENT_STAGE = 1
        
        pass_trades = 20 # > 15 but < 200 (Target)
        shaper = RewardShaper()
        
        metrics = {
            "n_trades": pass_trades,
            "trades_per_year": float(pass_trades),
            "win_rate": 0.5,
            "mdd_pct": 10.0,
            "exposure_ratio": 0.1,
            "total_return_pct": 10.0,
             "oos_bars": 252
        }
        bd = shaper.compute_breakdown(metrics)
        self.assertFalse(bd.is_rejected, f"Rejected: {bd.rejection_reason}")
        
        # Expected: sqrt(20/200) = sqrt(0.1) ~ 0.316
        # Linear would be 0.1
        self.assertGreater(bd.trades_component, 0.2) # Should be boosted above linear

        # Reset Stage
        config.CURRICULUM_CURRENT_STAGE = 1 

    def test_ontology_calibration(self):
        """Test Ontology Calibration Logic"""
        # Create temp analyst
        import shutil
        from pathlib import Path
        temp_dir = Path("temp_test_analyst")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            analyst = IndicatorPriorController(temp_dir)
            self.assertEqual(analyst.get_calibration_score(), 0.5)
            
            # Simulate a record that matches priors (suppose priors are uniform init)
            # Init: all cats 1/8 = 0.125
            # Record: Importance {"TREND": 1.0} -> Normalized: TREND=1.0, others=0.0
            # Alignment should be positive but noisy
            record = {"trade_logic": {"feature_importance": {"TREND__ema": 1.0}}}
            # update_with_record needs registry to map feature_id to category
            # Let's mock registry
            class MockRegistry:
                def get(self, fid):
                    class Meta:
                         category = "TREND"
                    return Meta()
            
            analyst.update_with_record(record, "BULL", MockRegistry())
            
            # Score should change
            new_score = analyst.get_calibration_score()
            self.assertNotEqual(new_score, 0.5)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_integrated_tagged_buffer(self):
        """Test Integrated Tagged Replay Buffer"""
        buf = IntegratedTaggedReplayBuffer(capacity=10, batch_size=2)
        
        # Push 2 samples
        s = np.array([1,2])
        ns = np.array([2,3])
        buf.push_transition(s, [0, 1], 1.0, ns, False) # Tag PASS default? push_transition calls push(exp)
        # TaggedReplayBuffer.push defaults to PASS
        
        buf.push_transition(s, [1, 0], -1.0, ns, True)
        
        self.assertEqual(len(buf), 2)
        
        batch = buf.sample()
        self.assertEqual(len(batch.actions_list), 2) # 2 heads
        self.assertEqual(batch.rewards.shape[0], 2) # batch size 2

if __name__ == '__main__':
    unittest.main()
