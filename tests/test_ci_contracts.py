
import unittest
import pandas as pd
from typing import Dict, Any
from src.features.registry import get_registry, FeatureMetadata
from src.shared.column_resolver import ColumnResolver, ColumnRef
from src.config import config
from src.shared.logic_tree import LogicTree, ConditionNode, LogicalOpNode

class TestCIContracts(unittest.TestCase):
    """
    CI Contract Tests
    Ensures critical system invariants are maintained before deployment/merge.
    """
    
    @classmethod
    def setUpClass(cls):
        # Load the REAL registry
        cls.registry = get_registry()
        cls.registry.initialize()
        
    def test_feature_registry_completeness(self):
        """
        Contract: All features must have explicit 'outputs' metadata.
        Feature factory relies on this for strictly mapped columns.
        """
        missing_outputs = []
        for meta in self.registry.list_all():
            if not meta.outputs:
                missing_outputs.append(meta.feature_id)
        
        self.assertFalse(missing_outputs, f"Features missing 'outputs' metadata: {missing_outputs}. This violates SSOT.")

    def test_column_resolution_determinism(self):
        """
        Contract: ColumnResolver must resolve standard output keys deterministically 
        given the mocked standard column names.
        """
        # Mock DataFrame columns based on what FeatureFactory produces
        # Convention: {FeatureID}__{OutputKey}
        mock_columns = []
        feature_map = {}
        
        for meta in self.registry.list_all():
            fid = meta.feature_id
            for key, alias in meta.outputs.items():
                col_name = f"{fid}__{alias}"
                mock_columns.append(col_name)
                feature_map[(fid, key)] = col_name

        resolver = ColumnResolver(mock_columns)
        
        failures = []
        for (fid, key), expected_col in feature_map.items():
            ref = ColumnRef(feature_id=fid, output_key=key)
            try:
                resolved = resolver.resolve(ref)
                if resolved != expected_col:
                    failures.append(f"{fid}.{key} -> Resolved {resolved} != Expected {expected_col}")
            except Exception as e:
                failures.append(f"{fid}.{key} -> Resolution Error: {e}")
                
        self.assertFalse(failures, f"Column Resolution Contract Failed:\n" + "\n".join(failures))

    def test_logic_tree_schema_validation(self):
        """
        Contract: LogicTree validation mechanism must reject invalid schemas.
        """
        # Valid Node
        valid_node = ConditionNode(feature_key="RSI_V1", op=">", value=50) # output_key removed
        valid_tree = LogicTree(root=valid_node)
        self.assertTrue(valid_tree.validate(), "Valid LogicTree rejected validation")
        
        # Invalid Node (Missing operator)
        invalid_node = ConditionNode(feature_key="RSI_V1", op="", value=50) # Missing op
        invalid_tree = LogicTree(root=invalid_node)
        self.assertFalse(invalid_tree.validate(), "Invalid LogicTree passed validation (missing op)")
        
    def test_reward_mechanics_integrity(self):
        """
        Contract: RewardShaper must produce consistent breakdown and score.
        """
        from src.l3_meta.reward_shaper import get_reward_shaper
        shaper = get_reward_shaper()
        
        # Test Case 1: Minimal Activity (Penalty expected)
        metrics_low = {
            "total_return_pct": 0.0,
            "win_rate": 0.0,
            "n_trades": 0,
            "mdd_pct": 0.0,
            "reward_risk": 0.0
        }
        breakdown_low = shaper.compute_breakdown(metrics_low)
        self.assertLess(breakdown_low.total, -10.0, "Zero activity did not trigger sufficient penalty")
        
        # Test Case 2: Good Performance
        metrics_high = {
            "total_return_pct": 30.0,
            "win_rate": 0.6,
            "n_trades": 50,
            "mdd_pct": 10.0,
            "reward_risk": 1.5,
            # Implicit Regime Trade alignment assumes 100% if not passed? 
            # Actually RewardShaper implementation defaults regime_alignment to 0 if missing? 
            # Let's check logic: r_regime uses 'valid_trade_count' / 'n_trades'. 
            "valid_trade_count": 40,
            "regime_aligned_trades": 40,
            "exposure_ratio": 0.5,
            "oos_bars": 252,
            "profit_factor": 2.0,
            "avg_trade_return": 1.0,
            "trades_per_year": 50.0  # Added to pass validation
        }
        breakdown_high = shaper.compute_breakdown(metrics_high)
        self.assertGreater(breakdown_high.total, 0.0, "Good performance yield negative score")
        self.assertGreater(breakdown_high.regime_trade_component, 0, "Regime alignment component failed")

if __name__ == "__main__":
    unittest.main()
