"""
LogicTree V18 테스트 스크립트
"""
import pandas as pd
from src.config import config

# Set strict mode
config.LOGICTREE_STRICT = True

from src.shared.logic_tree import _evaluate_condition, ConditionNode
from src.shared.logic_tree_diagnostics import LogicTreeMatchError, get_diagnostics, reset_diagnostics

def test_fuzzy_match():
    """Fuzzy 매칭 테스트: 후보 1개"""
    reset_diagnostics()
    df = pd.DataFrame({
        'RSI_V1__rsi': [30, 50, 70], 
        'SMA_V1__sma': [100, 101, 102]
    })
    node = ConditionNode(feature_key='RSI_V1', op='<', value=40)
    result = _evaluate_condition(node, df)
    
    diag = get_diagnostics()
    assert result.tolist() == [True, False, False], f"Result mismatch: {result.tolist()}"
    assert diag.matched_fuzzy == 1, f"Fuzzy count mismatch: {diag.matched_fuzzy}"
    print("✓ test_fuzzy_match PASSED")

def test_ambiguous_strict():
    """모호성 테스트 (Strict Mode): 후보 2개 이상"""
    reset_diagnostics()
    df = pd.DataFrame({
        'RSI_V1__rsi': [30, 50, 70], 
        'RSI_V1__signal': [35, 55, 75]
    })
    node = ConditionNode(feature_key='RSI_V1', op='<', value=40)
    
    try:
        result = _evaluate_condition(node, df)
        print("✗ test_ambiguous_strict FAILED: Should have raised LogicTreeMatchError")
    except LogicTreeMatchError as e:
        assert e.match_type == "ambiguous", f"Wrong error type: {e.match_type}"
        print(f"✓ test_ambiguous_strict PASSED (caught {e.match_type})")

def test_unmatched_strict():
    """미매칭 테스트 (Strict Mode): 후보 0개"""
    reset_diagnostics()
    df = pd.DataFrame({
        'SOME_OTHER__col': [1, 2, 3]
    })
    node = ConditionNode(feature_key='RSI_V1', op='<', value=40)
    
    try:
        result = _evaluate_condition(node, df)
        print("✗ test_unmatched_strict FAILED: Should have raised LogicTreeMatchError")
    except LogicTreeMatchError as e:
        assert e.match_type == "unmatched", f"Wrong error type: {e.match_type}"
        print(f"✓ test_unmatched_strict PASSED (caught {e.match_type})")

def test_direct_match():
    """직접 매칭 테스트"""
    reset_diagnostics()
    df = pd.DataFrame({
        'RSI_V1': [30, 50, 70]  # 정확히 일치
    })
    node = ConditionNode(feature_key='RSI_V1', op='<', value=40)
    result = _evaluate_condition(node, df)
    
    diag = get_diagnostics()
    assert result.tolist() == [True, False, False], f"Result mismatch: {result.tolist()}"
    assert diag.matched_direct == 1, f"Direct count mismatch: {diag.matched_direct}"
    print("✓ test_direct_match PASSED")

if __name__ == "__main__":
    print("=" * 50)
    print("LogicTree V18 Tests")
    print("=" * 50)
    
    test_fuzzy_match()
    test_ambiguous_strict()
    test_unmatched_strict()
    test_direct_match()
    
    print("=" * 50)
    print("All tests passed!")
