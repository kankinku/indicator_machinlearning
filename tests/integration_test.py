
"""
통합 테스트 - 전체 파이프라인 검증

이 테스트는 다음을 검증합니다:
1. DataLoader - 실제 데이터 로드
2. FeatureRegistry 싱글톤 - 중복 초기화 방지
3. PolicySpec - Genome 기반 정책 생성
4. run_experiment - 전체 실험 파이프라인
"""
import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestration.run_experiment import run_experiment
from src.templates.registry import default_registry
from src.contracts import PolicySpec
from src.ledger.repo import LedgerRepo
from src.data.loader import DataLoader
from src.features.registry import get_registry  # 싱글톤 사용

def test_pipeline():
    print(">>> 1. Fetching Real Data (QQQ) via DataLoader...")
    # Fetch ample history to test Macro/VIX joins
    loader = DataLoader("QQQ", "2018-01-01") 
    df = loader.fetch_all()
    
    if df.empty:
        raise ValueError("Failed to fetch data for integration test.")
        
    print(f"Data Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print(">>> 2. Initializing Registry & Repo...")
    # 싱글톤 레지스트리 사용 (DI 적용)
    registry = get_registry()
    repo = LedgerRepo(Path("./test_ledger"))
    
    # Select Template T01
    template_id = "T01"
    
    print(f">>> 3. Creating PolicySpec for {template_id}...")
    
    # 레지스트리에서 피처 선택
    available_features = registry.list_all()
    if not available_features:
        raise ValueError("No features available in registry")
    
    # 테스트용 Genome 생성 (RSI, MACD 등 기본 지표 사용)
    test_genome = {}
    feature_names_to_use = ["RSI", "MACD", "BB"]
    for feat in available_features:
        if any(name in feat.feature_id for name in feature_names_to_use):
            # 기본 파라미터 사용
            test_genome[feat.feature_id] = {
                p.name: p.default if p.default is not None 
                else (int((p.min + p.max) / 2) if p.param_type == "int" else (p.min + p.max) / 2)
                for p in feat.params
            }
    
    if not test_genome:
        # 폴백: 첫 번째 피처 사용
        feat = available_features[0]
        test_genome[feat.feature_id] = {
            p.name: p.default if p.default is not None
            else (int((p.min + p.max) / 2) if p.param_type == "int" else (p.min + p.max) / 2)
            for p in feat.params
        }
    
    print(f"    Test Genome: {list(test_genome.keys())}")
    
    # PolicySpec 생성 (필수 인수 포함)
    policy = PolicySpec(
        spec_id="test_spec_001",
        template_id=template_id,
        feature_genome=test_genome,  # 필수 인수
        tuned_params={"entry_threshold": 0.6},
        data_window={"lookback": 500},
        risk_budget={
            "k_up": 1.5,
            "k_down": 1.0,
            "horizon": 20,
            "stop_loss": 0.02,
            "max_leverage": 1.0,
        },
        execution_assumption={"cost_bps": 5}
    )
    
    print(">>> 4. Running Experiment (CPCV Loop)...")
    try:
        # run_experiment는 (LedgerRecord, ArtifactBundle) tuple을 반환
        record, artifact = run_experiment(registry, policy, df, repo)
        
        print("\n>>> 5. Success! Verdict:")
        print(f"Approved: {record.reason_codes == []}")
        print(f"Reason Codes: {record.reason_codes}")
        print(f"Scorecard: {record.cpcv_metrics}")
        print(f"PBO: {record.pbo}")
        print(f"Artifact Ref: {record.model_artifact_ref}")
        
    except Exception as e:
        print(f"\n!!! FAILURE !!!")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()

