from typing import Dict, List, Set, Any
from dataclasses import dataclass, field

@dataclass
class FeatureRelation:
    feature_id: str
    complements: List[str] = field(default_factory=list)  # 함께 쓰면 시너지가 나는 피처
    conflicts: List[str] = field(default_factory=list)     # 함께 쓰면 신호가 오염되는 피처 (배타적)
    required_regimes: List[str] = field(default_factory=list) # 특정 국면에서만 사용 권장

class FeatureOntology:
    """
    [Genome v2] L0.5 Layer: Feature Relationship Network.
    This layer defines static 'common sense' for features.
    """
    
    # 1. Market Interrogation Questions
    MARKET_QUESTIONS = {
        "OVERHEATED": ["MOMENTUM_RSI_V1", "MEANREV_STOCH_V1", "VOLATILITY_BB_V1"],
        "STRUCTURAL_ENERGY": ["VOLATILITY_ATR_V1", "VOLATILITY_KC_V1", "VOLATILITY_BB_V1"],
        "TREND_CONFIRMATION": ["TREND_ADX_V1", "VOLUME_OBV_V1", "MOMENTUM_MACD_V1"],
        "EXTREME_REVERSAL": ["PA_FAKE_OUT_V1", "MOMENTUM_RSI_V1", "VOLATILITY_BB_V1"],
        "VOLATILITY_EXPANSION": ["VOLATILITY_ATR_V1", "TREND_ADX_V1"]
    }

    def __init__(self):
        self.relations: Dict[str, FeatureRelation] = {}
        self._initialize_relations()

    def _initialize_relations(self):
        # 1. RSI (Momentum/Exhaustion)
        self.relations["MOMENTUM_RSI_V1"] = FeatureRelation(
            feature_id="MOMENTUM_RSI_V1",
            complements=["VOLATILITY_BB_V1", "PA_FAKE_OUT_V1"], 
            conflicts=["TREND_ADX_V1"], 
            required_regimes=["SIDEWAYS", "HIGH_VOL"]
        )

        # 2. ATR (Volatility/Energy)
        self.relations["VOLATILITY_ATR_V1"] = FeatureRelation(
            feature_id="VOLATILITY_ATR_V1",
            complements=["VOLATILITY_BB_V1", "PA_FAKE_OUT_V1", "VOLATILITY_KC_V1"],
            required_regimes=["LOW_VOL", "BREAKOUT"]
        )

        # 3. BB (Volatility/Distribution)
        self.relations["VOLATILITY_BB_V1"] = FeatureRelation(
            feature_id="VOLATILITY_BB_V1",
            complements=["MOMENTUM_RSI_V1", "VOLATILITY_ATR_V1", "VOLATILITY_KC_V1"],
            required_regimes=["SIDEWAYS", "BREAKOUT"]
        )

        # 4. Fakeout (Price Action)
        self.relations["PA_FAKE_OUT_V1"] = FeatureRelation(
            feature_id="PA_FAKE_OUT_V1",
            complements=["VOLATILITY_BB_V1", "MOMENTUM_RSI_V1", "VOLUME_OBV_V1"],
            required_regimes=["BREAKOUT", "HIGH_VOL"]
        )

        # 5. ADX (Trend Strength)
        self.relations["TREND_ADX_V1"] = FeatureRelation(
            feature_id="TREND_ADX_V1",
            complements=["TREND_MACROSS_V1", "TREND_SUPER_V1", "MOMENTUM_MACD_V1"],
            conflicts=["MOMENTUM_RSI_V1", "MEANREV_STOCH_V1"],
            required_regimes=["TREND_UP", "TREND_DOWN"]
        )

        # 6. MACD (Trend/Momentum Blend)
        self.relations["MOMENTUM_MACD_V1"] = FeatureRelation(
            feature_id="MOMENTUM_MACD_V1",
            complements=["VOLUME_OBV_V1", "TREND_ADX_V1"],
            conflicts=["VOLATILITY_BB_V1"],
            required_regimes=["TREND_UP", "TREND_DOWN", "BREAKOUT"]
        )

        # 7. Volume Patterns (OBV)
        self.relations["VOLUME_OBV_V1"] = FeatureRelation(
            feature_id="VOLUME_OBV_V1",
            complements=["MOMENTUM_MACD_V1", "PA_FAKE_OUT_V1"],
            required_regimes=["BREAKOUT", "TREND_UP"]
        )

        # 8. Keltner Channels (Volatility/Trend)
        self.relations["VOLATILITY_KC_V1"] = FeatureRelation(
            feature_id="VOLATILITY_KC_V1",
            complements=["VOLATILITY_BB_V1"],
            required_regimes=["LOW_VOL", "BREAKOUT"]
        )

        # 9. Ichimoku (Structural Trend)
        self.relations["TREND_ICHIMOKU_V1"] = FeatureRelation(
            feature_id="TREND_ICHIMOKU_V1",
            complements=["TREND_ADX_V1", "VOLUME_MFI_V1"],
            conflicts=["MEANREV_STOCH_V1"],
            required_regimes=["TREND_UP", "TREND_DOWN"]
        )

    def get_relation(self, feature_id: str) -> FeatureRelation:
        return self.relations.get(feature_id, FeatureRelation(feature_id=feature_id))

    def evaluate_combination(self, feature_list: List[str], current_regime_label: str = None) -> Dict[str, Any]:
        """
        [Genome v2] 지능적 조합 평가. 
        """
        score = 0.0
        n = len(feature_list)
        conflicts = []
        complements = []
        
        if n < 2: return {"score": 0.0, "status": "NEUTRAL"}

        for i in range(n):
            r1 = self.get_relation(feature_list[i])
            for j in range(i + 1, n):
                r2 = self.get_relation(feature_list[j])

                # 1. 보완성 체크
                if feature_list[j] in r1.complements or feature_list[i] in r2.complements:
                    score += 0.8
                    complements.append((feature_list[i], feature_list[j]))
                
                # 2. 충돌 체크 (Critical)
                if feature_list[j] in r1.conflicts or feature_list[i] in r2.conflicts:
                    score -= 1.5
                    conflicts.append((feature_list[i], feature_list[j]))
                
                # 3. 범주간 중복 체크 (Redundancy)
                f1_cat = r1.feature_id.split('_')[0]
                f2_cat = r2.feature_id.split('_')[0]
                if f1_cat == f2_cat:
                    score -= 0.3

        # 최종 점수 정규화
        final_score = max(-1.0, min(1.0, score / n))
        
        status = "HEALTHY"
        if final_score < -0.2: status = "CONFLICTED"
        if final_score > 0.4: status = "SYNERGETIC"

        return {
            "score": final_score,
            "status": status,
            "conflicts": conflicts,
            "complements": complements
        }

    def check_compatibility(self, feature_list: List[str]) -> float:
        """Legacy interface for DiversitySampler"""
        res = self.evaluate_combination(feature_list)
        return res["score"]

_ontology_instance = None
def get_feature_ontology():
    global _ontology_instance
    if _ontology_instance is None:
        _ontology_instance = FeatureOntology()
    return _ontology_instance
