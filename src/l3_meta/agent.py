
import random
import uuid
import copy
from typing import List, Dict, Any

from src.contracts import PolicySpec, LedgerRecord, FeatureMetadata
from src.ledger.repo import LedgerRepo
from src.l3_meta.state import RegimeState
from src.features.definitions import INDICATOR_UNIVERSE # Keep for fallback mapping if needed
from src.features.registry import FeatureRegistry
from src.config import config
from src.shared.logger import get_logger
from src.l3_meta.q_learner import QLearner

logger = get_logger("meta.agent")

class MetaAgent:
    def __init__(self, registry: object, repo: LedgerRepo):
        self.registry = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH)) # Use real registry
        self.registry.initialize()
        self.repo = repo
        # Initialize RL Brain
        self.rl = QLearner(repo.base_dir)

    def propose_policy(self, regime: RegimeState, history: List[LedgerRecord]) -> PolicySpec:
        """
        [V3] Regime-Aware Policy Construction.
        1. Observe Regime (State)
        2. Select Strategy Archetype (Action) via Q-Learning
        3. Construct Genome (Implementation) - Regime Aware
        4. Define Risk Profiling - Regime Aware
        """
        logger.info(f"[MetaAgent] Regime Detected: {regime.label} (Trend: {regime.trend_score:.2f}, VIX: {regime.vol_level:.2f})")

        # Get Action from RL
        action_name, _ = self.rl.get_action(regime)
        
        # Build Genome based on Action AND Regime
        feature_genome = self._construct_genome_from_action(action_name, regime)
        
        return self._make_spec(feature_genome, action_name, regime)

    def learn(self, reward: float, next_regime: RegimeState):
        """
        Feedback loop for the RL agent.
        """
        self.rl.update(reward, next_regime)

    def _construct_genome_from_action(self, action_name: str, regime: RegimeState) -> Dict[str, Any]:
        """
        [V2 Evolution Logic]
        Decodes the high-level RL action into a specific set of indicators and parameters
        using the Dynamic Feature Registry.
        """
        # Map Action to Feature Category
        category_map = {
            "TREND_FOLLOWING": ["TREND"],
            "MEAN_REVERSION": ["MOMENTUM", "VOLUME"], 
            "VOLATILITY_BREAK": ["VOLATILITY", "TREND"],
            "MOMENTUM_ALPHA": ["MOMENTUM"],
            "DIP_BUYING": ["MOMENTUM", "TREND"], 
            "DEFENSIVE": ["VOLATILITY", "TREND"]
        }
        
        target_categories = category_map.get(action_name, ["TREND", "MOMENTUM"])
        
        # 1. Fetch Candidates from Registry
        candidates = []
        for cat in target_categories:
            candidates.extend(self.registry.list_by_category(cat))
            
        if not candidates:
            candidates = self.registry.list_all()
            
        # 2. Select Genome Structure
        k = random.randint(2, 4)
        selected_features = random.sample(candidates, min(k, len(candidates)))
        
        # 3. Parameter Evolution
        genome = {}
        for feature_meta in selected_features:
            params = self._evolve_params(feature_meta, action_name, regime)
            genome[feature_meta.feature_id] = params
            
        return genome

    def _evolve_params(self, feature_meta: FeatureMetadata, action_context: str, regime: RegimeState = None) -> Dict[str, Any]:
        """
        Generate parameters based on metadata schema, optionally biased by the strategy context.
        """
        params = {}
        
        for p_schema in feature_meta.params:
            if p_schema.param_type == "int":
                min_v, max_v = int(p_schema.min), int(p_schema.max)
                
                # Contextual Mutations - Strategy Type
                if action_context == "DEFENSIVE" and "window" in p_schema.name:
                    min_v = max(min_v, int(max_v * 0.5)) 
                elif action_context == "SCALPING" and "window" in p_schema.name:
                    max_v = min(max_v, 20)
                
                # Contextual Mutations - Regime
                if regime and regime.label == "PANIC" and "window" in p_schema.name:
                     # In Panic, cap max window to avoid too much lag
                     max_v = min(max_v, 30)

                # Clamp
                min_v = max(int(p_schema.min), min_v)
                max_v = min(int(p_schema.max), max_v)
                if min_v > max_v: min_v = max_v
                
                val = random.randint(min_v, max_v)
                
            elif p_schema.param_type == "float":
                val = random.uniform(p_schema.min, p_schema.max)
                if p_schema.step:
                    steps = round((val - p_schema.min) / p_schema.step)
                    val = p_schema.min + (steps * p_schema.step)
                val = round(val, 2)
            elif p_schema.param_type == "choice":
                if p_schema.choices:
                    val = random.choice(p_schema.choices)
                else:
                    val = p_schema.default
            else:
                val = p_schema.default
            params[p_schema.name] = val
            
        return params

    def _make_spec(self, genome: Dict[str, Any], template_tag: str, regime: RegimeState = None) -> PolicySpec:
        
        # [V2] Dynamic Risk Budgeting based on Macro Regime
        stop_loss = 0.02
        max_leverage = 1.0
        
        if regime:
            label = regime.label
            if label == "PANIC":
                stop_loss = 0.01  # Tighten Stop
                max_leverage = 0.5 # De-risk
            elif label == "STAGFLATION":
                stop_loss = 0.015
                max_leverage = 1.0
            elif label == "BULL_RUN":
                stop_loss = 0.025
                max_leverage = 1.25
            elif label == "GOLDILOCKS":
                stop_loss = 0.03
                max_leverage = 1.5
        
        return PolicySpec(
            spec_id=str(uuid.uuid4()),
            template_id=template_tag, 
            feature_genome=genome,
            tuned_params={}, 
            data_window={"lookback": 500},
            risk_budget={"stop_loss": stop_loss, "max_leverage": max_leverage},
            execution_assumption={"cost_bps": 5}
        )
