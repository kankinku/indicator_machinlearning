
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
from src.l3_meta.risk_profiles import RiskProfile, get_default_risk_profiles

logger = get_logger("meta.agent")

class MetaAgent:
    def __init__(self, registry: object, repo: LedgerRepo):
        self.registry = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH)) # Use real registry
        self.registry.initialize()
        self.repo = repo
        # Initialize RL Brains
        self.strategy_rl = QLearner(repo.base_dir)
        self.risk_profiles = get_default_risk_profiles()
        self.risk_profile_map = {profile.profile_id: profile for profile in self.risk_profiles}
        risk_actions = list(self.risk_profile_map.keys()) or ["DEFAULT"]
        self.risk_rl = QLearner(repo.base_dir, actions=risk_actions, table_name="q_table_risk.json")

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
        action_name, action_idx = self.strategy_rl.get_action(regime)
        risk_profile_id, risk_action_idx = self.risk_rl.get_action(regime)
        risk_profile = self._resolve_risk_profile(risk_profile_id)
        
        # Build Genome based on Action AND Regime
        feature_genome = self._construct_genome_from_action(action_name, regime)
        
        rl_meta = {
            "state_key": regime.label,
            "strategy_action": action_name,
            "strategy_action_idx": action_idx,
            "risk_profile": risk_profile_id,
            "risk_action_idx": risk_action_idx,
        }
        
        return self._make_spec(feature_genome, action_name, regime, risk_profile=risk_profile, rl_meta=rl_meta)

    def learn(self, reward: float, next_regime: RegimeState, policy_spec: PolicySpec = None):
        """
        Feedback loop for the RL agent.
        """
        if not policy_spec or not policy_spec.rl_meta:
            self.strategy_rl.update(reward, next_regime)
            self.risk_rl.update(reward, next_regime)
            return

        state_key = policy_spec.rl_meta.get("state_key")
        strategy_idx = policy_spec.rl_meta.get("strategy_action_idx")
        risk_idx = policy_spec.rl_meta.get("risk_action_idx")

        self.strategy_rl.update(reward, next_regime, state_key=state_key, action_idx=strategy_idx)
        if risk_idx is not None:
            self.risk_rl.update(reward, next_regime, state_key=state_key, action_idx=risk_idx)

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
        target_candidates = []
        for cat in target_categories:
            target_candidates.extend(self.registry.list_by_category(cat))
            
        # [V5 Autonomy Enforced]
        # We respect the RL's choice. If RL says "MOMENTUM", we give MOMENTUM.
        # We do NOT dilute it with random indicators from other categories just to fill a quota.
        
        selected_features = []
        if target_candidates:
            # Select a subset from the TARGET candidates only.
            # Example: If we have 10 momentum indicators, pick 1 to 5 of them.
            subset_size = random.randint(1, min(len(target_candidates), 5))
            selected_features = random.sample(target_candidates, subset_size)
        else:
            # Fallback: If registry is empty or category missing, pick 1 random from all.
            all_candidates = self.registry.list_all()
            if all_candidates:
                selected_features = random.sample(all_candidates, 1)
        
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

    def _make_spec(
        self,
        genome: Dict[str, Any],
        template_tag: str,
        regime: RegimeState = None,
        risk_profile: RiskProfile = None,
        rl_meta: Dict[str, Any] = None,
    ) -> PolicySpec:
        """
        [V5] Full Evolution Mode - Risk Parameters as Part of Genome.
        
        Instead of fixed strategy profiles, ALL risk parameters are randomly sampled
        from configured ranges. Good combinations are naturally selected through
        the reward mechanism (survival of the fittest).
        
        This allows the system to discover optimal risk/reward profiles without
        human bias or assumptions about what works for each strategy type.
        """
        
        # ========================================
        # 1. Sample Risk Parameters from Config Ranges
        # ========================================
        # These are completely independent of strategy type - pure evolution
        
        if risk_profile:
            k_up = random.uniform(*risk_profile.k_up_range)
            k_down = random.uniform(*risk_profile.k_down_range)
            horizon = random.randint(*risk_profile.horizon_range)
            risk_profile_id = risk_profile.profile_id
        else:
            # Fallback to full range if profile is missing
            k_up = random.uniform(config.RISK_K_UP_MIN, config.RISK_K_UP_MAX)
            k_down = random.uniform(config.RISK_K_DOWN_MIN, config.RISK_K_DOWN_MAX)
            horizon = random.randint(config.RISK_HORIZON_MIN, config.RISK_HORIZON_MAX)
            risk_profile_id = "DEFAULT"
        
        # ========================================
        # 2. Derived Parameters (Based on sampled values)
        # ========================================
        # Risk/Reward Ratio (for informational purposes)
        risk_reward_ratio = k_up / k_down if k_down > 0 else 1.0
        
        # Max Leverage: Inversely related to k_down (tighter stops = more leverage possible)
        # Logic: If k_down is low (tight stop), we can afford more leverage
        # Range: 0.5 to 1.5
        max_leverage = 0.5 + (1.0 / (k_down + 0.5))  # Inverted relationship
        max_leverage = max(0.5, min(1.5, max_leverage))
        
        # Stop Loss (Actual %): Derived from k_down and typical daily volatility (~1.5%)
        # This is an approximation; actual stop depends on market volatility
        estimated_daily_vol = 0.015  # ~1.5% typical daily vol
        stop_loss = k_down * estimated_daily_vol
        stop_loss = max(0.005, min(0.05, stop_loss))  # Clamp to 0.5% - 5%
        
        # ========================================
        # 3. Logging for Analysis
        # ========================================
        logger.debug(
            f"[PolicySpec] Generated Risk Params: "
            f"k_up={k_up:.2f}, k_down={k_down:.2f}, horizon={horizon}, "
            f"RR={risk_reward_ratio:.2f}, SL={stop_loss:.3f}"
        )
        
        # [V6] Autonomy - Entry Threshold Control
        # Instead of fixed global threshold, let the agent decide per strategy.
        # Range: 0.51 (Aggressive) to 0.70 (Conservative)
        entry_threshold = round(random.uniform(0.51, 0.70), 2)
        
        return PolicySpec(
            spec_id=str(uuid.uuid4()),
            template_id=template_tag,  # Strategy type from RL (for feature selection bias)
            feature_genome=genome,
            tuned_params={"entry_threshold": entry_threshold}, 
            data_window={"lookback": 500},
            risk_budget={
                # Core Triple Barrier Parameters (Used by run_experiment)
                "k_up": round(k_up, 2),
                "k_down": round(k_down, 2),
                "horizon": horizon,
                "risk_profile": risk_profile_id,
                # Derived/Informational
                "stop_loss": round(stop_loss, 4),
                "max_leverage": round(max_leverage, 2),
                "risk_reward_ratio": round(risk_reward_ratio, 2),
            },
            execution_assumption={"cost_bps": 5},
            rl_meta=rl_meta or {},
        )

    def _resolve_risk_profile(self, profile_id: str) -> RiskProfile:
        return self.risk_profile_map.get(profile_id)
