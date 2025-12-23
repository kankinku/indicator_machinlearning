
import random
import uuid
import copy
from typing import List, Dict, Any, Optional
import pandas as pd
import json

from src.contracts import PolicySpec, LedgerRecord, FeatureMetadata
from src.ledger.repo import LedgerRepo
from src.l3_meta.state import RegimeState
from src.features.definitions import INDICATOR_UNIVERSE # Keep for fallback mapping if needed
from src.features.registry import get_registry
from src.config import config
from src.shared.logger import get_logger
from src.l3_meta.q_learner import QLearner
from src.l3_meta.risk_profiles import RiskProfile, get_default_risk_profiles

# D3QN 조건부 임포트
if config.D3QN_ENABLED:
    from src.l3_meta.d3qn_agent import D3QNAgent, create_rl_agent
    from src.l3_meta.reward_shaper import get_reward_shaper

from src.l3_meta.curriculum_controller import get_curriculum_controller, CurriculumController
from src.l3_meta.analyst import get_indicator_analyst

logger = get_logger("meta.agent")

class MetaAgent:
    def __init__(self, registry: object, repo: LedgerRepo):
        # DI: 외부에서 주입받거나 싱글톤 사용
        self.registry = get_registry()
        self.repo = repo
        
        # [V11.4] Integrated Multi-head D3QN 모드
        if config.D3QN_ENABLED:
            logger.info("[MetaAgent] Integrated Multi-head D3QN 모드로 초기화")
            from src.l3_meta.d3qn_agent import get_integrated_agent, DEFAULT_ACTIONS
            self.integrated_rl = get_integrated_agent(
                repo.base_dir,
                strategy_actions=DEFAULT_ACTIONS,
                risk_actions=list(self._get_risk_profile_ids())
            )
            self.reward_shaper = get_reward_shaper()
        else:
            logger.info("[MetaAgent] Q-Table 모드로 초기화")
            from src.l3_meta.q_learner import QLearner
            self.strategy_rl = QLearner(repo.base_dir)
            risk_actions = list(self._get_risk_profile_ids()) or ["DEFAULT"]
            self.risk_rl = QLearner(repo.base_dir, actions=risk_actions, table_name="q_table_risk.json")
            self.reward_shaper = None
        
        # [V11.4] Indicator Prior Analyst
        self.analyst = get_indicator_analyst(repo.base_dir)
        
        self.curriculum = get_curriculum_controller()
        self.risk_profiles = get_default_risk_profiles()
        self.risk_profile_map = {profile.profile_id: profile for profile in self.risk_profiles}
        
        # [V11.3] Warm-start Baselines
        self.baselines = self._load_baselines()
        
        # 현재 시장 데이터 참조 (D3QN용)
        self._current_df: Optional[pd.DataFrame] = None
    
    def _load_baselines(self) -> List[Dict[str, Any]]:
        """data/baselines/*.json 파일을 로드합니다."""
        baselines = []
        path = config.WARM_START_BASE_DIR
        if not path.exists():
            return []
        
        for f in path.glob("*.json"):
            try:
                with open(f, "r") as r:
                    baselines.append(json.load(r))
            except Exception as e:
                logger.error(f"Failed to load baseline {f}: {e}")
        
        logger.info(f"[MetaAgent] {len(baselines)} baselines loaded from {path}")
        return baselines

    def _get_risk_profile_ids(self) -> List[str]:
        """리스크 프로파일 ID 목록을 반환합니다."""
        profiles = get_default_risk_profiles()
        return [p.profile_id for p in profiles] or ["DEFAULT"]
    
    def set_market_data(self, df: pd.DataFrame) -> None:
        """현재 시장 데이터를 설정합니다 (D3QN 상태 인코딩용)."""
        self._current_df = df
    def propose_policy(self, regime: RegimeState, history: List[LedgerRecord]) -> PolicySpec:
        """
        [V11.3] Warm-start Integrated Policy Construction.
        """
        # 0. Mix with Baselines if in Warm-start period
        total_exp = len(history)
        baseline_prob = 0.0
        
        if total_exp < config.WARM_START_N1:
            baseline_prob = 0.5
        elif total_exp < config.WARM_START_N2:
            baseline_prob = 0.3
        else:
            baseline_prob = 0.1 # Keep small amount of baseline for prior stability
            
        if self.baselines and random.random() < baseline_prob:
            base = random.choice(self.baselines)
            logger.debug(f"[MetaAgent] Warm-start: Using baseline '{base.get('name')}'")
            return self._make_spec(
                genome=base["feature_genome"],
                template_tag=base["template_id"],
                regime=regime,
                # risk_profile is optional in baseline, but we can override or use it
                rl_meta={
                    "is_baseline": True,
                    "baseline_name": base.get("name"),
                    "state_key": regime.label,
                }
            )

        # 1. Standard RL Flow
        logger.debug(f"[MetaAgent] Regime Detected: {regime.label} (Trend: {regime.trend_score:.2f}, VIX: {regime.vol_level:.2f})")

        # Get Action from RL (Multi-head Integrated)
        if config.D3QN_ENABLED:
            action_name, action_idx, risk_profile_id, risk_action_idx = self.integrated_rl.get_action(
                regime, df=self._current_df
            )
        else:
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
            "d3qn_mode": config.D3QN_ENABLED,
        }
        
        return self._make_spec(feature_genome, action_name, regime, risk_profile=risk_profile, rl_meta=rl_meta)


    def learn(
        self, 
        reward: float, 
        next_regime: RegimeState, 
        policy_spec: PolicySpec = None,
        metrics: Optional[Dict] = None,  # CPCV 지표 (D3QN 보상 재계산용)
    ):
        """
        Feedback loop for the RL agent.
        
        D3QN 모드에서는 metrics를 사용하여 더 정교한 보상을 계산합니다.
        
        Args:
            reward: 보상 (Q-Table 모드) 또는 기본 보상 (D3QN에서 재계산 가능)
            next_regime: 다음 시장 상태
            policy_spec: 실행된 정책 (RL 메타정보 포함)
            metrics: CPCV 평가 지표 (D3QN 보상 계산용)
        """
        if not policy_spec or not policy_spec.rl_meta:
            # D3QN 모드
            if config.D3QN_ENABLED:
                self.integrated_rl.update(reward, next_regime, next_df=self._current_df)
            else:
                self.strategy_rl.update(reward, next_regime)
                self.risk_rl.update(reward, next_regime)
            return

        state_key = policy_spec.rl_meta.get("state_key")
        strategy_idx = policy_spec.rl_meta.get("strategy_action_idx")
        risk_idx = policy_spec.rl_meta.get("risk_action_idx")

        # [V11.4] Integrated D3QN 모드
        if config.D3QN_ENABLED:
            self.integrated_rl.update(reward, next_regime, next_df=self._current_df)
            
            # [V11.4] Update Indicator Priors from feedback
            if metrics:
                is_rejected = getattr(metrics, 'is_rejected', metrics.get('is_rejected', False))
                if not is_rejected:
                    self.analyst.update_with_record(metrics, state_key, self.registry)
        else:
            # Q-Table 모드
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
            "TREND_FOLLOWING": ["TREND", "ADAPTIVE"],
            "MEAN_REVERSION": ["MOMENTUM", "VOLUME", "MEAN_REVERSION"], 
            "VOLATILITY_BREAK": ["VOLATILITY", "TREND"],
            "MOMENTUM_ALPHA": ["MOMENTUM", "PRICE_ACTION"],
            "DIP_BUYING": ["MOMENTUM", "TREND", "PATTERN"], 
            "DEFENSIVE": ["VOLATILITY", "TREND", "ADAPTIVE"]
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
            # [V11.2] Stage-based Feature Count Expansion
            # Stage가 올라갈수록 더 복잡한 전략(더 많은 지표)을 시도하도록 유도
            stage = self.curriculum.current_stage
            min_count = config.GENOME_FEATURE_COUNT_MIN + (stage - 1)
            max_count = min(len(target_candidates), config.GENOME_FEATURE_COUNT_MAX + (stage - 1) * 2)
            
            subset_size = random.randint(min_count, max_count)
            
            # [V11.4] Weighted Sampling by Category Priors
            priors = self.analyst.get_priors(regime.name)
            
            # Filter and normalize priors for target categories
            valid_target_cats = [cat for cat in target_categories if self.registry.list_by_category(cat)]
            if not valid_target_cats:
                selected_features = random.sample(target_candidates, subset_size)
            else:
                cat_weights = [priors.get(cat, 0.1) for cat in valid_target_cats]
                sum_w = sum(cat_weights)
                cat_weights = [w/sum_w for w in cat_weights]
                
                selected_features = []
                # Distribute slots among categories based on weights
                # Using random.choices for category selection, then random.choice for indicator within category
                for _ in range(subset_size):
                    chosen_cat = random.choices(valid_target_cats, weights=cat_weights, k=1)[0]
                    cat_items = self.registry.list_by_category(chosen_cat)
                    if cat_items:
                        feat = random.choice(cat_items)
                        if feat not in selected_features:
                            selected_features.append(feat)
                
                # Fill up if duplicates reduced the count
                if len(selected_features) < subset_size:
                    remaining = [f for f in target_candidates if f not in selected_features]
                    if remaining:
                        selected_features.extend(random.sample(remaining, min(len(remaining), subset_size - len(selected_features))))
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
                    min_v = max(min_v, int(max_v * config.PARAM_DEFENSIVE_WINDOW_RATIO)) 
                elif action_context == "SCALPING" and "window" in p_schema.name:
                    max_v = min(max_v, config.PARAM_SCALPING_MAX_WINDOW)
                
                # Contextual Mutations - Regime
                if regime and regime.label == "PANIC" and "window" in p_schema.name:
                     # In Panic, cap max window to avoid too much lag
                     max_v = min(max_v, config.PARAM_PANIC_MAX_WINDOW)

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
        max_leverage = config.MAX_LEVERAGE_MIN + (1.0 / (k_down + 0.5))  # Inverted relationship
        max_leverage = max(config.MAX_LEVERAGE_MIN, min(config.MAX_LEVERAGE_MAX, max_leverage))
        
        # Stop Loss (Actual %): Derived from k_down and typical daily volatility (~1.5%)
        # This is an approximation; actual stop depends on market volatility
        estimated_daily_vol = config.RISK_EST_DAILY_VOL
        stop_loss = k_down * estimated_daily_vol
        stop_loss = max(config.STOP_LOSS_MIN, min(config.STOP_LOSS_MAX, stop_loss))
        
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
        entry_threshold = round(random.uniform(
            config.ENTRY_THRESHOLD_MIN, 
            config.ENTRY_THRESHOLD_MAX
        ), 2)
        
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
