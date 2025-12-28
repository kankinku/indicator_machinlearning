
import random
import uuid
import copy
import numpy as np
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
from src.l3_meta.epsilon_manager import get_epsilon_manager

# D3QN 조건부 임포트
if config.D3QN_ENABLED:
    from src.l3_meta.d3qn_agent import D3QNAgent, create_rl_agent
    from src.l3_meta.reward_shaper import get_reward_shaper

from src.l3_meta.curriculum_controller import get_curriculum_controller, CurriculumController
from src.l3_meta.analyst import get_indicator_analyst
from src.shared.hashing import generate_policy_id

logger = get_logger("meta.agent")

class MetaAgent:
    def __init__(self, registry: object, repo: LedgerRepo):
        # DI: 외부에서 주입받거나 싱글톤 사용
        self.registry = get_registry()
        self.repo = repo
        
        # [V11.4] Integrated Multi-head D3QN 모드
        if config.D3QN_ENABLED:
            logger.info("[MetaAgent] D3QN 통합 모드 초기화")
            from src.l3_meta.d3qn_agent import get_integrated_agent, DEFAULT_ACTIONS
            self.integrated_rl = get_integrated_agent(
                repo.base_dir,
                strategy_actions=DEFAULT_ACTIONS,
                risk_actions=list(self._get_risk_profile_ids())
            )
            self.reward_shaper = get_reward_shaper()
        else:
            logger.info("[MetaAgent] Q-Table 모드 초기화")
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
        
        # [vAlpha+] EAGL Engine
        from src.l3_meta.eagl import get_eagl_engine
        self.eagl = get_eagl_engine()
        
        # [V11.3] Warm-start Baselines
        self.baselines = self._load_baselines()
        
        # Epsilon Manager
        self.eps_manager = get_epsilon_manager()

        # Gate-guided mutation focus (Stage 4)
        self.mutation_gate_focus: Optional[str] = None
        self.mutation_operator_weights = self._get_mutation_operator_weights(None)
        
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
                logger.error(f"[MetaAgent] 베이스라인 로드 실패: {f} ({e})")
        
        logger.info(f"[MetaAgent] 베이스라인 로드: {len(baselines)}개 ({path})")
        return baselines

    def _get_risk_profile_ids(self) -> List[str]:
        """리스크 프로파일 ID 목록을 반환합니다."""
        profiles = get_default_risk_profiles()
        return [p.profile_id for p in profiles] or ["DEFAULT"]
    
    def set_market_data(self, df: pd.DataFrame) -> None:
        """현재 시장 데이터를 설정합니다 (D3QN 상태 인코딩용)."""
        self._current_df = df
        
    def adjust_policy(self, diagnostic_status: object) -> None:
        """
        [V15] Self-Healing Feedback Loop (SSOT EpsilonManager)
        """
        status = diagnostic_status
        if isinstance(diagnostic_status, dict):
            status = diagnostic_status.get("status", "")

        if "RIGID" in status:
            self.eps_manager.request_reheat("RIGID_DIAGNOSTIC", strength=0.7)
            
        elif "COLLAPSED" in status:
            self.eps_manager.request_reheat("COLLAPSED_DIAGNOSTIC", strength=0.5)
                
        elif "STAGNANT" in status:
            self.eps_manager.request_reheat("STAGNANT_DIAGNOSTIC", strength=0.4)

        elif "SOFT" in status:
            # Maybe slow down decay or just log
            logger.info("[MetaAgent] Self-Healing: SOFT 상태 (활용 비중 강화)")

    def apply_market_context_modulation(self, regime: RegimeState):
        """
        [Alpha-Power V1] Apply EAGL guidance based on current market context.
        Handles De-sync (Trust Discount) and High/Low Volatility.
        """
        # 1. De-sync (Correlation breakdown) detection
        is_desync, trust_mult = self.eagl.should_discount_trust(regime)
        if is_desync:
            logger.warning(f"[EAGL] 디싱크 감지 (Corr: {regime.corr_score:.2f}) -> 신뢰도 할인")
            # For now, we increase exploration as a way to "re-learn"
            self.eps_manager.request_reheat("DE_SYNC_DISCOUNT", strength=0.3)
            
        # 2. Vol Squeeze / Explosion
        if regime.vol_level < 0.7:
             logger.info(f"[EAGL] VOL_SQUEEZE (Vol: {regime.vol_level:.2f}) -> 탐색 강화")
             # Reheat slightly to explore new patterns before breakout
             self.eps_manager.request_reheat("VOL_SQUEEZE", strength=0.2)
        elif regime.vol_level > 2.0:
             logger.info(f"[EAGL] VOL_EXPLOSION (Vol: {regime.vol_level:.2f}) -> 보수적 동작")
             # Maybe slow down epsilon decrease? Or just log.
             pass

            

    def _get_mutation_operator_weights(self, gate_focus: Optional[str]) -> Dict[str, float]:
        base = getattr(config, "MUTATION_OPERATOR_BASE_WEIGHTS", {}) or {}
        weights = {op: float(base.get(op, 1.0)) for op in base}
        if not weights:
            weights = {
                "ADD_CONDITION": 1.0,
                "REMOVE_CONDITION": 1.0,
                "MUTATE_THRESHOLD": 1.0,
                "SWAP_FEATURE": 1.0,
                "CHANGE_OP": 1.0,
            }

        bias_map = getattr(config, "MUTATION_GATE_BIASES", {}) or {}
        if gate_focus and gate_focus in bias_map:
            for op, mult in bias_map[gate_focus].items():
                if op in weights:
                    weights[op] *= float(mult)

        return weights

    def _choose_mutation_operator(self) -> str:
        ops = list(self.mutation_operator_weights.keys())
        if not ops:
            return "MUTATE_THRESHOLD"
        weights = [max(0.0, float(self.mutation_operator_weights.get(op, 1.0))) for op in ops]
        if sum(weights) <= 0.0:
            return random.choice(ops)
        return random.choices(ops, weights=weights, k=1)[0]

    def update_mutation_gate_focus(self, gate_code: str) -> None:
        self.mutation_gate_focus = gate_code
        self.mutation_operator_weights = self._get_mutation_operator_weights(gate_code)

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
            logger.debug(f"[MetaAgent] 웜스타트: 베이스라인 사용 '{base.get('name')}'")
            spec = self._make_spec(
                genome=base["feature_genome"],
                template_tag=base.get("template_id", "BASELINE"),
                regime=regime,
                rl_meta={
                    "is_baseline": True,
                    "baseline_name": base.get("name"),
                    "state_key": regime.label,
                }
            )
            # Override spec_id with deterministic hash
            spec.spec_id = generate_policy_id(spec)
            return spec

        # 1. Standard RL Flow
        logger.debug(f"[MetaAgent] 레짐 감지: {regime.label} (Trend {regime.trend_score:.2f}, VIX {regime.vol_level:.2f})")

        from src.l3_meta.auto_tuner import get_auto_tuner
        tuner = get_auto_tuner()
        levers = tuner.get_levers()
        
        # [V16] AutoTuner intervention: search_profile_mix
        mix = levers.get("search_profile_mix")
        if mix and isinstance(mix, dict):
            # Sample action name from mix
            profiles = list(mix.keys())
            weights = list(mix.values())
            action_name = random.choices(profiles, weights=weights, k=1)[0]
            action_idx = self.integrated_rl.strategy_actions.index(action_name) if hasattr(self.integrated_rl, 'strategy_actions') and action_name in self.integrated_rl.strategy_actions else 0
            
            # Risk profile still from RL for now or default
            _, _, risk_profile_id, risk_action_idx = self.integrated_rl.get_action(regime, df=self._current_df)
            logger.info(f"[AutoTuner] 행동 오버라이드: {action_name} (mix)")
        else:
            # Get Action from RL (Multi-head Integrated)
            if config.D3QN_ENABLED:
                action_name, action_idx, risk_profile_id, risk_action_idx = self.integrated_rl.get_action(
                    regime, df=self._current_df
                )
            else:
                action_name, action_idx = self.strategy_rl.get_action(regime)
                risk_profile_id, risk_action_idx = self.risk_rl.get_action(regime)
        
        risk_profile = self._resolve_risk_profile(risk_profile_id)
        
        # Build LogicTree based on Action AND Regime (V17)
        logic_trees = self._construct_trees_from_action(action_name, regime, history)
        
        # [V17] Genome-Rule Sync: Extract feature_genome from LogicTree
        feature_genome = self._sync_genome_from_trees(logic_trees, action_name, regime)
        
        rl_meta = {
            "state_key": regime.label,
            "strategy_action": action_name,
            "strategy_action_idx": action_idx,
            "risk_profile": risk_profile_id,
            "risk_action_idx": risk_action_idx,
            "d3qn_mode": config.D3QN_ENABLED,
            "tuner_intervened": mix is not None,
            "logic_tree_mode": True
        }
        
        spec = self._make_spec(feature_genome, action_name, regime, risk_profile=risk_profile, rl_meta=rl_meta)
        spec.logic_trees = logic_trees
        # decision_rules for legacy/logging
        from src.shared.logic_tree import LogicTree
        spec.decision_rules = {
            "entry": str(LogicTree.from_dict(logic_trees["entry"]).root) if logic_trees.get("entry") else "True",
            "exit": str(LogicTree.from_dict(logic_trees["exit"]).root) if logic_trees.get("exit") else "False"
        }
        spec.spec_id = generate_policy_id(spec)
        
        # [vAlpha+] CRM check: If policy is dormant, try to revive or propose another
        if spec.status == "dormant" and not self.eagl.should_revive(spec, {}):
             return self.propose_policy(regime, history)
             
        return spec

    def propose_batch(self, regime: RegimeState, history: List[LedgerRecord], n: int, seed: Optional[int] = None) -> List[PolicySpec]:
        """
        [V15] Propose a batch of N diverse policies.
        Uses sample_with_diversity to strike a balance between quality (RL/Ontology) and novelty.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        from src.l3_meta.sampler import sample_with_diversity
        from src.l3_meta.auto_tuner import get_auto_tuner
        
        tuner = get_auto_tuner()
        levers = tuner.get_levers()
        
        # [V16] Use dynamic diversity pressure from tuner
        jaccard_th = levers.get("diversity_pressure", getattr(config, 'DIVERSITY_JACCARD_TH', 0.8))
        
        # [vAlpha+] CRM: Filter or Revive dormant policies from history if needed
        # In this implementation, we mostly use it to skip generating "already failed" profiles
        
        # Iterative proposal to fill the batch with diverse candidates
        policies = sample_with_diversity(
            self.propose_policy, 
            n, 
            regime, 
            history, 
            jaccard_th=jaccard_th
        )
        
        # [vAlpha+] EBR: If AOS is available, we could prune or weight here.
        # But AOS is usually populated AFTER evaluation. 
        # For now, we ensure dormancy is respected in propose_policy.
        return policies


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
                self.integrated_rl.update(reward, next_regime, next_df=self._current_df, metrics=metrics)
            else:
                self.strategy_rl.update(reward, next_regime)
                self.risk_rl.update(reward, next_regime)
            return

        state_key = policy_spec.rl_meta.get("state_key")
        strategy_idx = policy_spec.rl_meta.get("strategy_action_idx")
        risk_idx = policy_spec.rl_meta.get("risk_action_idx")

        # [V11.4] Integrated D3QN 모드
        if config.D3QN_ENABLED:
            self.integrated_rl.update(reward, next_regime, next_df=self._current_df, metrics=metrics)
            
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

    def _construct_genome_from_action(self, action_name: str, regime: RegimeState, history: List[LedgerRecord]) -> Dict[str, Any]:
        """
        [V14] Decodes Search Profile into Feature Genome Constraints.
        """
        profile_map = {
            "TREND_ALPHA":      (["TREND", "ADAPTIVE"], 2, 2, "tail"),
            "TREND_STABLE":     (["TREND"], 3, 3, "center"),
            "MOMENTUM_TAIL":    (["MOMENTUM", "PRICE_ACTION"], 2, 2, "tail"),
            "MOMENTUM_CENTER":  (["MOMENTUM"], 2, 2, "center"),
            "VOLATILITY_SNIPER":(["VOLATILITY"], 2, 2, "tail"),
            "PATTERN_COMPLEX":  (["TREND", "MOMENTUM", "VOLATILITY"], 3, 4, "spread"),
            "DEFENSIVE_CORE":   (["TREND", "ADAPTIVE"], 2, 2, "center"),
            "SCALPING_FAST":    (["MOMENTUM", "VOLATILITY"], 1, 2, "spread"),
            "EVOLVE":           (["TREND", "MOMENTUM"], 2, 3, "center") # Mapping for fallback
        }
        
        # [V15] Special handling for EVOLVE
        if action_name == "EVOLVE":
            from src.evolution.ops import crossover, mutate
            # Pick elite parents from history
            elite_records = [r for r in history if not r.is_rejected and (r.cpcv_metrics.get("sharpe", 0) > 1.0)]
            if len(elite_records) >= 2:
                p1 = random.choice(elite_records).policy_spec
                p2 = random.choice(elite_records).policy_spec
                feature_genome = crossover(p1, p2).feature_genome
                
                # [V16] Use mutation_strength from AutoTuner
                from src.l3_meta.auto_tuner import get_auto_tuner
                mut_strength = get_auto_tuner().get_levers().get("mutation_strength", 0.3)
                
                if random.random() < mut_strength:
                    temp_spec = copy.deepcopy(p1)
                    temp_spec.feature_genome = feature_genome
                    feature_genome = mutate(temp_spec).feature_genome
                return feature_genome
            else:
                logger.debug("[MetaAgent] EVOLVE 엘리트 부족 -> Trend Alpha로 대체")
                action_name = "TREND_ALPHA"

        cats, f_min, f_max, q_bias = profile_map.get(action_name, (["TREND"], 2, 2, "center"))
        
        # 1. Fetch Candidates from Registry
        target_candidates = []
        for cat in cats:
            target_candidates.extend(self.registry.list_by_category(cat))
            
        selected_features = []
        subset_size = random.randint(f_min, f_max)
        
        if target_candidates:
            # Weighted Sampling by Category Priors
            priors = self.analyst.get_priors(regime.label)
            
            valid_target_cats = [cat for cat in cats if self.registry.list_by_category(cat)]
            if not valid_target_cats:
                selected_features = random.sample(target_candidates, min(len(target_candidates), subset_size))
            else:
                # [V12.3] Auto-calibration of Ontology Influence (Continuous/Smooth)
                calib_score = self.analyst.get_calibration_score()
                t0 = getattr(config, 'ONTOLOGY_CALIB_THRESHOLD', 0.08)
                temp = getattr(config, 'ONTOLOGY_CALIB_TEMP', 0.02)
                
                # Sigmoid Trust Factor: 0.0 (Uniform) to 1.0 (Full Prior)
                trust_factor = 1.0 / (1.0 + np.exp(-(calib_score - t0) / temp))
                trust_factor = np.clip(trust_factor, 0.0, 1.0)
                
                # Use raw priors
                cat_weights_prior = [priors.get(cat, 0.1) for cat in valid_target_cats]
                sum_w = sum(cat_weights_prior)
                cat_weights_prior = [w/sum_w for w in cat_weights_prior]
                
                # Uniform weights
                n_cats = len(valid_target_cats)
                cat_weights_uniform = [1.0/n_cats] * n_cats
                
                # Blend: Trust * Prior + (1-Trust) * Uniform
                mix_min = getattr(config, 'ONTOLOGY_MIX_MIN', 0.1)
                mix_max = getattr(config, 'ONTOLOGY_MIX_MAX', 0.6)
                actual_mix_strength = mix_min + (mix_max - mix_min) * trust_factor
                
                cat_weights = []
                for i in range(n_cats):
                    w = actual_mix_strength * cat_weights_prior[i] + (1.0 - actual_mix_strength) * cat_weights_uniform[i]
                    cat_weights.append(w)
                
                selected_features = []
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
            all_candidates = self.registry.list_all()
            if all_candidates:
                selected_features = random.sample(all_candidates, 1)
        
        # 3. Parameter Evolution
        genome = {}
        for feature_meta in selected_features:
            params = self._evolve_params(feature_meta, action_name, regime)
            genome[feature_meta.feature_id] = params
            
        # Store bias for rule generation
        genome["__meta__"] = {"q_bias": q_bias, "profile": action_name}
        return genome

    def _construct_trees_from_action(self, action_name: str, regime: RegimeState, history: List[LedgerRecord]) -> Dict[str, Dict[str, Any]]:
        """
        [V17] Build LogicTree structure based on Action and Regime.
        """
        from src.shared.logic_tree import LogicTree, ConditionNode, LogicalOpNode, mutate_tree, asdict

        def _is_false_tree(tree_dict: Optional[Dict[str, Any]]) -> bool:
            if not tree_dict:
                return True
            try:
                tree = LogicTree.from_dict(tree_dict)
            except Exception:
                return False
            if not tree or not tree.root:
                return True
            if isinstance(tree.root, ConditionNode):
                if tree.root.feature_key == "FALSE" and tree.root.op == "==" and tree.root.value == 1.0:
                    return True
            return False

        def _build_exit_node(
            candidates: List[ConditionNode],
            q_bias: str,
        ) -> ConditionNode:
            if candidates:
                base = random.choice(candidates)
                exit_op = "<" if base.op in (">", ">=") else ">"
                return ConditionNode(
                    feature_key=base.feature_key,
                    column_ref=base.column_ref,
                    op=exit_op,
                    value=base.value if base.value else "[q0.5]",
                )

            fallback_features = self.registry.list_all() if hasattr(self.registry, "list_all") else []
            if not fallback_features:
                return ConditionNode(feature_key="FALSE", op="==", value=1.0)

            from src.contracts import ColumnRef
            feat = random.choice(fallback_features)
            meta = self.registry.get(feat.feature_id)
            outputs = meta.outputs if meta else {"value": "value"}
            key = "value" if "value" in outputs else list(outputs.keys())[0]
            exit_quantiles = [0.3, 0.5, 0.7]
            if q_bias == "tail":
                exit_weights = [0.45, 0.1, 0.45]
            elif q_bias == "center":
                exit_weights = [0.2, 0.6, 0.2]
            else:
                exit_weights = [0.34, 0.32, 0.34]
            q_val = random.choices(exit_quantiles, weights=exit_weights, k=1)[0]
            exit_op = ">" if random.random() > 0.5 else "<"
            return ConditionNode(
                feature_key=feat.feature_id,
                column_ref=ColumnRef(feature_id=feat.feature_id, output_key=key),
                op=exit_op,
                value=f"[q{q_val}]",
            )
        
        # [EVOLVE] Logic: Mutate existing elite tree
        if action_name == "EVOLVE":
            elite_records = [r for r in history if not r.is_rejected and (r.cpcv_metrics.get("sharpe", 0) > 1.0)]
            if elite_records:
                parent = random.choice(elite_records).policy_spec
                if parent.logic_trees:
                    entry_tree = LogicTree.from_dict(parent.logic_trees["entry"])
                    action_type = self._choose_mutation_operator()
                    mutated_entry = mutate_tree(entry_tree, self.registry, action_type=action_type)
                    exit_tree_dict = parent.logic_trees.get("exit")
                    if _is_false_tree(exit_tree_dict):
                        exit_node = _build_exit_node(mutated_entry.get_condition_nodes(), q_bias="center")
                        exit_tree_dict = asdict(exit_node)
                    return {
                        "entry": asdict(mutated_entry.root),
                        "exit": exit_tree_dict,
                    }

        # [Genome v2] Profile to Market Question Mapping
        # Phase 3: Map high-level RL actions to specific structural questions.
        PROFILE_TO_QUESTIONS = {
            "TREND_ALPHA":      ["TREND_CONFIRMATION", "VOLATILITY_EXPANSION"],
            "TREND_STABLE":     ["TREND_CONFIRMATION"],
            "MOMENTUM_TAIL":    ["OVERHEATED", "EXTREME_REVERSAL"],
            "MOMENTUM_CENTER":  ["OVERHEATED"],
            "VOLATILITY_SNIPER": ["STRUCTURAL_ENERGY", "VOLATILITY_EXPANSION"],
            "PATTERN_COMPLEX":  ["STRUCTURAL_ENERGY", "TREND_CONFIRMATION", "OVERHEATED"],
            "DEFENSIVE_CORE":   ["STRUCTURAL_ENERGY"],
            "SCALPING_FAST":    ["OVERHEATED", "VOLATILITY_EXPANSION"]
        }
        
        profile_info = {
            "TREND_ALPHA":      (2, "tail"),
            "TREND_STABLE":     (3, "center"),
            "MOMENTUM_TAIL":    (2, "tail"),
            "MOMENTUM_CENTER":  (2, "center"),
            "VOLATILITY_SNIPER": (2, "tail"),
            "PATTERN_COMPLEX":  (3, "spread"),
            "DEFENSIVE_CORE":   (2, "center"),
            "SCALPING_FAST":    (1, "spread")
        }
        
        subset_size, q_bias = profile_info.get(action_name, (2, "center"))
        stage_id = int(getattr(config, "CURRICULUM_CURRENT_STAGE", 1))
        stage_cfg = getattr(config, "CURRICULUM_STAGES", {}).get(stage_id)
        if stage_cfg:
            min_terms, max_terms = getattr(stage_cfg, "and_terms_range", (1, subset_size))
            subset_size = max(min_terms, min(subset_size, max_terms))
        target_questions = PROFILE_TO_QUESTIONS.get(action_name, ["TREND_CONFIRMATION"])
        
        # [Genome v2] Combinator Mode: Pick features that answer the questions
        from src.features.ontology import get_feature_ontology
        ontology = get_feature_ontology()
        
        # 1. Collect candidate features from relevant Market Questions
        potential_fids = set()
        for q in target_questions:
            potential_fids.update(ontology.MARKET_QUESTIONS.get(q, []))
        
        # 2. Fetch Metadata for these IDs
        candidates = [self.registry.get(fid) for fid in potential_fids if self.registry.get(fid)]
        
        if not candidates:
            # Fallback to general category if questions yield nothing
            logger.warning(f"[Genome v2] 질문 {target_questions}에 해당 특징 없음 -> 레지스트리로 대체")
            candidates = self.registry.list_all()
            
        selected_features = []
        if candidates:
            # Pick a seed from the primary question
            seed = random.choice(candidates)
            selected_features.append(seed)
            
            # 3. Add complementary features while respecting conflicts
            relation = ontology.get_relation(seed.feature_id)
            for _ in range(subset_size - 1):
                # Score remaining candidates based on synergy and lack of conflict
                scored_candidates = []
                for cand in candidates:
                    if cand in selected_features: continue
                    
                    # Compatibility score with currently selected set
                    comp_score = ontology.check_compatibility([f.feature_id for f in selected_features] + [cand.feature_id])
                    
                    if comp_score > -0.5: # Allow slight overlap but reject hard conflicts
                        scored_candidates.append((cand, comp_score))
                
                if scored_candidates:
                    # Sort by compatibility + small randomness
                    scored_candidates.sort(key=lambda x: x[1] + random.uniform(-0.1, 0.1), reverse=True)
                    selected_features.append(scored_candidates[0][0])
                else:
                    break
        
        # [V18] Build Entry Tree (Flat AND for now) with ColumnRef
        from src.contracts import ColumnRef
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if q_bias == "center": w = [0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05]
        elif q_bias == "tail": w = [0.3, 0.15, 0.05, 0.0, 0.0, 0.0, 0.05, 0.15, 0.3]
        else: w = [0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1]
        
        entry_nodes = []
        for feat in selected_features:
            q = random.choices(quantiles, weights=w, k=1)[0]
            op = ">" if random.random() > 0.5 else "<"
            
            # [V18] SSOT Resolution: Ensure output mapping exists
            meta = self.registry.get(feat.feature_id)
            if not meta: continue
            
            outputs = meta.outputs or {"value": "value"}
            # Prefer 'value' key if exists, else first available
            key = "value" if "value" in outputs else list(outputs.keys())[0]
            
            entry_nodes.append(ConditionNode(
                feature_key=feat.feature_id, 
                column_ref=ColumnRef(feature_id=feat.feature_id, output_key=key),
                op=op, 
                value=f"[q{q}]"
            ))
            
        if len(entry_nodes) > 1:
            entry_root = LogicalOpNode(op="and", children=entry_nodes)
        elif entry_nodes:
            entry_root = entry_nodes[0]
        else:
            entry_root = ConditionNode(feature_key="TRUE", op="==", value=1.0)
            
        # Build Simple Exit Tree
        exit_node = _build_exit_node(entry_nodes, q_bias)
        
        return {"entry": asdict(entry_root), "exit": asdict(exit_node)}

    def _sync_genome_from_trees(self, logic_trees: Dict[str, Dict[str, Any]], action_name: str, regime: RegimeState) -> Dict[str, Any]:
        """
        [V17] Genome-Rule Sync: Derive feature_genome from tree.
        """
        from src.shared.logic_tree import LogicTree
        feat_ids = set()
        for name, root_dict in logic_trees.items():
            tree = LogicTree.from_dict(root_dict)
            feat_ids.update(tree.get_referenced_features())
            
        genome = {}
        for fid in feat_ids:
            # Fetch metadata to evolve params
            meta = self.registry.get(fid)
            if meta:
                params = self._evolve_params(meta, action_name, regime)
                genome[fid] = params
            else:
                genome[fid] = {}
        
        # Store metadata for rules (q_bias might be useful if we mutate)
        genome["__meta__"] = {"profile": action_name}
        return genome
            

    def _construct_rules_from_genome(self, genome: Dict[str, Any]) -> Dict[str, str]:
        """
        [V14] Managed Rule Generation using Profile-driven Bias.
        """
        if not genome:
            return {"entry": "False", "exit": "True"}
            
        # Extract meta provided by _construct_genome_from_action
        meta = genome.get("__meta__", {})
        q_bias = meta.get("q_bias", "center") or "center"
        
        cur_stage = getattr(config, 'CURRICULUM_CURRENT_STAGE', 1)
        stages = getattr(config, 'CURRICULUM_STAGES', {})
        stage_cfg = stages.get(cur_stage, stages.get(1, {}))
        
        # [V14] Logic Complexity (AND terms) based on Stage & Profile
        and_range = getattr(stage_cfg, "and_terms_range", (1, 3))
        n_terms = random.randint(and_range[0], and_range[1])
        
        # Filter out meta keys
        feature_ids = [fid for fid in genome.keys() if not fid.startswith("__")]
        selected_fids = random.sample(feature_ids, min(len(feature_ids), n_terms)) if feature_ids else []

        # Quantile Sampling Logic
        quantiles = [round(x * 0.1, 1) for x in range(1, 10)]
        if q_bias == "center":
            weights = [0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05]
        elif q_bias == "tail":
            weights = [0.3, 0.15, 0.05, 0.0, 0.0, 0.0, 0.05, 0.15, 0.3]
        else: # "spread"
            weights = [0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1]
            
        entry_parts = []
        for fid in selected_fids:
            op = ">" if random.random() > 0.5 else "<"
            q = random.choices(quantiles, weights=weights, k=1)[0]
            
            # [V18] SSOT Resolution: No more fid.split('_')
            meta = self.registry.get(fid)
            if meta:
                outputs = meta.outputs or {"value": "value"}
                key = "value" if "value" in outputs else list(outputs.keys())[0]
                suffix = outputs[key]
                col_name = f"{fid}__{suffix}"
            else:
                col_name = fid # Fallback
                
            entry_parts.append(f"`{col_name}` {op} [q{q}]")
            
        entry_rule = " and ".join(entry_parts) if entry_parts else "True"
        
        # Simple Exit Rule
        exit_rule = "False"
        if selected_fids:
            exit_fid = random.choice(selected_fids)
            col_exit = f"{exit_fid}__" + exit_fid.split('_')[1].lower() if '_' in exit_fid else exit_fid
            exit_rule = f"`{col_exit}` > [q0.5]" if q_bias == "tail" else f"`{col_exit}` < [q0.3]"

        return {"entry": entry_rule, "exit": exit_rule}
            




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
            f"[PolicySpec] 리스크 파라미터: "
            f"k_up={k_up:.2f}, k_down={k_down:.2f}, horizon={horizon}, "
            f"RR={risk_reward_ratio:.2f}, SL={stop_loss:.3f}"
        )
        
        # [V6] Autonomy - Entry Threshold Control
        # Instead of fixed global threshold, let the agent decide per strategy.
        from src.l3_meta.auto_tuner import get_auto_tuner
        tuner_adj = get_auto_tuner().get_levers().get("s1_threshold_adjust", 0.0)
        
        entry_threshold = round(random.uniform(
            config.ENTRY_THRESHOLD_MIN, 
            config.ENTRY_THRESHOLD_MAX
        ) + tuner_adj, 2)
        entry_threshold = max(0.1, min(0.9, entry_threshold)) # Sanity clamp
        
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
