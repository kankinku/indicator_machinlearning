from __future__ import annotations

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple
from src.contracts import PolicySpec
from src.shared.logger import get_logger
from src.config import config

logger = get_logger("orchestration.policy_evaluator")

class RuleEvaluator:
    """
    V13-PRO Safe Rule Evaluator.
    
    Features:
    - Token-based safe evaluation (No arbitrary code execution).
    - NaN Policy: Any comparison with NaN returns False.
    - Complexity measurement.
    """
    
    # Allowed symbols for security and DSL parsing
    ALLOWED_OPS = {"<", ">", "<=", ">=", "==", "!=", "and", "or", "not", "(", ")"}
    
    # [V14-O] Rule Evaluation Cache
    _rule_cache: Dict[str, Tuple[List[str], float]] = {}

    def evaluate_signals(
        self,
        df: pd.DataFrame,
        policy_spec: PolicySpec,
        close_prices: Optional[pd.Series] = None,
        tp_pct: Optional[float] = None,
        sl_pct: Optional[float] = None,
        max_hold_bars: Optional[int] = None,
    ) -> Tuple[pd.Series, pd.Series, float, pd.Series, pd.Series, pd.Series]:
        """
        Returns (entry_signals, exit_signals, complexity_score, act_signals, hold_short_signals, hold_long_signals).
        [V17] Prefers LogicTree (AST) over text rules.
        [V18] Catches LogicTreeMatchError and returns explicit failure signals.
        
        If match fails, returns (all_false, all_false, -1.0) where -1.0 complexity 
        indicates a matching failure. The caller should check for this.
        """
        from src.shared.logic_tree import LogicTree, evaluate_logic_tree, parse_text_to_logic, asdict
        from src.shared.logic_tree_diagnostics import LogicTreeMatchError
        
        # 1. Sync LogicTree (Phase 1 Migration)
        entry_tree_dict = None
        exit_tree_dict = None
        timing_tree_dict = None
        hold_short_tree_dict = None
        hold_long_tree_dict = None
        if policy_spec.logic_trees:
            entry_tree_dict = policy_spec.logic_trees.get("entry")
            exit_tree_dict = policy_spec.logic_trees.get("exit")
            timing_tree_dict = policy_spec.logic_trees.get("timing")
            hold_short_tree_dict = policy_spec.logic_trees.get("hold_short")
            hold_long_tree_dict = policy_spec.logic_trees.get("hold_long")

        if not entry_tree_dict:
            entry_rule = policy_spec.decision_rules.get("entry", "True")
            entry_tree_dict = asdict(parse_text_to_logic(entry_rule).root)
            if policy_spec.logic_trees is None:
                policy_spec.logic_trees = {}
            policy_spec.logic_trees["entry"] = entry_tree_dict

        if not exit_tree_dict:
            exit_rule = policy_spec.decision_rules.get("exit", "")
            if exit_rule and exit_rule.strip().lower() not in {"false", ""}:
                exit_tree_dict = asdict(parse_text_to_logic(exit_rule).root)
                if policy_spec.logic_trees is None:
                    policy_spec.logic_trees = {}
                policy_spec.logic_trees["exit"] = exit_tree_dict
            else:
                policy_spec._exit_tree_disabled = True

        if not timing_tree_dict:
            timing_rule = policy_spec.decision_rules.get("timing", "")
            if timing_rule and timing_rule.strip().lower() not in {"false", ""}:
                timing_tree_dict = asdict(parse_text_to_logic(timing_rule).root)
                if policy_spec.logic_trees is None:
                    policy_spec.logic_trees = {}
                policy_spec.logic_trees["timing"] = timing_tree_dict
            else:
                policy_spec._timing_tree_disabled = True

        if not hold_short_tree_dict:
            hold_short_rule = policy_spec.decision_rules.get("hold_short", "")
            if hold_short_rule and hold_short_rule.strip().lower() not in {"false", ""}:
                hold_short_tree_dict = asdict(parse_text_to_logic(hold_short_rule).root)
                if policy_spec.logic_trees is None:
                    policy_spec.logic_trees = {}
                policy_spec.logic_trees["hold_short"] = hold_short_tree_dict
            else:
                policy_spec._hold_short_tree_disabled = True

        if not hold_long_tree_dict:
            hold_long_rule = policy_spec.decision_rules.get("hold_long", "")
            if hold_long_rule and hold_long_rule.strip().lower() not in {"false", ""}:
                hold_long_tree_dict = asdict(parse_text_to_logic(hold_long_rule).root)
                if policy_spec.logic_trees is None:
                    policy_spec.logic_trees = {}
                policy_spec.logic_trees["hold_long"] = hold_long_tree_dict
            else:
                policy_spec._hold_long_tree_disabled = True
            
        try:
            # 2. Evaluate using LogicTree
            entry_tree = LogicTree.from_dict(entry_tree_dict) if entry_tree_dict else None
            exit_tree = LogicTree.from_dict(exit_tree_dict) if exit_tree_dict else None
            timing_tree = LogicTree.from_dict(timing_tree_dict) if timing_tree_dict else None
            hold_short_tree = LogicTree.from_dict(hold_short_tree_dict) if hold_short_tree_dict else None
            hold_long_tree = LogicTree.from_dict(hold_long_tree_dict) if hold_long_tree_dict else None
            
            # [V17] Use LogicTree for Signal Generation
            entry_sig = evaluate_logic_tree(entry_tree, df) if entry_tree else pd.Series(False, index=df.index)
            exit_sig = evaluate_logic_tree(exit_tree, df) if exit_tree else pd.Series(False, index=df.index)
            if timing_tree and getattr(config, "ACT_TIMING_ENABLED", True):
                act_sig = evaluate_logic_tree(timing_tree, df)
            else:
                act_sig = pd.Series(True, index=df.index)
            hold_short_sig = evaluate_logic_tree(hold_short_tree, df) if hold_short_tree else pd.Series(False, index=df.index)
            hold_long_sig = evaluate_logic_tree(hold_long_tree, df) if hold_long_tree else pd.Series(False, index=df.index)

            act_all_true = bool(act_sig.all())
            act_all_false = not bool(act_sig.any())
            if act_all_true:
                stride = int(getattr(config, "ACT_TIMING_FALLBACK_STRIDE", 1))
                stride = max(1, stride)
                mask = np.zeros(len(df), dtype=bool)
                mask[::stride] = True
                act_sig = pd.Series(mask, index=df.index)
            elif act_all_false:
                fallback = entry_sig | exit_sig
                act_sig = fallback if bool(fallback.any()) else pd.Series(False, index=df.index)

            if getattr(config, "ACT_TIMING_ENABLED", True):
                entry_sig = entry_sig & act_sig
                exit_sig = exit_sig & act_sig

            if getattr(config, "POLICY_STATE_GATE_ENABLED", True):
                entry_sig, exit_sig = self._apply_state_gate(
                    entry_sig,
                    exit_sig,
                    hold_short_sig=hold_short_sig,
                    hold_long_sig=hold_long_sig,
                    close_prices=close_prices,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                    max_hold_bars=max_hold_bars,
                )
            
            # 3. Complexity Calculation (Structural)
            complexity = self._calculate_tree_complexity(entry_tree)
            if exit_tree:
                complexity += self._calculate_tree_complexity(exit_tree)
            if timing_tree:
                complexity += self._calculate_tree_complexity(timing_tree)
            if hold_short_tree:
                complexity += self._calculate_tree_complexity(hold_short_tree)
            if hold_long_tree:
                complexity += self._calculate_tree_complexity(hold_long_tree)
            
            return entry_sig, exit_sig, complexity, act_sig, hold_short_sig, hold_long_sig
            
        except LogicTreeMatchError as e:
            # [V18] 매칭 실패: 명시적 실패 반환
            logger.error(f"[RuleEvaluator] LogicTree 매칭 실패: {e.message}")
            
            # 실패 마커: complexity = -1.0 (caller가 이를 확인해야 함)
            # 모든 신호를 False로 설정하고, 실패 원인을 policy_spec에 기록
            all_false = pd.Series(False, index=df.index)
            
            # 실패 원인을 policy에 기록 (REJECT 처리용)
            if not hasattr(policy_spec, '_logictree_error'):
                policy_spec._logictree_error = None
            policy_spec._logictree_error = {
                "type": e.match_type,
                "feature_key": e.feature_key,
                "message": e.message
            }
            
            # complexity = -1.0은 "FEATURE_MISSING" 실패 마커
            return all_false, all_false, -1.0, all_false, all_false, all_false

    def _apply_state_gate(
        self,
        entry_sig: pd.Series,
        exit_sig: pd.Series,
        hold_short_sig: Optional[pd.Series] = None,
        hold_long_sig: Optional[pd.Series] = None,
        close_prices: Optional[pd.Series] = None,
        tp_pct: Optional[float] = None,
        sl_pct: Optional[float] = None,
        max_hold_bars: Optional[int] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        if entry_sig.empty:
            return entry_sig, exit_sig

        entry_mask = np.zeros(len(entry_sig), dtype=bool)
        exit_mask = np.zeros(len(exit_sig), dtype=bool)
        state = 0  # 0: FLAT, 1: LONG
        hold_bars = 0
        current_min_hold = 0
        current_max_hold = max_hold_bars
        entry_price = None
        price_values = None
        use_forced_exit = bool(getattr(config, "POLICY_STATE_GATE_FORCED_EXIT", True))
        if close_prices is not None and use_forced_exit:
            try:
                price_values = close_prices.reindex(entry_sig.index).values
            except Exception:
                if len(close_prices) == len(entry_sig):
                    price_values = close_prices.values

        hold_min_values = None
        hold_max_values = None
        if hold_short_sig is not None and hold_long_sig is not None:
            stage_id = int(getattr(config, "CURRICULUM_CURRENT_STAGE", 1))
            buckets = getattr(config, "HOLD_DURATION_BUCKETS_BY_STAGE", {}).get(stage_id)
            if not buckets:
                buckets = getattr(config, "HOLD_DURATION_BUCKETS_BY_STAGE", {}).get(1, {})

            def _pair(key: str, fallback: tuple[int, int]) -> tuple[int, int]:
                raw = buckets.get(key, fallback)
                if isinstance(raw, dict):
                    min_v = int(raw.get("min", fallback[0]))
                    max_v = int(raw.get("max", fallback[1]))
                else:
                    min_v = int(raw[0]) if isinstance(raw, (list, tuple)) else fallback[0]
                    max_v = int(raw[1]) if isinstance(raw, (list, tuple)) else fallback[1]
                if max_v < min_v:
                    max_v = min_v
                return min_v, max_v

            short_min, short_max = _pair("short", (1, 5))
            medium_min, medium_max = _pair("medium", (5, 20))
            long_min, long_max = _pair("long", (20, 60))

            short_mask = hold_short_sig.astype(bool).values
            long_mask = hold_long_sig.astype(bool).values & ~short_mask
            hold_min_values = np.where(short_mask, short_min, np.where(long_mask, long_min, medium_min))
            hold_max_values = np.where(short_mask, short_max, np.where(long_mask, long_max, medium_max))

        entry_vals = entry_sig.values
        exit_vals = exit_sig.values
        for i in range(len(entry_vals)):
            if state == 0:
                if bool(entry_vals[i]):
                    entry_mask[i] = True
                    state = 1
                    hold_bars = 0
                    if hold_min_values is not None:
                        current_min_hold = int(hold_min_values[i])
                    else:
                        current_min_hold = 0
                    if hold_max_values is not None:
                        current_max_hold = int(hold_max_values[i])
                    else:
                        current_max_hold = max_hold_bars
                    if current_max_hold is not None and current_max_hold < current_min_hold:
                        current_max_hold = current_min_hold
                    if price_values is not None:
                        entry_price = float(price_values[i])
            else:
                if bool(exit_vals[i]) and hold_bars >= current_min_hold:
                    exit_mask[i] = True
                    state = 0
                    hold_bars = 0
                    current_min_hold = 0
                    current_max_hold = max_hold_bars
                    entry_price = None
                    continue

                hold_bars += 1
                forced_exit = False
                if use_forced_exit:
                    if entry_price is not None and price_values is not None:
                        price = float(price_values[i])
                        if tp_pct is not None and price >= entry_price * (1.0 + float(tp_pct)):
                            forced_exit = True
                        elif sl_pct is not None and price <= entry_price * (1.0 - float(sl_pct)):
                            forced_exit = True
                    if current_max_hold is not None and hold_bars >= int(current_max_hold):
                        forced_exit = True

                if forced_exit:
                    state = 0
                    hold_bars = 0
                    current_min_hold = 0
                    current_max_hold = max_hold_bars
                    entry_price = None

        return (
            pd.Series(entry_mask, index=entry_sig.index),
            pd.Series(exit_mask, index=exit_sig.index),
        )

    def _calculate_tree_complexity(self, tree: LogicTree) -> float:
        """
        Calculates complexity based on tree structure.
        structural_complexity = 1.0 * n_features + 0.5 * n_compares + 0.3 * n_logic + 0.2 * depth
        """
        from src.shared.logic_tree import ConditionNode, LogicalOpNode, NotNode
        if not tree or not tree.root: return 0.5
        
        n_features = set()
        n_compares = 0
        n_logic = 0
        max_depth = 0
        
        def walk(node, depth):
            nonlocal n_compares, n_logic, max_depth
            max_depth = max(max_depth, depth)
            if isinstance(node, ConditionNode):
                n_compares += 1
                n_features.add(node.feature_key)
            elif isinstance(node, LogicalOpNode):
                n_logic += len(node.children)
                for child in node.children: walk(child, depth + 1)
            elif isinstance(node, NotNode):
                n_logic += 1
                walk(node.child, depth + 1)
                
        walk(tree.root, 1)
        return (1.0 * len(n_features)) + (0.5 * n_compares) + (0.3 * n_logic) + (0.2 * max_depth)

    def _safe_eval(self, df: pd.DataFrame, expr: str) -> Tuple[pd.Series, float]:
        """
        Safely evaluates an expression and returns boolean series + structural complexity.
        [V14-O] Optimized: Caches tokenization and AST complexity.
        """
        if expr == "False" or not expr:
            return pd.Series(False, index=df.index), 0.5
        if expr == "True":
            return pd.Series(True, index=df.index), 0.5

        # Check Cache
        if expr in self._rule_cache:
            processed_tokens, complexity = self._rule_cache[expr]
        else:
            # 1. Normalize
            norm_expr = self._normalize_rule(expr)
                
            # 2. Tokenize with Quantile support [q0.xx]
            tokens = re.findall(r"[a-zA-Z0-9_.\-]+|\[q[0-9.]+\]|[<>=!]+|[\(\)]", norm_expr)
            
            # 3. Complexity (AST)
            complexity = self._calculate_ast_complexity(norm_expr, df.columns)
            
            # 4. Process Tokens & Fuzzy Matching
            processed_tokens = []
            for t in tokens:
                t_lower = t.lower()
                if t_lower in self.ALLOWED_OPS:
                    processed_tokens.append(t_lower)
                elif t.startswith("[q") and t.endswith("]"):
                    processed_tokens.append(t) # Template for quantile
                elif t.replace('.', '', 1).isdigit() or (t.startswith('-') and t[1:].replace('.', '', 1).isdigit()):
                    processed_tokens.append(t)
                elif t in df.columns:
                    processed_tokens.append(f"`{t}`")
                else:
                    # Fuzzy/Prefix Match
                    match = None
                    matches = [c for c in df.columns if c.lower() == t_lower]
                    if matches:
                        match = matches[0]
                    if not match:
                        if "__" in t:
                            prefix, suffix = t.split("__", 1)
                            feat_cols = [c for c in df.columns if c.startswith(prefix + "__")]
                            if feat_cols:
                                suffix_matches = [c for c in feat_cols if suffix.lower() in c.lower()]
                                match = suffix_matches[0] if suffix_matches else feat_cols[0]
                        else:
                            feat_cols = [c for c in df.columns if c.startswith(t + "__")]
                            if feat_cols:
                                match = feat_cols[0]
                    
                    if match:
                        processed_tokens.append(f"`{match}`")
                    else:
                        logger.warning(f"[Evaluator] 알 수 없는 토큰: {t}")
                        return pd.Series(False, index=df.index), 10.0
            
            # Store in Cache
            self._rule_cache[expr] = (processed_tokens, complexity)

        # 5. Resolve Quantiles Dynamically (Values depend on window)
        eval_tokens = list(processed_tokens)
        for i in range(len(eval_tokens)):
            if eval_tokens[i].startswith("[q"):
                try:
                    if i >= 2:
                        col_wrapped = eval_tokens[i-2]
                        if col_wrapped.startswith("`") and col_wrapped.endswith("`"):
                            col_name = col_wrapped[1:-1]
                            col_vals = df[col_name]
                            q_val = float(eval_tokens[i][2:-1])
                            quant_val = col_vals.quantile(q_val)
                            eval_tokens[i] = str(round(quant_val, 6))
                except Exception as e:
                    logger.error(f"[Evaluator] 분위수 해석 실패: {e}")
                    eval_tokens[i] = "0.0"

        safe_expr = " ".join(eval_tokens)
        try:
            result = df.eval(safe_expr, engine='python')
            if isinstance(result, pd.Series):
                return result.fillna(False).astype(bool), complexity
            return pd.Series(bool(result), index=df.index), complexity
        except Exception as e:
            logger.error(f"[Evaluator] 평가 실패: '{expr}' ({e})")
            return pd.Series(False, index=df.index), complexity + 5.0

    def _normalize_rule(self, expr: str) -> str:
        """
        Basic rule normalization: 
        - Remove redundant spaces
        - (Optional) Normalize directions x < a -> a > x etc (Skip for now to keep it simple but safe)
        """
        return " ".join(expr.split())

    def _calculate_ast_complexity(self, expr: str, columns: List[str]) -> float:
        """
        structural_complexity = 1.0 * n_features + 0.5 * n_compares + 0.3 * n_logic + 0.2 * depth
        """
        import ast
        try:
            # Prepare expression for AST (replacing some ops if needed, but python-style is ok)
            tree = ast.parse(expr, mode='eval')
            
            n_features = 0
            n_compares = 0
            n_logic = 0
            max_depth = 0
            
            unique_features = set()
            
            def walk(node, depth):
                nonlocal n_compares, n_logic, max_depth
                max_depth = max(max_depth, depth)
                
                if isinstance(node, ast.Compare):
                    n_compares += 1
                elif isinstance(node, ast.BoolOp):
                    n_logic += len(node.values) - 1
                elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                    n_logic += 1
                elif isinstance(node, ast.Name):
                    if node.id in columns:
                        unique_features.add(node.id)
                
                for child in ast.iter_child_nodes(node):
                    walk(child, depth + 1)
            
            walk(tree.body, 1)
            n_features = len(unique_features)
            
            # Weighted complexity
            complexity = (1.0 * n_features) + (0.5 * n_compares) + (0.3 * n_logic) + (0.2 * max_depth)
            return round(complexity, 2)
        except:
            # Fallback to token count if AST fails
            return len(expr.split()) * 0.2
