from __future__ import annotations

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple
from src.contracts import PolicySpec
from src.shared.logger import get_logger

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

    def evaluate_signals(self, df: pd.DataFrame, policy_spec: PolicySpec) -> Tuple[pd.Series, pd.Series, float]:
        """
        Returns (entry_signals, exit_signals, complexity_score).
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
        if policy_spec.logic_trees:
            entry_tree_dict = policy_spec.logic_trees.get("entry")
            exit_tree_dict = policy_spec.logic_trees.get("exit")

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
            
        try:
            # 2. Evaluate using LogicTree
            entry_tree = LogicTree.from_dict(entry_tree_dict) if entry_tree_dict else None
            exit_tree = LogicTree.from_dict(exit_tree_dict) if exit_tree_dict else None
            
            # [V17] Use LogicTree for Signal Generation
            entry_sig = evaluate_logic_tree(entry_tree, df) if entry_tree else pd.Series(False, index=df.index)
            exit_sig = evaluate_logic_tree(exit_tree, df) if exit_tree else pd.Series(False, index=df.index)
            
            # 3. Complexity Calculation (Structural)
            complexity = self._calculate_tree_complexity(entry_tree)
            if exit_tree:
                complexity += self._calculate_tree_complexity(exit_tree)
            
            return entry_sig, exit_sig, complexity
            
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
            return all_false, all_false, -1.0

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
