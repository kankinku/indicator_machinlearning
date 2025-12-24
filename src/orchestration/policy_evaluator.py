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
    
    def evaluate_signals(self, df: pd.DataFrame, policy_spec: PolicySpec) -> Tuple[pd.Series, pd.Series, float]:
        """
        Returns (entry_signals, exit_signals, complexity_score).
        """
        if not policy_spec.decision_rules:
            return pd.Series(False, index=df.index), pd.Series(False, index=df.index), 1.0
            
        entry_expr = policy_spec.decision_rules.get("entry", "False")
        exit_expr = policy_spec.decision_rules.get("exit", "False")
        
        entry_sig, entry_comp = self._safe_eval(df, entry_expr)
        exit_sig, exit_comp = self._safe_eval(df, exit_expr)
        
        total_complexity = entry_comp + exit_comp
        return entry_sig, exit_sig, total_complexity

    def _safe_eval(self, df: pd.DataFrame, expr: str) -> Tuple[pd.Series, float]:
        """
        Safely evaluates an expression and returns boolean series + structural complexity.
        """
        if expr == "False" or not expr:
            return pd.Series(False, index=df.index), 0.5
        if expr == "True":
            return pd.Series(True, index=df.index), 0.5

        # Normalize rule (e.g. standardizing whitespace, basic form)
        expr = self._normalize_rule(expr)
            
        # [V13.5] Tokenize with Quantile support [q0.xx]
        tokens = re.findall(r"[a-zA-Z0-9_.\-]+|\[q[0-9.]+\]|[<>=!]+|[\(\)]", expr)
        complexity = self._calculate_ast_complexity(expr, df.columns)
        
        processed_tokens = []
        for t in tokens:
            t_lower = t.lower()
            if t_lower in self.ALLOWED_OPS:
                processed_tokens.append(t_lower)
            elif t.startswith("[q") and t.endswith("]"):
                # Quantile detected: e.g. [q0.3]
                try:
                    q_val = float(t[2:-1])
                    # We can't put the value yet if we don't know the column it applies to
                    # But pd.eval doesn't support easy quantile mapping per comparison
                    # So we'll resolve it LATER or keep it as a special token
                    processed_tokens.append(t)
                except:
                    processed_tokens.append("0.0")
            elif t.replace('.', '', 1).isdigit() or (t.startswith('-') and t[1:].replace('.', '', 1).isdigit()):
                processed_tokens.append(t)
            elif t in df.columns:
                processed_tokens.append(f"`{t}`")
            else:
                # [V16] Fuzzy/Prefix Matching for Dynamic Column Names
                # Handles cases like ADAPTIVE_RAVI_V1__ravi guessed by Agent
                # OR just ADAPTIVE_RAVI_V1 used as a token
                match = None
                
                # A. Exact Match (Case-Insensitive)
                matches = [c for c in df.columns if c.lower() == t_lower]
                if matches:
                    match = matches[0]
                
                # B. Prefix + Signal Match (e.g. FEAT__signal or FEAT__FEAT_14)
                if not match:
                    # If t contains '__', try to match by prefix and a partial suffix
                    if "__" in t:
                        prefix, suffix = t.split("__", 1)
                        # Find all columns for this feature
                        feat_cols = [c for c in df.columns if c.startswith(prefix + "__")]
                        if feat_cols:
                            # Try to find one that contains the suffix
                            suffix_matches = [c for c in feat_cols if suffix.lower() in c.lower()]
                            if suffix_matches:
                                match = suffix_matches[0]
                            else:
                                # Fallback to first column of that feature
                                match = feat_cols[0]
                    else:
                        # If just FEATURE_ID is used, pick its first column
                        feat_cols = [c for c in df.columns if c.startswith(t + "__")]
                        if feat_cols:
                            match = feat_cols[0]

                if match:
                    processed_tokens.append(f"`{match}`")
                else:
                    logger.warning(f"[Evaluator] Unknown token: {t}")
                    return pd.Series(False, index=df.index), 10.0

        # Resolving Quantiles: x < [q0.3] -> x < column_quantile
        # This is tricky in one-pass eval. 
        # Better: Pre-resolve [q...] tokens if they are preceded by a column and a comparison op
        for i in range(len(processed_tokens)):
            if processed_tokens[i].startswith("[q"):
                try:
                    if i >= 2:
                        col_wrapped = processed_tokens[i-2]
                        if col_wrapped.startswith("`") and col_wrapped.endswith("`"):
                            col_name = col_wrapped[1:-1]
                            col_vals = df[col_name]
                            
                            # [V14] Collision Detection
                            if col_vals.std() < 1e-9:
                                logger.warning(f"[Evaluator] Collapsed distribution for feature '{col_name}' - possible constant value.")
                            
                            q_val = float(processed_tokens[i][2:-1])
                            quant_val = col_vals.quantile(q_val)
                            processed_tokens[i] = str(round(quant_val, 6))
                except Exception as e:
                    logger.error(f"[Evaluator] Quantile resolution failed: {e}")
                    processed_tokens[i] = "0.0"

        safe_expr = " ".join(processed_tokens)
        try:
            result = df.eval(safe_expr, engine='python')
            if isinstance(result, pd.Series):
                return result.fillna(False).astype(bool), complexity
            return pd.Series(bool(result), index=df.index), complexity
        except Exception as e:
            logger.error(f"[Evaluator] Eval failed for '{expr}': {e}")
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
