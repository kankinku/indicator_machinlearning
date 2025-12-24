from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import json

@dataclass
class LogicNode:
    """Base class for all logic nodes."""
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ConditionNode(LogicNode):
    type: str = "condition"
    feature_key: str = ""
    op: str = ">" # >, <, >=, <=, ==, cross_up, cross_down
    value: Union[float, str] = 0.0 # float or feature_key
    lookback: int = 1

    def __str__(self):
        v = f"'{self.value}'" if isinstance(self.value, str) else self.value
        return f"{self.feature_key} {self.op} {v}"

@dataclass
class LogicalOpNode(LogicNode):
    type: str = "logical"
    op: str = "and" # and, or
    children: List[Union[ConditionNode, LogicalOpNode, NotNode]] = field(default_factory=list)

    def __str__(self):
        if not self.children: return "True" if self.op == "and" else "False"
        return f"({f' {self.op} '.join(str(c) for c in self.children)})"

@dataclass
class NotNode(LogicNode):
    type: str = "not"
    child: Union[ConditionNode, LogicalOpNode] = None

    def __str__(self):
        return f"not ({self.child})"

@dataclass
class LogicTree:
    root: Union[ConditionNode, LogicalOpNode, NotNode]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LogicTree:
        if not data: return None
        def _build_node(d: Dict[str, Any]):
            if not d: return None
            node_type = d.get("type")
            if node_type == "condition":
                return ConditionNode(**d)
            if node_type == "logical":
                children = [_build_node(c) for c in d.get("children", []) if c]
                return LogicalOpNode(op=d.get("op", "and"), children=children)
            if node_type == "not":
                return NotNode(child=_build_node(d.get("child")))
            return None
        
        root = _build_node(data.get("root") or data) # Handle both nested and root-only
        return cls(root=root)

    def get_referenced_features(self) -> set:
        """Extracts all unique feature keys referenced in the tree."""
        features = set()
        if not self.root: return features
        
        def walk(node):
            if isinstance(node, ConditionNode):
                features.add(node.feature_key)
                if isinstance(node.value, str) and not node.value.startswith("[q") and node.value != "TRUE" and node.value != "FALSE":
                    features.add(node.value)
            elif isinstance(node, LogicalOpNode):
                for child in node.children: walk(child)
            elif isinstance(node, NotNode):
                walk(node.child)
        
        walk(self.root)
        return {f for f in features if f not in ("TRUE", "FALSE")}

    def get_all_nodes(self) -> List[LogicNode]:
        """Returns flat list of all nodes for sampling."""
        nodes = []
        def _walk(n):
            if not n: return
            nodes.append(n)
            if isinstance(n, LogicalOpNode):
                for c in n.children: _walk(c)
            elif isinstance(n, NotNode):
                _walk(n.child)
        _walk(self.root)
        return nodes

    def get_condition_nodes(self) -> List[ConditionNode]:
        return [n for n in self.get_all_nodes() if isinstance(n, ConditionNode)]

def mutate_tree(tree: LogicTree, registry: Any, action_type: str = "MUTATE_THRESHOLD") -> LogicTree:
    """
    Applies a mutation to the LogicTree.
    [V17] Direct Tree Evolution.
    """
    import random
    new_tree = LogicTree.from_dict(asdict(tree.root)) # Deep copy
    
    # 1. ADD_CONDITION
    if action_type == "ADD_CONDITION":
        all_features = registry.list_all_ids() if hasattr(registry, 'list_all_ids') else []
        if all_features:
            new_feat = random.choice(all_features)
            new_cond = ConditionNode(feature_key=new_feat, op=">", value="[q0.5]")
            
            # If root is logical, add to children
            if isinstance(new_tree.root, LogicalOpNode):
                if len(new_tree.root.children) < 5:
                    new_tree.root.children.append(new_cond)
            else:
                # Wrap existing root with AND
                old_root = new_tree.root
                new_tree.root = LogicalOpNode(op="and", children=[old_root, new_cond])
                
    # 2. MUTATE_THRESHOLD
    elif action_type == "MUTATE_THRESHOLD":
        conds = new_tree.get_condition_nodes()
        if conds:
            target = random.choice(conds)
            if isinstance(target.value, str) and target.value.startswith("[q"):
                curr_q = float(target.value[2:-1])
                new_q = np.clip(curr_q + random.uniform(-0.1, 0.1), 0.1, 0.9)
                target.value = f"[q{new_q:.2f}]"
            elif isinstance(target.value, (int, float)):
                target.value = round(target.value * random.uniform(0.9, 1.1), 4)

    # 3. SWAP_FEATURE
    elif action_type == "SWAP_FEATURE":
        conds = new_tree.get_condition_nodes()
        all_features = registry.list_all_ids() if hasattr(registry, 'list_all_ids') else []
        if conds and all_features:
            target = random.choice(conds)
            target.feature_key = random.choice(all_features)

    # 4. CHANGE_OP
    elif action_type == "CHANGE_OP":
        conds = new_tree.get_condition_nodes()
        if conds:
            target = random.choice(conds)
            ops = [">", "<", ">=", "<=", "==", "cross_up", "cross_down"]
            target.op = random.choice(ops)

    # 5. REMOVE_CONDITION
    elif action_type == "REMOVE_CONDITION":
        if isinstance(new_tree.root, LogicalOpNode) and len(new_tree.root.children) > 1:
            idx = random.randrange(len(new_tree.root.children))
            new_tree.root.children.pop(idx)
            if len(new_tree.root.children) == 1:
                new_tree.root = new_tree.root.children[0]
                
    return new_tree

def normalize_tree(tree: LogicTree) -> LogicTree:
    """
    Simplifies and normalizes the LogicTree.
    Example: (A and (B and C)) -> (A and B and C)
    """
    if not tree or not tree.root: return tree
    
    def _simplify(node):
        if isinstance(node, LogicalOpNode):
            # 1. Simplify children first
            new_children = []
            for c in [_simplify(child) for child in node.children]:
                if isinstance(c, LogicalOpNode) and c.op == node.op:
                    # Flatten same-type nested ops
                    new_children.extend(c.children)
                else:
                    new_children.append(c)
            
            # 2. Remove duplicates
            seen = set()
            unique_children = []
            for c in new_children:
                s = str(c)
                if s not in seen:
                    seen.add(s)
                    unique_children.append(c)
            
            if not unique_children:
                return ConditionNode(feature_key="TRUE", op="==", value=1.0)
            if len(unique_children) == 1:
                return unique_children[0]
            
            return LogicalOpNode(op=node.op, children=unique_children)
            
        if isinstance(node, NotNode):
            child = _simplify(node.child)
            if isinstance(child, NotNode):
                # not (not A) -> A
                return child.child
            return NotNode(child=child)
            
        return node

    tree.root = _simplify(tree.root)
    return tree

def evaluate_logic_tree(tree: LogicTree, df: pd.DataFrame) -> pd.Series:
    """Evaluates a LogicTree against a DataFrame and returns a boolean Series."""
    if not tree or not tree.root:
        return pd.Series(True, index=df.index)
        
    return _evaluate_node(tree.root, df)

def _evaluate_node(node: LogicNode, df: pd.DataFrame) -> pd.Series:
    if isinstance(node, ConditionNode):
        return _evaluate_condition(node, df)
    if isinstance(node, LogicalOpNode):
        results = [_evaluate_node(child, df) for child in node.children]
        if not results: return pd.Series(True, index=df.index)
        
        final = results[0]
        for res in results[1:]:
            if node.op == "and":
                final = final & res
            else: # or
                final = final | res
        return final
    if isinstance(node, NotNode):
        return ~_evaluate_node(node.child, df)
    
    return pd.Series(True, index=df.index)

def _evaluate_condition(node: ConditionNode, df: pd.DataFrame) -> pd.Series:
    """
    [V18] 4-Stage Feature Matching with KPI Tracking.
    
    Stage A: 직접 매칭 성공
    Stage B: Fuzzy 매칭 (prefix 후보 1개)
    Stage C: 모호성 (prefix 후보 2개 이상)
    Stage D: 미매칭 (후보 0개)
    
    학습 모드(STRICT=True): C, D에서 즉시 reject
    운영 모드(STRICT=False): C, D에서 fallback 허용 (KPI 기록 필수)
    """
    from src.config import config
    from src.shared.logic_tree_diagnostics import (
        get_diagnostics, LogicTreeMatchError
    )
    
    diag = get_diagnostics()
    f_key = node.feature_key
    
    # 특수 키 처리 (TRUE/FALSE)
    if f_key in ("TRUE", "False"):
        return pd.Series(f_key == "TRUE", index=df.index)
    
    # =========================================================
    # Stage A: 직접 매칭 시도
    # =========================================================
    if f_key in df.columns:
        diag.record_direct_match(f_key)
        col = df[f_key]
        return _apply_comparison(col, node, df)
    
    # =========================================================
    # Stage B~D: Fuzzy Matching 시도
    # =========================================================
    if not getattr(config, 'LOGICTREE_FUZZY_MATCH', True):
        # Fuzzy matching 비활성화 시 직접 unmatched 처리
        diag.record_unmatched(f_key)
        return _handle_unmatched(f_key, df, config, diag)
    
    # Prefix 기반 후보 검색
    prefix = f"{f_key}__"
    candidates = [c for c in df.columns if c.startswith(prefix)]
    
    # =========================================================
    # Stage B: Fuzzy 매칭 성공 (후보 1개)
    # =========================================================
    if len(candidates) == 1:
        actual_col = candidates[0]
        diag.record_fuzzy_match(f_key, actual_col)
        
        from src.shared.logger import get_logger
        logger = get_logger("shared.logic_tree")
        logger.debug(f"[LogicTree] Fuzzy match: '{f_key}' -> '{actual_col}'")
        
        col = df[actual_col]
        return _apply_comparison(col, node, df)
    
    # =========================================================
    # Stage C: 모호성 (후보 2개 이상)
    # =========================================================
    if len(candidates) > 1:
        diag.record_ambiguous(f_key, candidates)
        return _handle_ambiguous(f_key, candidates, node, df, config, diag)
    
    # =========================================================
    # Stage D: 미매칭 (후보 0개)
    # =========================================================
    diag.record_unmatched(f_key)
    return _handle_unmatched(f_key, df, config, diag)


def _apply_comparison(col: pd.Series, node: ConditionNode, df: pd.DataFrame) -> pd.Series:
    """조건 비교를 수행합니다."""
    # Resolve value (feature vs constant)
    if isinstance(node.value, str):
        if node.value.startswith("[q"):
            # Handle [q0.x] quantile
            q_val = float(node.value[2:-1])
            val = col.quantile(q_val)
        elif node.value in df.columns:
            val = df[node.value]
        else:
            # Maybe it's a string representation of a float
            try: val = float(node.value)
            except: val = 0.0
    else:
        val = node.value
        
    if node.op == "==": return col == val
    if node.op == "!=": return col != val
    if node.op == ">": return col > val
    if node.op == "<": return col < val
    if node.op == ">=": return col >= val
    if node.op == "<=": return col <= val
    
    if node.op == "cross_up":
        # (col[t-1] <= val[t-1]) and (col[t] > val[t])
        prev_col = col.shift(1)
        prev_val = val.shift(1) if isinstance(val, pd.Series) else val
        return (prev_col <= prev_val) & (col > val)

    if node.op == "cross_down":
        prev_col = col.shift(1)
        prev_val = val.shift(1) if isinstance(val, pd.Series) else val
        return (prev_col >= prev_val) & (col < val)
        
    return pd.Series(False, index=df.index)


def _handle_ambiguous(
    f_key: str, 
    candidates: list, 
    node: ConditionNode, 
    df: pd.DataFrame,
    config,
    diag
) -> pd.Series:
    """
    모호성(후보 2개 이상) 처리.
    
    학습 모드: 즉시 reject
    운영 모드: 정책에 따라 대표 컬럼 선택
    """
    from src.shared.logger import get_logger
    from src.shared.logic_tree_diagnostics import LogicTreeMatchError
    
    logger = get_logger("shared.logic_tree")
    strict_mode = getattr(config, 'LOGICTREE_STRICT', True)
    policy = getattr(config, 'LOGICTREE_AMBIGUOUS_POLICY', 'error')
    
    # 학습 모드 + error 정책: 즉시 예외
    if strict_mode and policy == "error":
        msg = f"[LogicTree] AMBIGUOUS: '{f_key}' has {len(candidates)} candidates: {candidates[:5]}"
        logger.error(msg)
        raise LogicTreeMatchError(
            message=msg,
            feature_key=f_key,
            match_type="ambiguous"
        )
    
    # 운영 모드 또는 warn 정책: 대표 컬럼 선택
    logger.warning(f"[LogicTree] AMBIGUOUS: '{f_key}' -> {candidates[:5]}. Selecting representative.")
    
    # 대표 선택 규칙: __value 우선, 없으면 알파벳 순
    selected = None
    if policy == "warn_pick_value":
        value_cols = [c for c in candidates if c.endswith("__value")]
        if value_cols:
            selected = value_cols[0]
    
    if selected is None:
        # 알파벳 순 첫 번째
        selected = sorted(candidates)[0]
    
    logger.warning(f"[LogicTree] Selected '{selected}' for ambiguous key '{f_key}'")
    
    col = df[selected]
    return _apply_comparison(col, node, df)


def _handle_unmatched(
    f_key: str, 
    df: pd.DataFrame,
    config,
    diag
) -> pd.Series:
    """
    미매칭(후보 0개) 처리.
    
    학습 모드: 즉시 reject 또는 예외
    운영 모드: False 반환 (KPI 기록 필수)
    """
    from src.shared.logger import get_logger
    from src.shared.logic_tree_diagnostics import LogicTreeMatchError
    
    logger = get_logger("shared.logic_tree")
    strict_mode = getattr(config, 'LOGICTREE_STRICT', True)
    
    if strict_mode:
        msg = f"[LogicTree] UNMATCHED: Feature key '{f_key}' not found in DataFrame columns. Available: {list(df.columns)[:10]}"
        logger.error(msg)
        raise LogicTreeMatchError(
            message=msg,
            feature_key=f_key,
            match_type="unmatched"
        )
    
    # 운영 모드: False 반환 + 로그
    logger.warning(f"[LogicTree] UNMATCHED: '{f_key}' not found. Returning False (production fallback).")
    return pd.Series(False, index=df.index)

def parse_text_to_logic(expr: str) -> LogicTree:
    """
    Phase 1: Minimal text parser to LogicTree.
    Supports basic 'and' / 'or' of conditions like 'feat > [q0.5]'.
    """
    # Simple recursive descent or regex-based parser for Phase 1.
    # For now, let's handle a flat list of AND terms as a Proof of Concept.
    import re
    
    # Split by ' and ' (simplified)
    terms = expr.split(" and ")
    nodes = []
    for term in terms:
        # Match 'feat op value'
        match = re.match(r"`?([a-zA-Z0-9_.\-]+)`?\s*([<>=!]+|cross_up|cross_down)\s*(\[q[0-9.]+\]|[0-9.\-]+)", term)
        if match:
            feat, op, val = match.groups()
            # Clean up val if it starts with [q
            if val.startswith("[q"):
                # Represent quantile as a special value string for now
                pass
            else:
                try: val = float(val)
                except: pass
            
            nodes.append(ConditionNode(feature_key=feat, op=op, value=val))
        else:
            # Fallback for complex ones or 'True'/'False'
            if term.strip().lower() == "true":
                nodes.append(ConditionNode(feature_key="TRUE", op="==", value=1.0))
            elif term.strip().lower() == "false":
                nodes.append(ConditionNode(feature_key="FALSE", op="==", value=1.0))

    if not nodes:
        return LogicTree(root=ConditionNode(feature_key="TRUE", op="==", value=1.0))
    
    if len(nodes) == 1:
        return LogicTree(root=nodes[0])
    
    return LogicTree(root=LogicalOpNode(op="and", children=nodes))
