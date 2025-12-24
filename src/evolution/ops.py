import random
import copy
from typing import Dict, Any, List
from src.contracts import PolicySpec, FeatureMetadata
from src.features.registry import get_registry

def crossover(parent1: PolicySpec, parent2: PolicySpec) -> PolicySpec:
    """[V15] Genetic Crossover between two PolicySpecs."""
    # 1. Genome Crossover (Indices/Indicators)
    g1 = parent1.feature_genome
    g2 = parent2.feature_genome
    all_features = set(g1.keys()).union(set(g2.keys()))
    
    new_genome = {}
    for feat_id in all_features:
        if feat_id in g1 and feat_id in g2:
            # Randomly pick from one parent
            new_genome[feat_id] = copy.deepcopy(random.choice([g1[feat_id], g2[feat_id]]))
        elif feat_id in g1:
            if random.random() < 0.6: # Favor existing ones slightly
                new_genome[feat_id] = copy.deepcopy(g1[feat_id])
        else:
            if random.random() < 0.6:
                new_genome[feat_id] = copy.deepcopy(g2[feat_id])
                
    # 2. Risk Crossover
    r1 = parent1.risk_budget
    r2 = parent2.risk_budget
    new_risk = {}
    for k in set(r1.keys()).union(set(r2.keys())):
        if k in r1 and k in r2:
            new_risk[k] = random.choice([r1[k], r2[k]])
        else:
            new_risk[k] = r1.get(k) or r2.get(k)
            
    # 3. Create Child
    child = copy.deepcopy(parent1)
    child.spec_id = f"evo_{child.spec_id[:8]}_{random.getrandbits(16)}"
    child.feature_genome = new_genome
    child.risk_budget = new_risk
    
    return child

def mutate(policy: PolicySpec, mutation_rate: float = 0.2) -> PolicySpec:
    """[V15] Genetic Mutation of a PolicySpec."""
    registry = get_registry()
    new_policy = copy.deepcopy(policy)
    genome = new_policy.feature_genome
    
    # 1. Parameter Mutation
    for feat_id, params in genome.items():
        if random.random() < mutation_rate:
            meta = registry.get(feat_id)
            if meta:
                # Randomly pick a parameter to mutate
                param_spec = random.choice(meta.params)
                if param_spec.param_type in ("int", "float"):
                    # Slight nudge (Gaussian)
                    current_val = params.get(param_spec.name, param_spec.default)
                    span = (param_spec.max - param_spec.min) if (param_spec.max and param_spec.min) else 10.0
                    nudge = random.gauss(0, span * 0.1)
                    new_val = current_val + nudge
                    # Clamp
                    if param_spec.min is not None: new_val = max(new_val, param_spec.min)
                    if param_spec.max is not None: new_val = min(new_val, param_spec.max)
                    params[param_spec.name] = int(new_val) if param_spec.param_type == "int" else float(new_val)
                elif param_spec.param_type == "categorical":
                    params[param_spec.name] = random.choice(param_spec.choices)

    # 2. Structural Mutation (Add/Remove feature)
    if random.random() < mutation_rate:
        if len(genome) > 1 and random.random() < 0.5:
            # Remove
            target = random.choice(list(genome.keys()))
            del genome[target]
        else:
            # Add
            all_meta = registry.list_all()
            if all_meta:
                new_meta = random.choice(all_meta)
                if new_meta.feature_id not in genome:
                    # Generate default params
                    new_params = {}
                    for p in new_meta.params:
                        if p.param_type == "categorical":
                            new_params[p.name] = random.choice(p.choices)
                        else:
                            new_params[p.name] = p.default or (p.min + p.max) / 2
                    genome[new_meta.feature_id] = new_params
                    
    return new_policy
