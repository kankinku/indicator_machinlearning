"""Test all indicators in the registry"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.factory import FeatureFactory
import pandas as pd
import numpy as np

# Generate test data
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'open': 100 + np.cumsum(np.random.randn(n) * 0.5),
    'high': 101 + np.cumsum(np.random.randn(n) * 0.5),
    'low': 99 + np.cumsum(np.random.randn(n) * 0.5),
    'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
    'volume': np.random.randint(1000, 5000, n).astype(float)
})
df['high'] = df[['open', 'close', 'high']].max(axis=1) + 0.5
df['low'] = df[['open', 'close', 'low']].min(axis=1) - 0.5

# Initialize factory
factory = FeatureFactory()

# Get all features from registry
all_features = factory.registry.list_all()
print(f'Total registered indicators: {len(all_features)}')

# Test each indicator
success = []
failed = []

for feat in all_features:
    try:
        # Use default params
        default_params = {}
        for p in feat.params:
            if p.default is not None:
                default_params[p.name] = p.default
        
        genome = {feat.feature_id: default_params}
        result = factory.generate_from_genome(df, genome)
        
        if not result.empty:
            success.append(feat.feature_id)
        else:
            failed.append((feat.feature_id, 'Empty result'))
    except Exception as e:
        failed.append((feat.feature_id, str(e)[:100]))

print()
print('=== TEST RESULTS ===')
print(f'SUCCESS: {len(success)} indicators')
print(f'FAILED:  {len(failed)} indicators')

if failed:
    print()
    print('Failed indicators:')
    for fid, err in failed:
        print(f'  - {fid}')
        print(f'    Error: {err}')
else:
    print()
    print('ALL INDICATORS WORKING CORRECTLY!')

# Category breakdown
print()
print('=== CATEGORY BREAKDOWN ===')
cats = {}
for f in all_features:
    cats[f.category] = cats.get(f.category, 0) + 1
for cat, count in sorted(cats.items()):
    print(f'  {cat}: {count}')
