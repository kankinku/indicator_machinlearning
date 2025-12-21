"""Fix deprecated fillna method in features.json"""
import json
import os

# Get project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
features_path = os.path.join(project_root, 'data', 'features.json')

# Read file
with open(features_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# PA_PIVOT_V1 fix - deprecated fillna(method='bfill') -> bfill()
pivot_code = """
import pandas as pd

class PivotPointsIndicator:
    def compute(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        # Rolling Standard Pivot Points
        # Using a window to simulate "Previous Period" (e.g., last 20 days as 'last month')
        
        high = df['high'].rolling(window=window).max()
        low = df['low'].rolling(window=window).min()
        close = df['close'] # Current close? No, Pivot uses PREVIOUS period close.
        # So shift everything by 1
        
        pp_high = high.shift(1)
        pp_low = low.shift(1)
        pp_close = df['close'].shift(window).bfill() # Approximate previous 'session' close via lag
        
        # PP = (H + L + C) / 3
        pp = (pp_high + pp_low + pp_close) / 3
        r1 = (2 * pp) - pp_low
        s1 = (2 * pp) - pp_high
        
        # Return distance to Pivot
        dist_pp = (df['close'] - pp) / pp * 100
        
        return pd.DataFrame({
            f"Pivot_Level_{window}": pp,
            f"Dist_to_PP_{window}": dist_pp,
            f"R1_{window}": r1,
            f"S1_{window}": s1
        }, index=df.index)
"""

# Apply fix
fixed_count = 0
for item in data:
    if item['feature_id'] == 'PA_PIVOT_V1':
        item['code_snippet'] = pivot_code
        print('PA_PIVOT_V1 fixed: fillna(method="bfill") -> bfill()')
        fixed_count += 1

# Save
with open(features_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f'\nTotal fixed: {fixed_count} indicator(s)')
print(f'features.json updated!')
