"""Fix broken indicators in features.json"""
import json
import os

# Get project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
features_path = os.path.join(project_root, 'data', 'features.json')

# Read file
with open(features_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# TREND_AROON_V1 fix - needs high and low, not just close
aroon_code = """
import pandas as pd
import ta

class AroonIndicator:
    def compute(self, df: pd.DataFrame, length: int = 25) -> pd.DataFrame:
        aroon = ta.trend.AroonIndicator(df["high"], df["low"], window=length)
        return pd.DataFrame({
            f"AROON_up_{length}": aroon.aroon_up(),
            f"AROON_down_{length}": aroon.aroon_down(),
            f"AROON_ind_{length}": aroon.aroon_indicator()
        }, index=df.index)
"""

# PATTERN_HEIKIN_V1 fix - proper implementation
heikin_code = """
import pandas as pd
import numpy as np

class HeikinAshiIndicator:
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # HA Close: (O + H + L + C) / 4
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # HA Open: requires recursive calculation
        ha_open = np.zeros(len(df))
        ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
        
        ha_close_arr = ha_close.values
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i-1] + ha_close_arr[i-1]) / 2
        
        ha_open_series = pd.Series(ha_open, index=df.index)
        
        # HA High: Max(High, HA_Open, HA_Close)
        ha_high = pd.concat([df['high'], ha_open_series, ha_close], axis=1).max(axis=1)
        
        # HA Low: Min(Low, HA_Open, HA_Close)
        ha_low = pd.concat([df['low'], ha_open_series, ha_close], axis=1).min(axis=1)
        
        return pd.DataFrame({
            "HA_Close": ha_close,
            "HA_Open": ha_open_series,
            "HA_High": ha_high,
            "HA_Low": ha_low
        }, index=df.index)
"""

# Apply fixes
fixed_count = 0
for item in data:
    if item['feature_id'] == 'TREND_AROON_V1':
        item['code_snippet'] = aroon_code
        print('TREND_AROON_V1 fixed')
        fixed_count += 1
    elif item['feature_id'] == 'PATTERN_HEIKIN_V1':
        item['code_snippet'] = heikin_code
        item['description'] = 'Heikin Ashi Smoothed OHLC.'
        print('PATTERN_HEIKIN_V1 fixed')
        fixed_count += 1

# Save
with open(features_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f'\nTotal fixed: {fixed_count} indicators')
print(f'features.json updated at: {features_path}')
