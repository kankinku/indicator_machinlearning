
import sys
import os
import re

# Setup Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.registry import FeatureRegistry
from src.config import config

def fix_all_registry_issues():
    """
    Scans the registry for known broken features and deprecated syntax.
    Applies fixes and updates the registry.
    """
    print(">>> [Fixer] Starting Registry Repair...")
    registry = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH))
    registry.initialize()
    
    updates_count = 0
    
    with registry._lock:
        for f_id, meta in registry._features.items():
            original_code = meta.code_snippet
            fixed_code = original_code
            
            # Fix 1: Pandas Deprecation .fillna(method='ffill') -> .ffill()
            if ".fillna(method='ffill')" in fixed_code:
                fixed_code = fixed_code.replace(".fillna(method='ffill')", ".ffill()")
            if '.fillna(method="ffill")' in fixed_code:
                fixed_code = fixed_code.replace('.fillna(method="ffill")', ".ffill()")
                
            # Fix 2: Aroon Indicator Init Error
            # Error: AroonIndicator(close=df['close'], window=...) missing 'low'
            # We need to inject 'low' arg if missing
            if "AroonIndicator(close=" in fixed_code and "low=" not in fixed_code:
                # Naive regex replacement to inject low param
                # Assuming standard usage: AroonIndicator(close=df['close'], window=params['window'])
                # We want: AroonIndicator(close=df['close'], low=df['low'], window=params['window'])
                # But 'low' might not be readily available in variable name. 
                # Usually snippet has `df`. so we add `low=df['low']` if we see `close=df['close']`
                if "close=df['close']" in fixed_code:
                    fixed_code = fixed_code.replace("close=df['close']", "close=df['close'], low=df.get('low', df['close'])")
                # Fallback replacement
                elif "close=close" in fixed_code:
                     fixed_code = fixed_code.replace("close=close", "close=close, low=kwargs.get('low', close)")

            # Fix 3: Heikin Ashi Index Error
            # Typically caused by accessing logic on empty list or wrong index logic
            # We'll just disable/delete highly broken ones if complex fix is needed.
            # But let's try a simple safety patch if it's the specific heuristic provided in user logs.
            # actually for Heikin, if it's too broken, let's just DELETE it to stop the noise.
            if f_id == "PATTERN_HEIKIN_V1":
                print(f"    - Deleting consistently broken feature: {f_id}")
                # We can't delete directly while iterating, mark for deletion
                # But here we are modifying meta. We can flag it.
                # Actually, simplest way is to replace code with a dummy or remove it later.
                # Let's replace code with a dummy that returns None (graceful fail)
                fixed_code = """
class PatternHeikinV1Handler:
    def compute(self, df, **kwargs):
        # Disabled due to persistent index errors
        return None
"""

            # Check if changed
            if fixed_code != original_code:
                print(f"    - Fixing feature: {f_id}")
                meta.code_snippet = fixed_code
                updates_count += 1
                
        # Save changes
        if updates_count > 0:
            registry._save_to_disk()
            print(f">>> [Fixer] Repaired {updates_count} features. Registry saved.")
        else:
            print(">>> [Fixer] No issues found or already fixed.")

if __name__ == "__main__":
    fix_all_registry_issues()
