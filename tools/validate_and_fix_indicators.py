
import sys
import os
import pandas as pd
import numpy as np
import re
import traceback

# Setup Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.registry import FeatureRegistry
from src.features.factory import FeatureFactory
from src.config import config
from src.contracts import FeatureMetadata

def create_dummy_ohlcv(n=500):
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='D')
    df = pd.DataFrame(index=dates)
    df['open'] = np.random.uniform(100, 200, n)
    df['high'] = df['open'] * 1.05
    df['low'] = df['open'] * 0.95
    df['close'] = np.random.uniform(100, 200, n)
    df['volume'] = np.random.randint(1000, 100000, n)
    return df

def fix_handler_name(snippet: str, current_handler: str) -> str:
    # Find class definition
    match = re.search(r"class\s+(\w+)", snippet)
    if match:
        found_class = match.group(1)
        if found_class != current_handler:
            print(f"        [Fix] Updating handler name: {current_handler} -> {found_class}")
            return found_class
    return current_handler

def inject_argument(snippet: str, arg_name: str) -> str:
    # Simple regex injection for library calls
    # E.g. Indicator(close=...) -> Indicator(close=..., low=...)
    if f"{arg_name}=" in snippet:
        return snippet # Already there?
    
    print(f"        [Fix] Injecting missing argument: {arg_name}")
    
    # Heuristic: Find a closing parenthesis of a function call that has 'close=' or 'high='
    # and insert the missing arg before it.
    
    # Case: AroonIndicator(close=...)
    pattern = r"(\w+\([^)]*close=[^)]*)(\))"
    
    def replacer(match):
        content = match.group(1)
        if arg_name not in content:
            return f"{content}, {arg_name}=df['{arg_name}']" + match.group(2)
        return match.group(0)
    
    new_snippet = re.sub(pattern, replacer, snippet)
    
    # Fallback: if 'df' is not available in snippet (unlikely), try kwargs
    if new_snippet == snippet:
        # Try generic injection
        new_snippet = snippet.replace("close=df['close']", f"close=df['close'], {arg_name}=df['{arg_name}']")
        
    return new_snippet

def validate_and_fix():
    print(">>> [Doctor] Starting Feature Health Check...")
    
    # 1. Load Data
    try:
        from src.data.loader import DataLoader
        loader = DataLoader(config.TARGET_TICKER)
        # Fetch small amount
        df = loader.fetch_ohlcv()
        if len(df) > 500: df = df.tail(500)
    except Exception as e:
        print(f"    [Warn] Could not fetch live data: {e}. Using dummy data.")
        df = create_dummy_ohlcv()
    
    # Ensure lowercase columns
    df.columns = [c.lower() for c in df.columns]
    
    # 2. Load Registry
    registry = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH))
    registry.initialize()
    factory = FeatureFactory()
    
    fixed_count = 0
    deleted_count = 0
    
    features_to_remove = []
    
    # We iterate over a copy of items to allow modification
    for f_id, meta in list(registry._features.items()):
        print(f"    Checking {f_id}...", end=" ")
        
        try:
            # Test Run
            factory.generate_single_feature(df, meta, {})
            print("OK")
            
        except Exception as e:
            print("FAIL")
            err_msg = str(e)
            print(f"        [Error] {err_msg}")
            
            # --- Auto-Fix Logic ---
            new_snippet = meta.code_snippet
            new_handler = meta.handler_func
            modified = False
            
            # Fix 1: Handler Name Mismatch
            if "Handler" in err_msg and "not found" in err_msg:
                new_handler = fix_handler_name(new_snippet, meta.handler_func)
                if new_handler != meta.handler_func:
                    meta.handler_func = new_handler
                    modified = True
            
            # Fix 2: Missing 'low', 'high', 'open' arguments for ta lib
            if "missing 1 required positional argument" in err_msg:
                if "'low'" in err_msg:
                    new_snippet = inject_argument(new_snippet, "low")
                elif "'high'" in err_msg:
                    new_snippet = inject_argument(new_snippet, "high")
                elif "'open'" in err_msg:
                    new_snippet = inject_argument(new_snippet, "open")
                
                if new_snippet != meta.code_snippet:
                    meta.code_snippet = new_snippet
                    modified = True

            # Retry if modified
            if modified:
                print("        [Retry] Testing fix...", end=" ")
                try:
                    factory.generate_single_feature(df, meta, {})
                    print("PASS (Fixed)")
                    fixed_count += 1
                except Exception as e2:
                    print(f"FAIL (Fix Failed: {e2})")
                    features_to_remove.append(f_id)
            else:
                # If cannot fix, mark for deletion
                print("        [Action] Cannot auto-fix. Removing feature.")
                features_to_remove.append(f_id)

    # 3. Apply Removals
    if features_to_remove:
        print(f">>> [Doctor] Removing {len(features_to_remove)} broken features: {features_to_remove}")
        with registry._lock:
            for f_id in features_to_remove:
                if f_id in registry._features:
                    del registry._features[f_id]
        deleted_count = len(features_to_remove)

    # 4. Save
    if fixed_count > 0 or deleted_count > 0:
        registry._save_to_disk()
        print(f">>> [Doctor] Registry updated. Fixed: {fixed_count}, Deleted: {deleted_count}")
    else:
        print(">>> [Doctor] No changes needed.")

if __name__ == "__main__":
    validate_and_fix()
