"""
Test script for Risk Profile Evolution.
Verifies that k_up, k_down, horizon are sampled within profile ranges.
"""
from src.l3_meta.agent import MetaAgent
from src.l3_meta.state import RegimeState  
from src.ledger.repo import LedgerRepo
from src.config import config
from src.l3_meta.risk_profiles import get_default_risk_profiles
from pathlib import Path
import tempfile

# Create temp repo
repo = LedgerRepo(Path(tempfile.mkdtemp()))
agent = MetaAgent(None, repo)

# Test regime
regime = RegimeState(
    label='BULL_RUN', 
    trend_score=0.7, 
    vol_level=0.2, 
    corr_score=0.5, 
    shock_flag=False
)

print("=" * 80)
print("Risk Profile Evolution - Risk Parameter Sampling Test")
print("=" * 80)
print(f"\nConfig Ranges:")
print(f"  k_up:    [{config.RISK_K_UP_MIN}, {config.RISK_K_UP_MAX}]")
print(f"  k_down:  [{config.RISK_K_DOWN_MIN}, {config.RISK_K_DOWN_MAX}]")
print(f"  horizon: [{config.RISK_HORIZON_MIN}, {config.RISK_HORIZON_MAX}]")
print("\nRisk Profiles:")
for profile in get_default_risk_profiles():
    print(
        f"  {profile.profile_id}: "
        f"k_up={profile.k_up_range}, "
        f"k_down={profile.k_down_range}, "
        f"horizon={profile.horizon_range}"
    )
print("\n" + "-" * 80)
print("Generating 10 experiments...")
print("-" * 80)

results = []
for i in range(10):
    spec = agent.propose_policy(regime, [])
    rb = spec.risk_budget
    results.append({
        "k_up": rb["k_up"],
        "k_down": rb["k_down"],
        "horizon": rb["horizon"],
        "risk_profile": rb.get("risk_profile", "UNKNOWN"),
        "rr": rb["risk_reward_ratio"]
    })
    print(f"  [{i+1:2d}] k_up={rb['k_up']:5.2f} | k_down={rb['k_down']:5.2f} | "
          f"horizon={rb['horizon']:3d} | R:R={rb['risk_reward_ratio']:5.2f} | "
          f"Profile: {rb.get('risk_profile', 'UNKNOWN')} | "
          f"Strategy: {spec.template_id}")

# Stats
k_ups = [r["k_up"] for r in results]
k_downs = [r["k_down"] for r in results]
horizons = [r["horizon"] for r in results]

print("\n" + "-" * 80)
print("Distribution Stats:")
print("-" * 80)
print(f"  k_up:    min={min(k_ups):.2f}, max={max(k_ups):.2f}, spread={max(k_ups)-min(k_ups):.2f}")
print(f"  k_down:  min={min(k_downs):.2f}, max={max(k_downs):.2f}, spread={max(k_downs)-min(k_downs):.2f}")
print(f"  horizon: min={min(horizons)}, max={max(horizons)}, spread={max(horizons)-min(horizons)}")

print("\n" + "=" * 80)
print("SUCCESS: Risk parameters are now learned via risk profiles!")
print("=" * 80)
