from __future__ import annotations
from typing import Optional
from src.testing.scenario import Scenario

_active_scenario: Optional[Scenario] = None

def set_test_context(scenario: Scenario):
    global _active_scenario
    _active_scenario = scenario

def get_test_context() -> Optional[Scenario]:
    return _active_scenario

def clear_test_context():
    global _active_scenario
    _active_scenario = None

def is_test_mode() -> bool:
    return _active_scenario is not None
