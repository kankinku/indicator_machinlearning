from __future__ import annotations
from typing import List, Dict, Any, Optional
from src.shared.event_bus import EventRecorder, Event

class AssertionEngine:
    def __init__(self, events: List[Event]):
        self.events = events

    def assert_event_count(self, event_type: str, count: int, msg: str = ""):
        actual = len([e for e in self.events if e.event_type == event_type])
        if actual != count:
            raise AssertionError(f"{msg or event_type}: Expected {count}, got {actual}")

    def assert_event_occurred(self, event_type: str, msg: str = ""):
        actual = len([e for e in self.events if e.event_type == event_type])
        if actual == 0:
            raise AssertionError(f"{msg or event_type}: Event did not occur")

    def assert_event_not_occurred(self, event_type: str, msg: str = ""):
        actual = len([e for e in self.events if e.event_type == event_type])
        if actual > 0:
            raise AssertionError(f"{msg or event_type}: Event occurred but was forbidden")

    def assert_sequence(self, event_types: List[str]):
        # Implementation to check if events occurred in a specific order
        pass

class TestRunner:
    def __init__(self):
        self.recorder = EventRecorder()

    def run_case(self, name: str, scenario: Any, workflow_fn: Any):
        print(f"=== Running Test Case: {name} ===")
        self.recorder.activate()
        
        try:
            # Set global test context for scenarios
            from src.testing.context import set_test_context
            set_test_context(scenario)
            
            # Run the actual system loop or a subset
            workflow_fn()
            
            # Validate results
            events = self.recorder.get_events()
            self.validate(name, events)
            print(f"✅ Test Case {name} PASSED")
            
        except AssertionError as e:
            print(f"❌ Test Case {name} FAILED: {str(e)}")
            self.print_timeline()
        except Exception as e:
            print(f"💥 Test Case {name} CRASHED: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.recorder.deactivate()
            from src.testing.context import clear_test_context
            clear_test_context()

    def validate(self, name: str, events: List[Event]):
        engine = AssertionEngine(events)
        if name == "STAGE1_DROP" or name == "STANDARD_RUN_DEMO":
            engine.assert_event_occurred("POLICY_PROPOSED")
            if name == "STAGE1_DROP":
                engine.assert_event_occurred("POLICY_FAILED")
        elif name == "FORCE_RIGID_TEST":
            engine.assert_event_occurred("AUTOTUNER_RIGID_DETECTED")
            engine.assert_event_occurred("AUTOTUNER_INTERVENTION_APPLIED")

    def print_timeline(self):
        events = self.recorder.get_events()
        print("\n--- Event Timeline ---")
        for i, e in enumerate(events):
            policy_info = f" | {e.policy_id[:8]}" if e.policy_id else ""
            print(f"{i:03d}: {e.event_type:<20} | {e.stage or 'GLOBAL':<10}{policy_info}")
        print("----------------------\n")
