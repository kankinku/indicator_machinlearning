# Config Boundaries (Proposed, No Refactor Yet)

This document defines the intended ownership boundaries for configuration keys.
Do not move any config yet; use this as the target map for future refactors.

ObservabilityConfig
- OBS_* metrics, bins, thresholds
- REGRESSION_* thresholds
- EVAL_HARNESS_* settings

CurriculumConfig (Stage SSOT)
- CURRICULUM_* and StageSpec fields
- STAGE_AUTO_* rules
- EVAL_CAPS_BY_STAGE (compute caps tied to stage)

GateConfig
- VAL_* hard gate thresholds
- LEARNING_GATE_* (soft gate distances/weights)
- REJECT_* penalty settings
- SIGNAL_DEGENERATE_* thresholds

SelectionConfig
- SELECTION_* weights
- DIVERSITY_* thresholds
- V12_ELITE_TOP_N, EVAL_FAST_TOP_PCT (selection mix)

MutationConfig
- MUTATION_* weights and biases
- AUTOTUNE_* mutation levers
