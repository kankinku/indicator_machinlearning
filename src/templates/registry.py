from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from src.contracts import StrategyTemplate, TunableParamSpec, ValidationError


@dataclass
class TemplateRegistry:
    templates: Dict[str, StrategyTemplate]

    def list_templates(self) -> List[StrategyTemplate]:
        return list(self.templates.values())

    def get_template(self, template_id: str) -> StrategyTemplate:
        if template_id not in self.templates:
            raise ValidationError(f"Unknown template: {template_id}")
        return self.templates[template_id]

    def validate_params(self, template_id: str, tuned_params: Dict[str, object]) -> Dict[str, object]:
        template = self.get_template(template_id)
        return template.validate_params(tuned_params)

    def required_features(self, template_id: str) -> List[str]:
        return self.get_template(template_id).required_features


def _range_param(name: str, min_v: float, max_v: float, default: float, param_type: str = "float") -> TunableParamSpec:
    return TunableParamSpec(name=name, param_type=param_type, min=min_v, max=max_v, default=default)


def _categorical_param(name: str, choices: Iterable[str], default: str) -> TunableParamSpec:
    return TunableParamSpec(name=name, param_type="categorical", choices=list(choices), default=default)


def default_registry() -> TemplateRegistry:
    templates: Dict[str, StrategyTemplate] = {}

    templates["T01"] = StrategyTemplate(
        template_id="T01",
        name="Trend-follow vol-scaling",
        required_features=["price_action", "volatility", "regime"],
        labeling_family="vol-scaling",
        model_family="gbdt",
        risk_family="vol-target",
        tunable_params_schema=[
            _range_param("k", 0.8, 2.5, 1.2),
            _range_param("H", 10, 60, 30, param_type="int"),
            _range_param("entry_threshold", 0.5, 0.7, 0.55),
        ],
        base_constraints={"min_holding": 3, "turnover_cap": 0.35},
    )

    templates["T02"] = StrategyTemplate(
        template_id="T02",
        name="Mean-reversion tight barrier",
        required_features=["price_action", "volatility", "microstructure"],
        labeling_family="barrier",
        model_family="gbdt",
        risk_family="vol-target",
        tunable_params_schema=[
            _range_param("k", 0.5, 1.5, 0.9),
            _range_param("H", 5, 30, 10, param_type="int"),
            _range_param("entry_threshold", 0.52, 0.7, 0.6),
        ],
        base_constraints={"min_holding": 2, "turnover_cap": 0.5},
    )

    templates["T03"] = StrategyTemplate(
        template_id="T03",
        name="Breakout with regime filter",
        required_features=["price_action", "volatility", "regime"],
        labeling_family="barrier",
        model_family="gbdt",
        risk_family="dd-guard",
        tunable_params_schema=[
            _range_param("k", 1.0, 3.0, 1.8),
            _range_param("H", 15, 80, 40, param_type="int"),
            _range_param("entry_threshold", 0.55, 0.75, 0.6),
        ],
        base_constraints={"regime_filter": "low_vol", "turnover_cap": 0.4, "min_holding": 4},
    )

    templates["T04"] = StrategyTemplate(
        template_id="T04",
        name="Vol expansion risk-aware",
        required_features=["volatility", "tail_risk", "regime"],
        labeling_family="vol-scaling",
        model_family="linear",
        risk_family="dd-guard",
        tunable_params_schema=[
            _range_param("k", 0.8, 2.0, 1.1),
            _range_param("H", 10, 50, 20, param_type="int"),
            _categorical_param("vol_model", ["ewma", "garch"], "ewma"),
        ],
        base_constraints={"max_dd": 0.12, "turnover_cap": 0.25},
    )

    templates["T05"] = StrategyTemplate(
        template_id="T05",
        name="Carry-like low vol only",
        required_features=["carry", "volatility", "regime"],
        labeling_family="horizon",
        model_family="linear",
        risk_family="fixed-fraction",
        tunable_params_schema=[
            _range_param("H", 5, 40, 20, param_type="int"),
            _range_param("entry_threshold", 0.51, 0.65, 0.55),
            _range_param("vol_cap", 0.1, 0.4, 0.2),
        ],
        base_constraints={"regime_filter": "low_vol", "min_holding": 5},
    )

    templates["T06"] = StrategyTemplate(
        template_id="T06",
        name="Range trading with freq cap",
        required_features=["price_action", "liquidity", "microstructure"],
        labeling_family="horizon",
        model_family="gbdt",
        risk_family="fixed-fraction",
        tunable_params_schema=[
            _range_param("H", 5, 25, 12, param_type="int"),
            _range_param("entry_threshold", 0.5, 0.68, 0.55),
            _range_param("stop_width", 0.5, 1.5, 1.0),
        ],
        base_constraints={"trade_frequency_cap": 20, "min_holding": 2},
    )

    templates["T07"] = StrategyTemplate(
        template_id="T07",
        name="Event-aware defensive",
        required_features=["event_calendar", "volatility", "regime"],
        labeling_family="barrier",
        model_family="linear",
        risk_family="dd-guard",
        tunable_params_schema=[
            _range_param("k", 0.8, 2.0, 1.0),
            _range_param("H", 5, 30, 10, param_type="int"),
            _categorical_param("event_exposure", ["avoid", "neutral"], "avoid"),
        ],
        base_constraints={"event_blackout": True, "max_dd": 0.1},
    )

    templates["T08"] = StrategyTemplate(
        template_id="T08",
        name="Defensive drawdown guard",
        required_features=["price_action", "volatility", "risk_regime"],
        labeling_family="vol-scaling",
        model_family="linear",
        risk_family="dd-guard",
        tunable_params_schema=[
            _range_param("k", 0.6, 1.5, 0.9),
            _range_param("H", 5, 25, 10, param_type="int"),
            _range_param("entry_threshold", 0.52, 0.65, 0.57),
        ],
        base_constraints={"max_dd": 0.08, "turnover_cap": 0.2, "min_holding": 3},
    )

    return TemplateRegistry(templates=templates)

