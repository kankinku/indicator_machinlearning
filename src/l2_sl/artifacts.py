from __future__ import annotations

import json
import joblib
import pandas as pd
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List


@dataclass
class ArtifactBundle:
    label_config: Dict[str, Any]
    direction_model: Any  # Can be dict (legacy) or sklearn estimator
    risk_model: Dict[str, Any]
    calibration_metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    backtest_results: Optional[List[Dict[str, Any]]] = None  # Detailed trade log / equity curve

    def to_dict(self) -> Dict[str, Any]:
        # Shallow copy to avoid mutating actual object during serialization prep
        data = asdict(self)
        # Avoid serializing the complex model object into JSON
        if not isinstance(self.direction_model, dict):
             data["direction_model"] = "serialized_external"
        
        # Avoid serializing potentially large backtest results into main JSON
        if self.backtest_results is not None:
             data["backtest_results"] = "saved_as_csv"
             
        return data

    def save(self, path: Path) -> None:
        """
        Saves the artifact bundle.
        - valid JSON parts go to `path` (e.g. {exp_id}.json)
        - binary model parts go to `path.with_suffix('.joblib')`
        - backtest results go to `path` dir with suffix `_results.csv`
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Handle Binary Model
        model_ref = "stub_dict"
        if not isinstance(self.direction_model, dict):
            model_path = path.with_suffix(".joblib")
            joblib.dump(self.direction_model, model_path)
            model_ref = model_path.name
            
        # 2. Handle Backtest Results (CSV)
        results_ref = None
        if self.backtest_results:
            results_path = path.parent / f"{path.stem}_results.csv"
            df_res = pd.DataFrame(self.backtest_results)
            # If 'index' is not in columns, reset index to save time info if it was the index
            # But here backtest_results is usually list of dicts. We assume 'date' or time is a column.
            # If not, caller must ensure list of dicts has time.
            df_res.to_csv(results_path, index=True) 
            results_ref = results_path.name

        # 3. Prepare JSON Data
        data = self.to_dict()
        data["direction_model_ref"] = model_ref
        if results_ref:
            data["backtest_results_ref"] = results_ref
        
        # 4. Save JSON
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: Path) -> "ArtifactBundle":
        # 1. Load JSON
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            
        # 2. Load Binary Model if needed
        model_ref = data.get("direction_model_ref", "stub_dict")
        if model_ref != "stub_dict":
            model_path = path.parent / model_ref
            if model_path.exists():
                model = joblib.load(model_path)
                data["direction_model"] = model
            else:
                data["direction_model"] = {"error": "model_file_missing"}
        
        # 3. Load Backtest Results if needed
        results_ref = data.get("backtest_results_ref", None)
        if results_ref:
             results_path = path.parent / results_ref
             if results_path.exists():
                 df_res = pd.read_csv(results_path, index_col=0)
                 data["backtest_results"] = df_res.to_dict(orient="records")
        
        # Cleanup transient keys
        keys_to_remove = ["direction_model_ref", "backtest_results_ref"]
        for k in keys_to_remove:
            if k in data:
                del data[k]
            
        return ArtifactBundle(**data)
