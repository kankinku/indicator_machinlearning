from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from src.contracts import LedgerRecord
from src.l2_sl.artifacts import ArtifactBundle


class LedgerRepo:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.base_dir / "experiments.jsonl"
        self.artifact_dir = self.base_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def save_record(self, record: LedgerRecord, artifact: Optional[ArtifactBundle] = None) -> None:
        if artifact:
            artifact_path = self.artifact_dir / f"{record.exp_id}.json"
            artifact.save(artifact_path)
            record.model_artifact_ref = str(artifact_path)

        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def load_records(self) -> List[LedgerRecord]:
        if not self.ledger_path.exists():
            return []
        records: List[LedgerRecord] = []
        with self.ledger_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    records.append(self._from_dict(raw))
                except Exception:
                    continue
        return records

    @staticmethod
    def _from_dict(data: dict) -> LedgerRecord:
        from src.contracts import FixSuggestion, PolicySpec  # lazy import to avoid cycles

        fix = data.get("fix_suggestion")
        fix_obj = FixSuggestion(**fix) if fix else None
        policy_spec = PolicySpec(**data["policy_spec"])
        return LedgerRecord(
            exp_id=data["exp_id"],
            timestamp=data.get("timestamp", 0.0),
            policy_spec=policy_spec,
            data_hash=data["data_hash"],
            feature_hash=data["feature_hash"],
            label_hash=data["label_hash"],
            model_artifact_ref=data.get("model_artifact_ref", ""),
            cpcv_metrics=data.get("cpcv_metrics", {}),
            pbo=data.get("pbo", 0.0),
            risk_report=data.get("risk_report", {}),
            reason_codes=data.get("reason_codes", []),
            fix_suggestion=fix_obj,
        )
