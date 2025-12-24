from __future__ import annotations

import json
import time
import random
import os
import copy
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from src.contracts import LedgerRecord, PolicySpec
from src.shared.logger import get_logger

logger = get_logger("ledger.repo")

class LedgerRepo:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.base_dir / "experiments.jsonl"
        self.artifact_dir = self.base_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def _retry_op(self, func, *args, max_retries=5, **kwargs):
        for i in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (PermissionError, OSError) as e:
                if i == max_retries - 1:
                    raise e
                time.sleep(0.1 + random.random() * 0.3)
        return None

    def save_record(self, record: LedgerRecord, artifact: Optional[Any] = None) -> None:
        """Saves a record to the ledger and persists its artifact bundle."""
        def _append():
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as f:
                data = asdict(record)
                f.write(json.dumps(data) + "\n")
        self._retry_op(_append)
        
        if artifact:
            # ArtifactBundle has its own save method
            # We import it here to avoid circular dep if any
            from src.l2_sl.artifacts import ArtifactBundle
            if isinstance(artifact, ArtifactBundle):
                art_path = self.artifact_dir / f"{record.exp_id}.json"
                self._retry_op(artifact.save, art_path)
            else:
                # Fallback for dict artifacts
                art_path = self.artifact_dir / f"{record.exp_id}.json"
                def _save_dict():
                    with art_path.open("w", encoding="utf-8") as f:
                        json.dump(artifact, f)
                self._retry_op(_save_dict)

    def load_records(self) -> List[LedgerRecord]:
        if not self.ledger_path.exists():
            return []
        def _read():
            records = []
            with self.ledger_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        data = json.loads(line)
                        records.append(self._from_dict(data))
                    except: continue
            return records
        return self._retry_op(_read) or []

    def _from_dict(self, data: Dict[str, Any]) -> LedgerRecord:
        d = copy.deepcopy(data)
        if "policy_spec" in d:
            spec_data = d.pop("policy_spec")
            p = PolicySpec(**spec_data)
        else: p = None
        
        from src.contracts import FixSuggestion
        fix = d.pop("fix_suggestion", None)
        fix_obj = FixSuggestion(**fix) if fix else None
        
        return LedgerRecord(
            policy_spec=p,
            fix_suggestion=fix_obj,
            **d
        )

    def prune_experiments(self, keep_n: int = 100) -> int:
        records = self.load_records()
        if len(records) <= keep_n: return 0
        records.sort(key=lambda r: (r.cpcv_metrics or {}).get("eval_score", -9999.0), reverse=True)
        to_keep = records[0:keep_n]
        to_delete = records[keep_n:]
        tmp_path = self.ledger_path.with_suffix(".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                for r in to_keep: f.write(json.dumps(asdict(r)) + "\n")
            def _replace():
                if self.ledger_path.exists(): os.replace(tmp_path, self.ledger_path)
                else: os.rename(tmp_path, self.ledger_path)
            self._retry_op(_replace)
        except:
            if tmp_path.exists(): tmp_path.unlink()
            return 0
        return len(to_delete)
