from __future__ import annotations

import json
import time
import random
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from src.contracts import LedgerRecord
from src.shared.ranking import check_return_stability, return_rank_key
from src.shared.returns import compute_compounded_return_pct, get_risk_unit
from src.l2_sl.artifacts import ArtifactBundle


class LedgerRepo:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.base_dir / "experiments.jsonl"
        self.artifact_dir = self.base_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def _retry_op(self, func, *args, **kwargs):
        """Retries a file operation on PermissionError (common on Windows)."""
        max_retries = 10
        for i in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (PermissionError, OSError) as e:
                # WinError 32: The process cannot access the file because it is being used by another process.
                if i == max_retries - 1:
                    print(f"[Repo] File access failed after {max_retries} retries: {e}")
                    raise
                time.sleep(0.1 + random.random() * 0.3)  # Jitter

    def save_record(self, record: LedgerRecord, artifact: Optional[ArtifactBundle] = None) -> None:
        if artifact:
            artifact_path = self.artifact_dir / f"{record.exp_id}.json"
            artifact.save(artifact_path)
            record.model_artifact_ref = str(artifact_path)

        def _append():
            with self.ledger_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        
        self._retry_op(_append)

    def load_records(self) -> List[LedgerRecord]:
        if not self.ledger_path.exists():
            return []
            
        def _read():
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

        return self._retry_op(_read)

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
            verdict_dump=data.get("verdict_dump"),
        )

    def prune_experiments(self, keep_n: int = 100) -> int:
        """
        Prune experiments, keeping only the top N performers.
        Uses atomic write pattern (write tmp -> rename) to avoid race conditions.
        """
        records = self.load_records()
        if len(records) <= keep_n:
            return 0
            
        def get_total_return(r: LedgerRecord) -> float:
            risk_budget = r.policy_spec.risk_budget or {}
            total_return = compute_compounded_return_pct(
                exp_id=r.exp_id,
                model_artifact_ref=r.model_artifact_ref,
                ledger_dir=self.base_dir,
                risk_budget=risk_budget,
            )
            if total_return is None:
                ret_mean = r.cpcv_metrics.get("cpcv_mean", 0.0)
                n_trades = r.cpcv_metrics.get("n_trades", 0)
                risk_unit = get_risk_unit(risk_budget)
                return ret_mean * n_trades * (risk_unit * 100.0)
            return float(total_return)

        def get_rank_key(r: LedgerRecord):
            metrics = r.cpcv_metrics or {}
            eval_score = metrics.get("eval_score")
            ret_mean = metrics.get("cpcv_mean", 0.0)
            vol_std = metrics.get("cpcv_std", 0.0)
            cpcv_worst = metrics.get("cpcv_worst", 0.0)
            n_trades = metrics.get("n_trades", 0)
            win_rate = metrics.get("win_rate", 0.0)
            stability_pass, vol_ratio = check_return_stability(
                cpcv_mean=ret_mean,
                cpcv_std=vol_std,
                cpcv_worst=cpcv_worst,
                win_rate=win_rate,
                n_trades=n_trades,
            )
            if eval_score is not None:
                try:
                    return (
                        1.0 if stability_pass else 0.0,
                        float(eval_score),
                        float(win_rate),
                        float(n_trades),
                        -float(vol_ratio),
                    )
                except (TypeError, ValueError):
                    pass
            total_return = get_total_return(r)
            return return_rank_key(
                total_return=total_return,
                stability_pass=stability_pass,
                win_rate=win_rate,
                n_trades=n_trades,
                vol_ratio=vol_ratio,
            )

        # Sort: Approved first, then by return-centric stable ranking
        records.sort(
            key=lambda r: ((1 if not r.reason_codes else 0),) + get_rank_key(r),
            reverse=True,
        )
        
        to_keep = records[:keep_n]
        to_delete = records[keep_n:]
        
        # 1. Rewrite JSONL Atomically
        tmp_path = self.ledger_path.with_suffix(".tmp")
        
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                for rec in to_keep:
                    f.write(json.dumps(asdict(rec)) + "\n")
            
            # Atomic replace
            def _replace():
                if self.ledger_path.exists():
                    os.replace(tmp_path, self.ledger_path)
                else:
                    os.rename(tmp_path, self.ledger_path)
            
            self._retry_op(_replace)
            
        except Exception as e:
            # Clean up tmp if failed
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except:
                    pass
            raise e
                
        # 2. Delete Artifacts
        deleted_count = 0
        for rec in to_delete:
            deleted_count += 1
            # Model Artifact
            if rec.model_artifact_ref:
                p = Path(rec.model_artifact_ref)
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass
            
            # Results CSV (implied path)
            res_csv = self.artifact_dir / f"{rec.exp_id}_results.csv"
            if res_csv.exists():
                try:
                    res_csv.unlink()
                except Exception:
                    pass
                    
        return deleted_count
