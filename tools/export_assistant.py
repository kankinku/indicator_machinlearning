"""
Export Assistant Tool
학습된 모델이나 전체 학습 상태(Brain)를 다른 곳으로 옮기기 쉽게 압축(zip)해주는 도구입니다.

사용법:
    # 1. 최고 성능 모델 하나만 추출 (배포용)
    python tools/export_assistant.py --mode best

    # 2. 전체 학습 데이터 백업 (이사용)
    python tools/export_assistant.py --mode full
"""
import os
import sys
import json
import argparse
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from src.config import config
from src.ledger.repo import LedgerRepo

def create_export_dir():
    export_dir = project_root / "export"
    export_dir.mkdir(exist_ok=True)
    return export_dir

def export_full_backup(ledger_dir: Path, export_dir: Path):
    """전체 ledger 폴더를 압축합니다."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = export_dir / f"vibe_brain_backup_{timestamp}.zip"
    
    print(f">>> [Backup] Compressing entire ledger from: {ledger_dir}")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(ledger_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(ledger_dir.parent) # ledger/ 부터 시작
                zipf.write(file_path, arcname)
                
    print(f">>> [Backup] Success! Saved to: {zip_filename}")
    print(f"    (이 파일을 다른 PC의 프로젝트 루트에 풀면 학습 상태가 복원됩니다.)")

def export_best_model(repo: LedgerRepo, export_dir: Path):
    """가장 성능이 좋은 모델 하나를 추출합니다."""
    print(">>> [Export] Searching for the best candidate...")
    
    # 1. Load Records & Sort
    records = repo.load_records()
    if not records:
        print("!!! [Error] No experiments found in ledger.")
        return

    # 간단한 랭킹 로직 (LedgerRepo의 로직 일부 차용하거나 단순화)
    # 여기서는 Holistic Score가 있다면 그걸 쓰고, 없다면 Sharpe 우선
    def get_score(rec):
        metrics = rec.cpcv_metrics or {}
        return metrics.get("cpcv_mean", -999.0) # Sharpe Mean

    best_record = max(records, key=get_score)
    
    exp_id = best_record.exp_id
    score = get_score(best_record)
    print(f">>> [Export] Best Model Found!")
    print(f"    ID: {exp_id}")
    print(f"    Algorithm: {best_record.policy_spec.template_id}")
    print(f"    Score (Sharpe): {score:.4f}")
    
    # 2. Identify Files
    files_to_zip = []
    
    # Artifacts
    if best_record.model_artifact_ref:
        artifact_path = Path(best_record.model_artifact_ref)
        if artifact_path.exists():
            files_to_zip.append(artifact_path)
            
            # Related files (like results csv if any, though usually inside bundle or separate)
            # Check for LightGBM model folder if applicable
            # The ArtifactBundle saves a .json meta file, and possibly a folder if MLGuard was saved separately.
            # But currently ArtifactBundle dumps to a single JSON usually, OR references a folder.
            # Let's check if there's a folder with the same ID.
            possible_folder = repo.artifact_dir / str(exp_id)
            if possible_folder.exists() and possible_folder.is_dir():
                 for root, dirs, files in os.walk(possible_folder):
                    for f in files:
                        files_to_zip.append(Path(root) / f)
        else:
            print(f"!!! [Warning] Artifact file missing: {artifact_path}")
    
    if not files_to_zip:
        print("!!! [Error] No artifact files found to export.")
        return

    # 3. Create Zip
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = export_dir / f"best_model_{best_record.policy_spec.template_id}_{timestamp}.zip"
    
    print(f">>> [Export] Packaging model files...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add Artifact(s)
        for file_path in files_to_zip:
            # zip 내 경로를 깔끔하게 'artifacts/' 아래로 정리
            try:
                arcname = str(file_path.relative_to(repo.base_dir))
            except ValueError:
                # base_dir 외부에 있다면(그럴리 없지만) 파일명만
                arcname = f"artifacts/{file_path.name}"
            zipf.write(file_path, arcname)
            
        # Add Metadata (Experiment Info) as a separate JSON for reference
        meta_str = json.dumps(best_record.__dict__, default=str, indent=4)
        zipf.writestr("model_manifest.json", meta_str)

    print(f">>> [Export] Success! Saved to: {zip_filename}")
    print(f"    (이 압축 파일은 모델 가중치와 설정값을 포함합니다.)")

def main():
    parser = argparse.ArgumentParser(description="Vibe Export Assistant")
    parser.add_argument("--mode", choices=["full", "best"], default="full", help="Export mode: 'full' (backup) or 'best' (deployment)")
    args = parser.parse_args()
    
    ledger_dir = Path(config.LEDGER_DIR)
    repo = LedgerRepo(ledger_dir)
    pass
    
    export_dir = create_export_dir()
    
    if args.mode == "full":
        export_full_backup(ledger_dir, export_dir)
    elif args.mode == "best":
        export_best_model(repo, export_dir)

if __name__ == "__main__":
    main()
