import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from src.config import config
from src.testing.tv_calibration import run_calibration


SCENARIO_FILES = {
    "timing": ("timing_dataset.csv", "timing_reference.json"),
    "cost": ("cost_dataset.csv", "cost_reference.json"),
    "collision": ("collision_dataset.csv", "collision_reference.json"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="TradingView calibration runner")
    parser.add_argument("--scenario", default="all", help="timing, cost, collision, or all")
    parser.add_argument("--data-dir", default=config.TV_CALIBRATION_DATA_DIR, help="CSV dataset directory")
    parser.add_argument("--ref-dir", default=config.TV_CALIBRATION_REFERENCE_DIR, help="Reference JSON directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ref_dir = Path(args.ref_dir)

    scenarios = list(SCENARIO_FILES.keys())
    if args.scenario != "all":
        scenarios = [args.scenario]

    reports = {}
    for scenario in scenarios:
        if scenario not in SCENARIO_FILES:
            raise ValueError(f"Unknown scenario: {scenario}")
        dataset_file, ref_file = SCENARIO_FILES[scenario]
        report = run_calibration(
            scenario=scenario,
            csv_path=data_dir / dataset_file,
            reference_path=ref_dir / ref_file,
        )
        reports[scenario] = asdict(report)
        status = "PASS" if report.passed else "FAIL"
        print(f"[{scenario}] {status} mode={report.mode} mismatches={len(report.mismatches)}")

    output_dir = Path(config.LOG_DIR) / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tv_calibration_{int(time.time())}.json"
    output_path.write_text(json.dumps(reports, indent=2))
    print(f"Saved calibration report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
