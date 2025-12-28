import argparse

from src.config import config
from src.data.loader import DataLoader
from src.testing.eval_harness import run_eval_harness


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Eval Harness scenarios.")
    parser.add_argument("--ticker", default=config.TARGET_TICKER)
    parser.add_argument("--start", default=config.DATA_START_DATE)
    args = parser.parse_args()

    loader = DataLoader(target_ticker=args.ticker, start_date=args.start)
    df = loader.fetch_all()
    if df.empty:
        raise ValueError("Loaded data is empty")

    run_eval_harness(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
