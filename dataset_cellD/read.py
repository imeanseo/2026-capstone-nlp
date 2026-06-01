"""Print original_text column from a summary CSV (default: cell_d_v4_summary.csv)."""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "cell_d_v4_summary.csv",
        help="Path to summary CSV",
    )
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    print(df["original_text"].to_string())


if __name__ == "__main__":
    main()
