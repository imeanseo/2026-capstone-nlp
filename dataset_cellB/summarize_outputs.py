"""cell_b_test.csv -> cell_b_summary.csv (original_text, output_text only)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def main() -> None:
    src = ROOT / "cell_b_test.csv"
    dst = ROOT / "cell_b_summary.csv"
    df = pd.read_csv(src)
    out = df[["text_clean", "generated_text"]].copy()
    out.columns = ["original_text", "output_text"]
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"Wrote {dst} ({len(out)} rows)")


if __name__ == "__main__":
    main()
