#!/usr/bin/env python3
"""
Latent Hatred → eval_latent_v1 구축.

규칙: experiment/data/eval/mapping_rule.md
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO / "experiment" / "data" / "raw" / "latent_hatred" / "latent_hatred_posts.csv"
EVAL_DIR = REPO / "experiment" / "data" / "eval"
DEFAULT_NORMALIZED = EVAL_DIR / "latent_hatred_normalized.csv"
DEFAULT_OUTPUT = EVAL_DIR / "eval_latent_v1.csv"
DEFAULT_LOG = EVAL_DIR / "eval_latent_v1_build.log"

RANDOM_SEED = 20260528
TARGET_PER_LABEL = 1000

LABEL_MAP = {
    "implicit_hate": "hate",
    "explicit_hate": "hate",
    "not_hate": "non-hate",
}

VALID_CLASSES = frozenset(LABEL_MAP)


def normalize_text(post: str) -> str:
    text = post.strip()
    text = text.replace('""', '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def map_label(class_raw: str) -> str:
    key = class_raw.strip()
    if key not in LABEL_MAP:
        raise ValueError(f"Unknown class: {key!r}")
    return LABEL_MAP[key]


def prepare_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict = {"input_rows": len(df)}

    work = df.copy()
    stats["input_class_distribution"] = (
        work["class"].value_counts().sort_index().to_dict()
    )

    work["text"] = work["post"].astype(str).map(normalize_text)
    work["class_raw"] = work["class"].astype(str).str.strip()
    work["subtype"] = work["class_raw"]

    empty_mask = (work["text"] == "") | (work["class_raw"] == "")
    stats["dropped_empty"] = int(empty_mask.sum())
    work = work.loc[~empty_mask].copy()

    unknown = set(work["class_raw"]) - VALID_CLASSES
    if unknown:
        raise ValueError(f"Unexpected class values: {sorted(unknown)}")

    work["label"] = work["class_raw"].map(LABEL_MAP)
    work["source"] = "latent_hatred"

    stats["after_normalize_rows"] = len(work)
    stats["normalized_class_distribution"] = (
        work["class_raw"].value_counts().sort_index().to_dict()
    )
    stats["normalized_label_distribution"] = (
        work["label"].value_counts().sort_index().to_dict()
    )

    cols = ["text", "class_raw", "label", "subtype", "source"]
    if "implicit_class" in work.columns:
        cols.append("implicit_class")
    return work[cols].reset_index(drop=True), stats


def sample_eval(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict = {}
    hate = df[df["label"] == "hate"]
    non = df[df["label"] == "non-hate"]

    stats["pool_hate"] = len(hate)
    stats["pool_non_hate"] = len(non)

    if len(hate) < TARGET_PER_LABEL or len(non) < TARGET_PER_LABEL:
        raise ValueError(
            f"Insufficient pool for 1:1 sampling "
            f"(need {TARGET_PER_LABEL} each, have hate={len(hate)}, non-hate={len(non)})"
        )

    hate_s = hate.sample(n=TARGET_PER_LABEL, random_state=RANDOM_SEED)
    non_s = non.sample(n=TARGET_PER_LABEL, random_state=RANDOM_SEED)
    out = (
        pd.concat([hate_s, non_s], ignore_index=True)
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    out.insert(0, "id", [f"LH_v1_{i:05d}" for i in range(1, len(out) + 1)])

    eval_cols = ["id", "text", "label", "subtype", "source"]
    out = out[eval_cols]

    stats["eval_rows"] = len(out)
    stats["eval_label_distribution"] = (
        out["label"].value_counts().sort_index().to_dict()
    )
    stats["eval_subtype_distribution"] = (
        out["subtype"].value_counts().sort_index().to_dict()
    )
    stats["random_seed"] = RANDOM_SEED
    return out, stats


def write_log(path: Path, normalize_stats: dict, sample_stats: dict, input_path: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"=== eval_latent_v1 build log ({ts}) ===",
        "",
        f"input: {input_path.relative_to(REPO)}",
        f"random_seed: {RANDOM_SEED}",
        f"target_per_label: {TARGET_PER_LABEL}",
        "",
        f"input_rows: {normalize_stats['input_rows']}",
        "input_class_distribution:",
    ]
    for k, v in normalize_stats["input_class_distribution"].items():
        lines.append(f"  {k}: {v}")
    lines += [
        "",
        f"dropped_empty: {normalize_stats['dropped_empty']}",
        f"after_normalize_rows: {normalize_stats['after_normalize_rows']}",
        "normalized_class_distribution:",
    ]
    for k, v in normalize_stats["normalized_class_distribution"].items():
        lines.append(f"  {k}: {v}")
    lines += ["", "normalized_label_distribution:"]
    for k, v in normalize_stats["normalized_label_distribution"].items():
        lines.append(f"  {k}: {v}")
    lines += [
        "",
        f"pool_hate: {sample_stats['pool_hate']}",
        f"pool_non_hate: {sample_stats['pool_non_hate']}",
        f"eval_rows: {sample_stats['eval_rows']}",
        "eval_label_distribution:",
    ]
    for k, v in sample_stats["eval_label_distribution"].items():
        lines.append(f"  {k}: {v}")
    lines += ["", "eval_subtype_distribution:"]
    for k, v in sample_stats["eval_subtype_distribution"].items():
        lines.append(f"  {k}: {v}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build eval_latent_v1.csv")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--normalized-out", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Missing input: {args.input}", file=sys.stderr)
        sys.exit(1)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input)
    normalized, norm_stats = prepare_frame(raw)
    normalized.to_csv(args.normalized_out, index=False, encoding="utf-8-sig")
    print(f"Wrote {args.normalized_out} ({len(normalized)} rows)")

    eval_df, sample_stats = sample_eval(normalized)
    eval_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {args.output} ({len(eval_df)} rows)")

    write_log(args.log, norm_stats, sample_stats, args.input)
    print(f"Log saved: {args.log}")


if __name__ == "__main__":
    main()
