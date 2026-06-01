#!/usr/bin/env bash
# Cell B v15 — analyze → rewrite → check
cd "$(dirname "$0")"

PYTHON="${PYTHON:-../.venv/bin/python3}"

"$PYTHON" gpt_inference_v15.py \
  --input ../dataset_2/cell_a_high_quality.csv \
  --output ../dataset_cellB/cell_b_v15_hq_full.csv \
  --prompt-keys analyze rewrite check \
  --model gpt-4o \
  --n 0
