#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Cell D v2.1: [TARGET] unified masking + forbidden tokens + 4-turn pipeline
# .env 파일에 OPENAI_API_KEY 설정 필요

python gpt_inference_v21.py \
  --input ./cell_a_test.csv \
  --output ./cell_d_test_v2_1.csv \
  --prompt-keys cell_d_analyze cell_d_step1 cell_d_step2 cell_d_check \
  --input-column text_clean \
  --model gpt-4o-mini \
  --n 10
