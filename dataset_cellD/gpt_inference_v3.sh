#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Cell D v3: NP-level masking + BOUNDARY box + wrapper fallback
# .env 파일에 OPENAI_API_KEY 설정 필요

python gpt_inference_v3.py \
  --input ./cell_a_test.csv \
  --output ./cell_d_test_v3.csv \
  --input-column text_clean \
  --model gpt-4o-mini \
  --n 10
