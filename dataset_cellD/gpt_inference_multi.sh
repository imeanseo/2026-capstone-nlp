#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Cell D: 분석 → step1 → step2 → 검증 (4턴)
# .env 파일에 OPENAI_API_KEY 설정 필요 (python-dotenv가 자동 로드)
python gpt_inference_multi.py \
  --input ./cell_a_test.csv \
  --output ./cell_d_test.csv \
  --prompt-keys cell_d_analyze cell_d_step1 cell_d_step2 cell_d_check \
  --input-column text_clean \
  --model gpt-4o-mini \
  --n 10
