python gpt_inference_multi.py \
  --input  ./cell_a_test.csv \
  --output  ./cell_b_test.csv \
  --prompt-keys cell_b_analyze cell_b_rewrite cell_b_check \
  --input-column text_clean \
  --model gpt-4o-mini \
  --n 10
