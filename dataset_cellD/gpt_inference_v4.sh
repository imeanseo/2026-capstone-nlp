python gpt_inference_v4.py \
  --input  ./cell_a_test.csv \
  --output  ./cell_d_test_v4.csv \
  --prompt-keys cell_d_analyze cell_d_step1 cell_d_step2 cell_d_check \
  --input-column text_clean \
  --model gpt-4o-mini \
  --n 10
