#!/usr/bin/env bash
# MacBook (Apple Silicon MPS) 로컬 실행용
# Usage: bash run_mac.sh [stage] [extra args...]
#   stage: layer-probe | random-seed | qual-examples | week1 | week2 | week3 | all (default layer-probe)

set -e

STAGE=${1:-layer-probe}
shift 1 2>/dev/null || true
EXTRA="$@"

echo "[run_mac.sh] stage=$STAGE  extra='$EXTRA'"

case $STAGE in
  layer-probe)
    # §1 Layer별 Linear Probe (~15분, latent만)
    python week3_analysis.py --layer-probe-only --eval latent --batch 16 $EXTRA
    ;;
  probe-swap)
    python build_probe_train_csvs.py
    python week3_analysis.py --probe-swap-only --batch 16 $EXTRA
    ;;
  random-seed)
    # B1 random vector seed 반복 (~25분/eval, latent만 권장)
    python -u week3_analysis.py --random-seed-only --eval latent --batch 16 $EXTRA
    ;;
  qual-examples)
    # 정성평가 예시 선정 (~8분/eval)
    python -u extract_qualitative_examples.py --eval latent --batch 16 $EXTRA
    ;;
  week1)
    python week1_pipeline.py --batch 16 $EXTRA
    ;;
  week2)
    python week2_sweep.py --batch 16 --eval latent $EXTRA
    ;;
  week3)
    python week3_analysis.py --eval latent --batch 16 $EXTRA
    ;;
  all)
    echo "[run_mac.sh] === week1 ==="
    python week1_pipeline.py --batch 16 $EXTRA
    echo "[run_mac.sh] === week2 ==="
    python week2_sweep.py --batch 16 --eval latent $EXTRA
    echo "[run_mac.sh] === week3 ==="
    python week3_analysis.py --eval latent --batch 16 $EXTRA
    ;;
  *)
    echo "unknown stage: $STAGE (layer-probe/random-seed/qual-examples/week1/week2/week3/all)"
    exit 1
    ;;
esac

echo "[run_mac.sh] done."
