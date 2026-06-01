

set -e   # 한 단계라도 실패하면 멈춤

if [ -z "$1" ]; then
  echo "Usage: bash run.sh <GPU_ID> [stage] [extra args...]"
  echo "  stage: week1 | week2 | week3 | all (default all)"
  exit 1
fi

GPU=$1
STAGE=${2:-all}
shift 2 2>/dev/null || shift 1   # GPU만 줬을 때 안전 처리
EXTRA="$@"

export CUDA_VISIBLE_DEVICES=$GPU
echo "[run.sh] CUDA_VISIBLE_DEVICES=$GPU  stage=$STAGE  extra='$EXTRA'"

case $STAGE in
  week1)
    python week1_pipeline.py $EXTRA
    ;;
  week2)
    python week2_sweep.py $EXTRA
    ;;
  week3)
    python week3_analysis.py $EXTRA
    ;;
  all)
    echo "[run.sh] === week1 ==="
    python week1_pipeline.py $EXTRA
    echo "[run.sh] === week2 ==="
    python week2_sweep.py $EXTRA
    echo "[run.sh] === week3 ==="
    python week3_analysis.py $EXTRA
    ;;
  *)
    echo "unknown stage: $STAGE (week1/week2/week3/all)"
    exit 1
    ;;
esac

echo "[run.sh] done."
