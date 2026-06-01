#!/usr/bin/env bash
# Mac 로컬 가상환경 + 최소 의존성 설치
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo "[setup] .venv 생성 완료"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install -U pip wheel
pip install -r requirements.txt

python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"mps_available={torch.backends.mps.is_available()}")
print(f"cuda_available={torch.cuda.is_available()}")
PY

echo
echo "[setup] 완료. 활성화: source .venv/bin/activate"
echo "[setup] Layer probe 실행:"
echo "  python week3_analysis.py --layer-probe-only --eval latent --batch 16"
