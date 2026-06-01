#!/usr/bin/env bash
# Cell D는 멀티턴 파이프라인입니다. 단일 --prompt-key 실행 대신 아래를 사용하세요.
set -euo pipefail
cd "$(dirname "$0")"
exec bash ./gpt_inference_multi.sh
