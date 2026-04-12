#!/bin/bash
# 빠른 실행 - 서버에 직접 연결해서 작업

echo "🔗 서버 연결 중..."
echo "📍 위치: 203.253.80.84:1002"
echo ""
echo "서버에서 다음 명령어들을 실행하세요:"
echo ""
echo "# 1️⃣ 가상환경 생성 (최초 1회만)"
echo "uv venv .venv"
echo ""
echo "# 2️⃣ 가상환경 활성화"
echo "source .venv/bin/activate"
echo ""
echo "# 3️⃣ 필요한 패키지 설치"
echo "uv pip install pandas numpy matplotlib seaborn jupyter nbconvert"
echo ""
echo "# 4️⃣ 노트북 실행"
echo "jupyter nbconvert --to notebook --execute normal_analysis.ipynb --output normal_analysis_result.ipynb"
echo ""
echo "또는 Python 스크립트로 실행:"
echo "jupyter nbconvert --to script normal_analysis.ipynb"
echo "python normal_analysis.py"
echo ""
echo "=========================================="
echo ""

# SSH 연결
ssh capstone_nlp@203.253.80.84 -p 1002
