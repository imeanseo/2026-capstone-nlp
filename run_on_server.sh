#!/bin/bash
# 서버에서 normal_analysis.ipynb 실행 스크립트

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== 서버에서 노트북 실행 스크립트 ===${NC}"

# 1. 파일 전송
echo -e "\n${GREEN}[1/4] 파일 전송 중...${NC}"
scp -P 1002 normal_analysis.ipynb hatexplain_prediction.csv requirements.txt capstone_nlp@203.253.80.84:~/

# 2. SSH 연결 및 환경 설정
echo -e "\n${GREEN}[2/4] 서버 연결 및 환경 설정...${NC}"
ssh capstone_nlp@203.253.80.84 -p 1002 << 'ENDSSH'
# uv로 가상환경 생성 및 패키지 설치
echo "📦 uv로 환경 설정 중..."
uv venv .venv
source .venv/bin/activate

# requirements.txt의 패키지 설치
echo "📥 패키지 설치 중..."
uv pip install pandas numpy matplotlib seaborn jupyter nbconvert

echo "✅ 환경 설정 완료!"
ENDSSH

# 3. 노트북 실행
echo -e "\n${GREEN}[3/4] 노트북 실행 중...${NC}"
ssh capstone_nlp@203.253.80.84 -p 1002 << 'ENDSSH'
source .venv/bin/activate
echo "🚀 노트북 실행 시작..."
jupyter nbconvert --to notebook --execute normal_analysis.ipynb --output normal_analysis_executed.ipynb
echo "✅ 노트북 실행 완료!"
ENDSSH

# 4. 결과 파일 다운로드
echo -e "\n${GREEN}[4/4] 결과 파일 다운로드...${NC}"
scp -P 1002 capstone_nlp@203.253.80.84:~/normal_analysis_executed.ipynb ./
scp -P 1002 "capstone_nlp@203.253.80.84:~/*.png" ./ 2>/dev/null || echo "PNG 파일이 없습니다."

echo -e "\n${BLUE}=== 완료! ===${NC}"
echo "실행된 노트북: normal_analysis_executed.ipynb"
