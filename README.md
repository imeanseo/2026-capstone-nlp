# HateXplain Normal 데이터 분석

HateXplain 데이터셋의 Normal 라벨(label=1) 샘플 분석 프로젝트

담당: Minseo

---

## 📂 프로젝트 구조

```
capstone_nlp/
├── hatexplain_prediction.csv          # 원본 데이터
├── README.md                           # 이 파일
│
├── scripts/                            # 분석 스크립트
│   ├── analyze_normal.py              # P0 분석 (Base)
│   ├── analyze_p1.py                  # P1 분석 (Surface cue, Disagreement, 어휘)
│   ├── normal_analysis_final.ipynb    # Jupyter 노트북 (백업)
│   └── requirements.txt               # 필요 라이브러리
│
├── results/                            # 분석 결과
│   ├── p0_base/                       # P0 결과
│   │   ├── normal_length_distribution.png
│   │   ├── normal_agreement_distribution.png
│   │   ├── borderline_samples.csv
│   │   └── normal_analysis_summary.csv
│   │
│   └── p1_analysis/                   # P1 결과 (실행 후 생성됨)
│       ├── surface_cue_analysis.png
│       ├── borderline_강함.csv
│       ├── borderline_중간.csv
│       ├── borderline_약함.csv
│       ├── borderline_없음.csv
│       ├── template_cell_c.csv       # Cell C(Implicit Hate) 후보
│       ├── template_cell_d.csv       # Cell D(중립) 후보
│       ├── template_cell_b.csv       # Cell B(일반 부정) 후보
│       └── p1_analysis_summary.csv
│
└── connect_server.sh / run_on_server.sh  # 서버 관련 스크립트
```

---

## 🎯 분석 단계

### ✅ P0: Base 분석 (완료)

**목적:** Normal 샘플의 기본 특성 파악

**분석 항목:**
1. 기본 EDA (문장 길이, 타깃 집단, Agreement 분포)
2. 라벨 구조 분석 (Borderline 케이스)
3. Target 그룹 중심 분석

**실행 방법:**
```bash
cd /Users/imeanseo/Documents/projects/capstone_nlp
python3 scripts/analyze_normal.py
```

**결과:** `results/p0_base/`

---

### ✅ P1: 실험 설계 직결 분석 (완료)

**목적:** Cell 분리 실험 설계를 위한 핵심 패턴 파악

**분석 항목:**
- **4. Surface Cue 분석**
  - 욕설/비속어 vs 비욕설
  - Intensifier/감정어 사전 기반 분석
  - 표면 큐 강도 분류
  
- **7. Annotator Disagreement 분석**
  - Borderline 케이스 심층 분석
  - 소수 의견 패턴 (Offensive vs Hate)
  - Surface cue별 Borderline 샘플
  
- **8. 어휘/구문 분석**
  - 상위 빈출 단어 & Bi-gram
  - 집단 일반화 vs 개인 지시 표현
  - 파생 데이터 템플릿 생성

**실행 방법:**
```bash
cd /Users/imeanseo/Documents/projects/capstone_nlp
python3 scripts/analyze_p1.py

# 결과를 results/p1_analysis/로 자동 이동
python3 scripts/organize_results.py
```

**결과:** `results/p1_analysis/`

---

### ⭐ P1-HurtLex: 전문 혐오 사전 기반 심화 분석 (완료)

**목적:** HurtLex 17개 카테고리로 정교한 Surface Cue 분석

**HurtLex 소개:**
- 50개 이상 언어 지원하는 다국어 혐오 어휘 사전
- 17개 카테고리로 세분화 (인종비하, 성적비하, 장애비하, 경멸어 등)
- 8,228개 영어 단어 포함 (v1.2)

**주요 분석:**
- HurtLex 17개 카테고리별 단어 매칭
- 카테고리별 빈도 및 분포 분석
- HurtLex 커버리지 (Borderline vs 완전 일치)
- **HurtLex 없는 Borderline = 진정한 Implicit Hate 후보**

**실행 방법:**
```bash
cd /Users/imeanseo/Documents/projects/capstone_nlp
python3 scripts/analyze_p1_hurtlex.py
```

**결과:** `results/p1_analysis/`
- `surface_cue_hurtlex_analysis.png`
- `borderline_no_hurtlex.csv` (501개 샘플)

---

## 📊 주요 결과 요약

### P0 결과

| 항목 | 값 |
|------|-----|
| Normal 샘플 수 | 7,814개 (40.6%) |
| 평균 길이 | 23.3 토큰 |
| 완전 일치 (3/3) | 65.6% |
| Borderline (2/3) | 34.4% (2,690개) |
| 타깃 없음 | 47.4% |

**핵심 결론:**
1. Normal 샘플은 평균 23 토큰, 대부분 12-34 토큰 구간
2. **Borderline 2,690개가 Cell C(Implicit Hate) 후보군**
3. 타깃 없음 47.4% → Cell D(중립/긍정) 구성에 적합
4. 매칭 시 길이 ±5 토큰 권장 (중앙값 20 기준)

---

### P1 결과 (기본 + HurtLex)

| 항목 | 기본 방식 | HurtLex 방식 |
|------|-----------|--------------|
| Surface cue 포함 | 23.2% (욕설) | **83.7%** (전체) |
| Borderline 커버리지 | 24.5% | **81.4%** |
| Implicit Hate 후보 | 1,797개 (cue 없음) | **501개** (HurtLex 없음) |

**HurtLex 주요 카테고리 (Normal에서도 출현):**
1. **CDS (40%)** - 경멸어 (Derogatory words)
2. **PS (34%)** - 인종/민족 비하 (Ethnic slurs)
3. **ASM (25%)** - 남성 성기 (Male genitalia)
4. **OM (20%)** - 동성애 (Homosexuality)
5. **RE (19%)** - 범죄/부도덕 (Crime/immoral)

**핵심 발견:**
1. **HurtLex 없는 Borderline 501개** = 진정한 Implicit Hate 후보
   - HurtLex 사전에도 없는 단어로 혐오 표현
   - 맥락 의존적 혐오, 우회적 표현, 신조어 가능성
   
2. **일반화 표현이 암묵적 편향 신호**
   - Borderline: 9.22% vs 완전 일치: 7.81% (+1.41%p)
   - "white people", "all white", "illegal immigrants" 등

3. **실험 설계 준비 완료**
   - Cell C 후보: 1,984개 (일반), 501개 (HurtLex 없음)
   - Cell D 대조군: 3,581개

---

## 🔧 환경 설정

### 필요 라이브러리

```bash
pip install -r scripts/requirements.txt
```

**포함 라이브러리:**
- pandas
- numpy
- matplotlib
- seaborn

---

## 📝 HateXplain 레이블 정의

| 레이블 | 이름 | 정의 |
|--------|------|------|
| **0** | **Hate** | 특정 집단에 대한 직접적 증오 및 해악 의도 |
| **1** | **Normal** | 혐오·공격적 표현이 아닌 문장 (긍정·중립 포함) |
| **2** | **Offensive** | 비하어·욕설 포함, hate만큼 명시적 해악 의도는 없음 |

---

## 🎯 실험 설계 연결

### Cell 분리 전략

| | Target-related 있음 | Target-related 없음 |
|---|---|---|
| **Polarity cue 강함** | Cell A: 명시적 혐오 | Cell B: 일반 부정문 |
| **Polarity cue 약함/없음** | **Cell C: Implicit Hate** | Cell D: 중립/긍정 |

### 데이터 매핑

- **Cell C 후보:** `results/p1_analysis/template_cell_c.csv`
  - Surface cue 약함/없음 + Borderline (2,690개 중 선별)
  
- **Cell D 후보:** `results/p1_analysis/template_cell_d.csv`
  - Surface cue 없음 + 완전 일치
  
- **Cell B 후보:** `results/p1_analysis/template_cell_b.csv`
  - Surface cue 강함(욕설) + 완전 일치

---

## 📌 다음 단계

1. ✅ P0 완료
2. 🔄 P1 실행 (4, 7, 8번)
3. ⏳ Borderline 샘플 육안 검수 (Cell C 후보 정제)
4. ⏳ Hate/Offensive 데이터와 매칭 (minimal pair 구성)
5. ⏳ GPT-5 API로 파생 데이터 생성

---

## 📚 참고 자료

- [HateXplain GitHub](https://github.com/hate-alert/HateXplain)
- [Notion 프로젝트 페이지](https://www.notion.so/3364266e3a8480399377c6caba6d2292)
- [P0 결과 페이지](https://www.notion.so/33d4266e3a8480b8820feb62bea9260f)
- [작업 리스트](https://www.notion.so/33d4266e3a8480348305e25c733c92e0)

---

**마지막 업데이트:** 2026-04-10
