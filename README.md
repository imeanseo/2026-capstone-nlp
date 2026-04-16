# Hatexplain Normal 분석 프로젝트

## 📌 프로젝트 개요

**목적**: Hatexplain Normal 데이터셋에서 Borderline(의견 갈림) vs 만장일치 Normal 분류 및 암묵적 혐오 표현 분석

**데이터**: 
- 전체: 7,814개 Normal 샘플
- Borderline (2-4명 동의): 2,690개 (34.4%)
- 만장일치 (5명 동의): 5,124개 (65.6%)

---

## 📂 프로젝트 구조

```
capstone_nlp/
├── scripts/                           # 분석 스크립트
│   ├── 1_regression_original_framing.py
│   ├── 2_regression_hurtlex_framing.py
│   ├── 3_compare_two_models.py
│   ├── framing_detection_rules.py
│   ├── platform_gap_analysis.py
│   └── archive/                       # 오래된 스크립트
│
├── results/                           # 분석 결과
│   ├── p0_base/                      # 기본 탐색
│   ├── p2_analysis/                  # 타겟 & Speech Act
│   ├── p3_framing/                   # 프레이밍 분석
│   └── p4_regression/                # 회귀 모델
│
├── lexicons/                          # 사전 파일
│   └── hurtlex_EN.tsv                # Hurtlex 표준 사전
│
└── hatexplain_prediction.csv         # 원본 데이터
```

---

## 🔍 분석 단계

### P0: 기본 데이터 탐색
📁 `results/p0_base/`

- Agreement 분포 분석
- 텍스트 길이 분포
- Borderline 샘플 추출

**주요 파일**:
- `README.md` - 분석 요약
- `normal_agreement_distribution.png`
- `normal_length_distribution.png`

---

### P2: 타겟 & Speech Act 분석
📁 `results/p2_analysis/`

**목적**: 타겟 집단별 특성 및 Speech Act 패턴 분석

**주요 분석**:
- 타겟 집단별 Borderline 비율
- Speech Act(진술, 질문, 명령 등) 분포
- Hurtlex 없는 암묵적 혐오 케이스

**주요 파일**:
- `README.md` - 분석 요약
- `target_speechact_analysis.png` - 종합 시각화
- `target_profile.csv` - 타겟 통계

---

### P3: 프레이밍 분석
📁 `results/p3_framing/`

**목적**: 문헌 기반 11개 프레이밍 카테고리 적용

**프레이밍 카테고리**:
- DEHUMANIZATION, THREAT_VIOLENCE, CRIMINALIZATION
- CONSPIRACY, EXCLUSION, ECONOMIC
- GENERALIZATION, DISEASE_FILTH, SEXUALITY
- CULTURAL_DEVIANCE, PHYSICAL_WEAKNESS

**주요 파일**:
- `README.md` - 분석 요약
- `framing_analysis.png` - 프레이밍 분포
- `true_implicit_with_framing.csv` - 라벨링된 샘플

---

### P4: 회귀 분석 (Regression)
📁 `results/p4_regression/`

**목적**: Borderline vs 만장일치 분류 (플랫폼 편향 제거)

**핵심 발견**: 
- ⭐ **has_target(타겟 집단 언급)이 가장 강력한 예측 변수 (+3.1)**
- 기존 11개 vs Hurtlex 17개 프레이밍 → **성능 완전 동일** (F1 72.7%)
- has_target이 너무 강력해서 프레이밍 방식 차이가 무의미

**주요 파일**:
- `README.md` - 전체 분석 요약 및 가이드
- `1_original_model_visualization.png` - 기존 11개 프레이밍 결과
- `2_hurtlex_model_visualization.png` - Hurtlex 17개 결과
- `3_model_comparison.png` - 두 모델 비교 ⭐

**스크립트**:
- `scripts/1_regression_original_framing.py`
- `scripts/2_regression_hurtlex_framing.py`
- `scripts/3_compare_two_models.py`

---

## 🎯 핵심 인사이트

### 1. has_target > 프레이밍 방식
```
타겟 집단 명시 → Borderline 확률 급증 (계수 +3.0)
프레이밍 종류는 2차적 효과 (계수 0.1~0.7)
```

### 2. 두 프레이밍 모델의 차이점

**기존 11개 프레이밍** (문헌 기반)
- ✅ 전략적 의미 분석 (음모론, 배제, 비인간화)
- ✅ 해석 가능성 높음
- ❌ 낮은 커버리지

**Hurtlex 17개 카테고리** (사전 기반)
- ✅ 표준화된 분류 체계
- ✅ 높은 커버리지 (83.7%)
- ✅ 세분화된 혐오 유형 분석
- ❌ 표면적 단어에 의존

**결과**: has_target 덕분에 두 모델 성능 동일 (F1 72.7%)

### 3. 실전 권장사항

| 목적 | 추천 모델 | 이유 |
|------|----------|------|
| 성능 최대화 | 둘 다 동일 | has_target이 핵심 |
| 세부 유형 분석 | Hurtlex | 17개 카테고리 |
| 해석/논문 | 기존 11개 | 문헌 근거 명확 |

---

## 📊 주요 메트릭

### 모델 성능 (플랫폼 제외)

| 지표 | 기존 (11개) | Hurtlex (17개) | 차이 |
|------|------------|---------------|------|
| Accuracy | 76.1% | 76.1% | 0.0%p |
| Precision | 59.9% | 59.9% | 0.0%p |
| Recall | 92.4% | 92.4% | 0.0%p |
| F1-score | 72.7% | 72.7% | 0.0%p |

### Feature Importance Top 3

**기존 모델**:
1. has_target: +3.07 ⭐
2. framing_economic: +0.41
3. framing_dehumanization: +0.16

**Hurtlex 모델**:
1. has_target: +3.14 ⭐
2. ddp (Body Parts): +0.70
3. rci (Religious): -0.67

---

## 🔄 다음 단계

- [ ] **has_target 세부 분석**
  - 어떤 타겟 집단이 가장 논쟁적인지
  - 타겟별 Borderline 비율 차이

- [ ] **하이브리드 모델 실험**
  - Hurtlex 17 + 기존 6 = 23개 피처
  - has_target 효과 제어 후 순수 프레이밍 비교

- [ ] **타겟별 Hurtlex 분석**
  - 여성 타겟 → asf(여성 비하) 높음?
  - 인종 타겟 → asm(민족 비하) 높음?

---

## 📚 참고 문헌

- **HurtLex**: Bassignana et al. (2018)
- **Framing**: ElSherief et al. (2021), Ocampo et al. (2023)
- **Dataset**: Mathew et al. (2021) - HateXplain

---

## 👤 작성자

**Minseo** | 2026년 4월 | 캡스톤 디자인 프로젝트
