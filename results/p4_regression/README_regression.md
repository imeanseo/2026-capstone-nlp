# P1 분석 결과 - 프레이밍 회귀 분석 (플랫폼 제외)

## 📌 분석 개요

- **날짜**: 2026년 4월 16일
- **목적**: 플랫폼 편향(platform_gab) 제거 후 순수 프레이밍 효과 측정
- **데이터**: Normal 7,814개 (Borderline 2,690개 + 만장일치 5,124개)
- **핵심 발견**: **has_target(타겟 집단 언급)이 압도적으로 가장 중요한 예측 변수**

---

## 📊 주요 결과 파일

### 1️⃣ 기존 프레이밍 모델 (11개 카테고리)

**시각화**
- `1_original_model_visualization.png` - 모델 성능, 피처 중요도, 혼동 행렬

**데이터**
- `1_original_model_performance.csv` - 성능 지표 (Accuracy 76.1%, F1 72.7%)
- `1_original_feature_importance.csv` - 피처 중요도 (has_target +3.07이 1위)
- `1_original_predictions.csv` - 전체 예측 결과

**Top 3 피처**
1. has_target: +3.07 (타겟 집단 언급)
2. framing_economic: +0.41 (경제적 부담 프레이밍)
3. framing_dehumanization: +0.16 (비인간화 프레이밍)

---

### 2️⃣ Hurtlex 프레이밍 모델 (17개 카테고리)

**시각화**
- `2_hurtlex_model_visualization.png` - 모델 성능, 피처 중요도, 혼동 행렬

**데이터**
- `2_hurtlex_model_performance.csv` - 성능 지표 (Accuracy 76.1%, F1 72.7%)
- `2_hurtlex_feature_importance.csv` - 피처 중요도 (has_target +3.14가 1위)
- `2_hurtlex_predictions.csv` - 전체 예측 결과
- `hurtlex_category_stats.csv` - Hurtlex 카테고리별 통계

**Top 3 피처**
1. has_target: +3.14 (타겟 집단 언급)
2. ddp (Body Parts): +0.70 (신체 부위 비하)
3. rci (Religious): -0.67 (종교 비하 → 만장일치)

---

### 3️⃣ 두 모델 비교

**시각화**
- `3_model_comparison.png` - 성능 비교, 피처 중요도 비교, 공통 피처 분석

**데이터**
- `model_performance_comparison.csv` - 성능 지표 비교
- `common_features_comparison.csv` - 공통 피처(has_target) 계수 비교

**핵심 결과**
- **모든 성능 지표가 완전히 동일** (Accuracy 76.1%, Precision 59.9%, Recall 92.4%, F1 72.7%)
- has_target이 너무 강력해서 프레이밍 방식(11개 vs 17개)의 차이가 무의미함

---

## 🎯 핵심 발견

### 1. has_target이 모든 것을 지배

```
has_target 계수: +3.0~3.1 (압도적 1위)
다른 피처들: 0.1~0.7 (미미한 수준)
```

**의미**: 타겟 집단(여성, 무슬림, 흑인 등)을 명시하는 것 자체가 Borderline 판단의 핵심 요인

### 2. 프레이밍 방식의 차이는 2차적

- 기존 11개: 경제적 부담, 비인간화 등 전략적 프레이밍
- Hurtlex 17개: 신체 비하, 여성 비하 등 표면적 단어 기반
- **결과적으로 성능 차이 없음** (has_target이 너무 강력함)

### 3. 두 모델의 장단점

**기존 모델 (11개)**
- ✅ 문헌 기반 프레이밍 (ElSherief, Ocampo, Carvalho)
- ✅ 전략적 의미 분석 가능 (음모론, 배제 등)
- ❌ 낮은 커버리지

**Hurtlex 모델 (17개)**
- ✅ 표준화된 분류 체계
- ✅ 높은 커버리지 (83.7%)
- ✅ 세분화된 혐오 유형 분석
- ❌ 표면적 단어에 의존

---

## 💡 시사점

### 어노테이터 판단 기준

```
타겟 명시 O → Borderline (의견 갈림)
타겟 명시 X → 만장일치 Normal
```

프레이밍 방식보다 **"누구를 타겟팅했는가"**가 더 중요

### 실전 권장사항

**시나리오 A: 성능만 중요**
→ 둘 다 동일, has_target이 핵심

**시나리오 B: 세부 분석 필요**
→ Hurtlex 추천 (17개 카테고리로 상세 분석)

**시나리오 C: 해석 가능성 중요**
→ 기존 프레이밍 추천 (전략적 의미 명확)

---

## 📁 기타 파일

- `framing_analysis.png` - 프레이밍 분포 및 분석 (이전 분석)
- `true_implicit_with_framing.csv` - True implicit 샘플 74개 (프레이밍 적용)

---

## 🔄 다음 단계

1. **has_target 세부 분석**
   - 어떤 타겟 집단이 가장 논쟁적인지
   - 타겟별 Borderline 비율 차이

2. **하이브리드 모델**
   - Hurtlex 17 + 기존 6 = 23개 피처
   - has_target 효과 제어 후 순수 프레이밍 비교

3. **타겟별 Hurtlex 분석**
   - 여성 타겟 → asf(여성 비하) 높음?
   - 인종 타겟 → asm(민족 비하) 높음?

---

**생성 스크립트**: 
- `scripts/1_regression_original_framing.py`
- `scripts/2_regression_hurtlex_framing.py`
- `scripts/3_compare_two_models.py`
