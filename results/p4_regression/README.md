# P4: 회귀 분석 (Regression)

## 📌 분석 개요

- **날짜**: 2026년 4월 16일
- **목적**: Borderline vs 만장일치 Normal 분류
- **데이터**: Normal 7,814개

---

## 🎯 주요 실험

### 실험 1: 임의 프레이밍 (11개 카테고리)
- 문헌 기반 프레이밍 lexicon 적용
- 플랫폼 변수 포함/제외 비교

### 실험 2: Hurtlex 사전 기반 (17개 카테고리)
- 표준 Hurtlex 8,228 단어
- 17개 표준 카테고리

### 실험 3: 두 모델 비교
- 기존 11개 vs Hurtlex 17개
- 플랫폼 편향 제거 후 재분석

---

## ⭐ 핵심 발견

### has_target이 압도적 1위 예측 변수

```
has_target: +3.0~3.1 (타겟 집단 언급)
다른 피처들: 0.1~0.7

→ 타겟 명시 여부가 Borderline 판단의 핵심!
```

### 두 모델 성능 완전 동일

| 지표 | 기존 (11개) | Hurtlex (17개) |
|------|------------|---------------|
| Accuracy | 76.1% | 76.1% |
| Precision | 59.9% | 59.9% |
| Recall | 92.4% | 92.4% |
| F1-score | 72.7% | 72.7% |

has_target이 너무 강력해서 프레이밍 방식 차이가 무의미해짐

---

## 📁 파일 목록

### 시각화 (3개)
- `1_original_model_visualization.png` - 기존 11개 프레이밍
- `2_hurtlex_model_visualization.png` - Hurtlex 17개
- `3_model_comparison.png` - 두 모델 비교 ⭐

### 성능 데이터 (3개)
- `1_original_model_performance.csv`
- `2_hurtlex_model_performance.csv`
- `model_performance_comparison.csv`

### Feature Importance (3개)
- `1_original_feature_importance.csv`
- `2_hurtlex_feature_importance.csv`
- `common_features_comparison.csv`

### 예측 결과 (3개)
- `1_original_predictions.csv`
- `2_hurtlex_predictions.csv`
- `hurtlex_category_stats.csv`

### 문서 (1개)
- `README_regression.md` - 상세 분석 가이드

---

## 💡 결론

1. **has_target > 프레이밍 종류**
   - 타겟 집단 명시 여부가 가장 중요

2. **두 프레이밍 방식 모두 유효**
   - 기존 11개: 전략적 의미 분석
   - Hurtlex 17개: 표준화 + 높은 커버리지

3. **실전 권장**
   - 성능 우선 → 둘 다 동일
   - 세부 분석 → Hurtlex
   - 해석성 → 기존 11개

---

## 🔄 연결

← **P3**: 프레이밍 분석
→ **다음 단계**: has_target 세부 분석, 하이브리드 모델
