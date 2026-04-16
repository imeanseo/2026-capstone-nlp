#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 프레이밍 vs Hurtlex 카테고리 회귀 모델 비교

목적:
1. 두 모델의 성능 지표 비교
2. Feature Importance 비교
3. 어떤 접근법이 더 효과적인지 분석
4. 앙상블 가능성 검토
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
import platform
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("기존 프레이밍 vs Hurtlex 카테고리 회귀 모델 비교")
print("="*80)

# ========================================================================
# 1. 모델 성능 비교
# ========================================================================
print("\n📊 [Step 1] 모델 성능 비교...")

# 기존 모델 성능
old_perf = pd.read_csv('results/p1_analysis/normal_model_performance.csv')
old_metrics = dict(zip(old_perf['지표'], old_perf['값']))

# Hurtlex 모델 성능
hurtlex_perf = pd.read_csv('results/p1_analysis/hurtlex_model_performance.csv')
hurtlex_metrics = dict(zip(hurtlex_perf['Metric'], hurtlex_perf['Value']))

print("\n【모델 성능 비교표】")
print(f"{'Metric':<15s} {'기존 (11개 프레이밍)':<25s} {'Hurtlex (17개 카테고리)':<25s} {'차이':<15s} {'승자':<10s}")
print("-" * 95)

comparison_data = []
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
    old_val = float(old_metrics.get(metric, '0'))
    hurtlex_val = float(hurtlex_metrics.get(metric, '0'))
    diff = hurtlex_val - old_val
    winner = 'Hurtlex' if diff > 0 else ('기존' if diff < 0 else '동점')
    
    comparison_data.append({
        'Metric': metric,
        'Old': old_val,
        'Hurtlex': hurtlex_val,
        'Diff': diff,
        'Winner': winner
    })
    
    print(f"{metric:<15s} {old_val:<25.3f} {hurtlex_val:<25.3f} {diff:+.3f}          {winner:<10s}")

# AUC (Hurtlex만 있음)
hurtlex_auc = float(hurtlex_metrics.get('AUC', '0'))
print(f"{'AUC':<15s} {'N/A':<25s} {hurtlex_auc:<25.3f} {'NEW':<15s} {'Hurtlex':<10s}")

comparison_df = pd.DataFrame(comparison_data)

# ========================================================================
# 2. Feature Importance 비교
# ========================================================================
print("\n📈 [Step 2] Feature Importance 비교...")

# 기존 모델
old_features = pd.read_csv('results/p1_analysis/normal_feature_importance.csv')
old_features['Model'] = 'Old'

# Hurtlex 모델
hurtlex_features = pd.read_csv('results/p1_analysis/hurtlex_feature_importance.csv')
hurtlex_features['Model'] = 'Hurtlex'

print("\n【기존 모델 - Top 5 피처】")
for i, row in old_features.head(5).iterrows():
    print(f"  {row['Feature']:30s}: {row['Coefficient']:+.4f}")

print("\n【Hurtlex 모델 - Top 5 피처】")
for i, row in hurtlex_features.head(5).iterrows():
    feat_name = row['Feature'].replace('hurtlex_', '')
    print(f"  {feat_name:30s}: {row['Coefficient']:+.4f}")

# ========================================================================
# 3. 공통 피처 분석
# ========================================================================
print("\n🔍 [Step 3] 공통 피처 분석...")

# 공통 피처: platform_gab, token_length, has_target
common_features = ['platform_gab', 'token_length', 'has_target']

print("\n【공통 피처 계수 비교】")
print(f"{'Feature':<20s} {'기존 모델':<15s} {'Hurtlex 모델':<15s} {'차이':<15s}")
print("-" * 65)

for feat in common_features:
    old_coef = old_features[old_features['Feature'] == feat]['Coefficient'].values
    hurtlex_coef = hurtlex_features[hurtlex_features['Feature'] == feat]['Coefficient'].values
    
    if len(old_coef) > 0 and len(hurtlex_coef) > 0:
        old_c = old_coef[0]
        hurtlex_c = hurtlex_coef[0]
        diff = hurtlex_c - old_c
        print(f"{feat:<20s} {old_c:+.4f}         {hurtlex_c:+.4f}         {diff:+.4f}")
    elif len(old_coef) > 0:
        print(f"{feat:<20s} {old_coef[0]:+.4f}         {'N/A':<15s}")
    elif len(hurtlex_coef) > 0:
        print(f"{feat:<20s} {'N/A':<15s} {hurtlex_coef[0]:+.4f}")

# ========================================================================
# 4. 시각화
# ========================================================================
print("\n📊 [Step 4] 시각화...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 4-1. 모델 성능 비교 (막대 그래프)
ax1 = fig.add_subplot(gs[0, :])
metrics = comparison_df['Metric'].tolist()
x = np.arange(len(metrics))
width = 0.35

ax1.bar(x - width/2, comparison_df['Old'], width, label='기존 (11개 프레이밍)', 
        color='#3498db', alpha=0.7)
ax1.bar(x + width/2, comparison_df['Hurtlex'], width, label='Hurtlex (17개 카테고리)', 
        color='#e74c3c', alpha=0.7)

ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('모델 성능 비교 (기존 vs Hurtlex)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=11)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')

# 각 막대 위에 값 표시
for i, (old_val, new_val) in enumerate(zip(comparison_df['Old'], comparison_df['Hurtlex'])):
    ax1.text(i - width/2, old_val + 0.01, f'{old_val:.3f}', ha='center', va='bottom', fontsize=9)
    ax1.text(i + width/2, new_val + 0.01, f'{new_val:.3f}', ha='center', va='bottom', fontsize=9)

# 4-2. 기존 모델 Top 10 피처
ax2 = fig.add_subplot(gs[1, 0])
top_10_old = old_features.head(10).copy()
colors_old = ['#e74c3c' if c > 0 else '#3498db' for c in top_10_old['Coefficient']]

ax2.barh(range(len(top_10_old)), top_10_old['Coefficient'], color=colors_old, alpha=0.7)
ax2.set_yticks(range(len(top_10_old)))
ax2.set_yticklabels([f[:20] for f in top_10_old['Feature']], fontsize=9)
ax2.set_xlabel('Coefficient', fontsize=11, fontweight='bold')
ax2.set_title('기존 모델 - Top 10 Features', fontsize=12, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax2.invert_yaxis()

# 4-3. Hurtlex 모델 Top 10 피처
ax3 = fig.add_subplot(gs[1, 1])
top_10_hurtlex = hurtlex_features.head(10).copy()
top_10_hurtlex['Feature_Short'] = top_10_hurtlex['Feature'].apply(
    lambda x: x.replace('hurtlex_', '')[:15]
)
colors_hurtlex = ['#e74c3c' if c > 0 else '#3498db' for c in top_10_hurtlex['Coefficient']]

ax3.barh(range(len(top_10_hurtlex)), top_10_hurtlex['Coefficient'], color=colors_hurtlex, alpha=0.7)
ax3.set_yticks(range(len(top_10_hurtlex)))
ax3.set_yticklabels(top_10_hurtlex['Feature_Short'], fontsize=9)
ax3.set_xlabel('Coefficient', fontsize=11, fontweight='bold')
ax3.set_title('Hurtlex 모델 - Top 10 Features', fontsize=12, fontweight='bold')
ax3.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax3.invert_yaxis()

# 4-4. 공통 피처 계수 비교
ax4 = fig.add_subplot(gs[2, 0])
common_comparison = []
for feat in common_features:
    old_coef = old_features[old_features['Feature'] == feat]['Coefficient'].values
    hurtlex_coef = hurtlex_features[hurtlex_features['Feature'] == feat]['Coefficient'].values
    
    if len(old_coef) > 0 and len(hurtlex_coef) > 0:
        common_comparison.append({
            'Feature': feat,
            'Old': old_coef[0],
            'Hurtlex': hurtlex_coef[0]
        })

if common_comparison:
    common_df = pd.DataFrame(common_comparison)
    x_pos = np.arange(len(common_df))
    
    ax4.bar(x_pos - width/2, common_df['Old'], width, label='기존', color='#3498db', alpha=0.7)
    ax4.bar(x_pos + width/2, common_df['Hurtlex'], width, label='Hurtlex', color='#e74c3c', alpha=0.7)
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(common_df['Feature'], fontsize=10, rotation=15)
    ax4.set_ylabel('Coefficient', fontsize=11, fontweight='bold')
    ax4.set_title('공통 피처 계수 비교', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.grid(True, alpha=0.3, axis='y')

# 4-5. 성능 차이 (델타)
ax5 = fig.add_subplot(gs[2, 1])
metrics_list = comparison_df['Metric'].tolist()
diffs = comparison_df['Diff'].tolist()
colors_diff = ['#2ecc71' if d > 0 else '#e74c3c' for d in diffs]

ax5.bar(range(len(metrics_list)), diffs, color=colors_diff, alpha=0.7)
ax5.set_xticks(range(len(metrics_list)))
ax5.set_xticklabels(metrics_list, fontsize=10)
ax5.set_ylabel('Difference (Hurtlex - 기존)', fontsize=11, fontweight='bold')
ax5.set_title('성능 차이 (양수 = Hurtlex 우세)', fontsize=12, fontweight='bold')
ax5.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax5.grid(True, alpha=0.3, axis='y')

# 각 막대 위에 값 표시
for i, d in enumerate(diffs):
    y_pos = d + (0.005 if d > 0 else -0.005)
    va = 'bottom' if d > 0 else 'top'
    ax5.text(i, y_pos, f'{d:+.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

plt.savefig('results/p1_analysis/model_comparison_old_vs_hurtlex.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ 시각화 저장: results/p1_analysis/model_comparison_old_vs_hurtlex.png")

# ========================================================================
# 5. 비교 분석 결과 저장
# ========================================================================
print("\n💾 [Step 5] 비교 분석 결과 저장...")

# 5-1. 성능 비교표
comparison_df.to_csv('results/p1_analysis/model_performance_comparison.csv',
                     index=False, encoding='utf-8-sig')
print("✅ results/p1_analysis/model_performance_comparison.csv")

# 5-2. 공통 피처 비교표
if common_comparison:
    common_df.to_csv('results/p1_analysis/common_features_comparison.csv',
                     index=False, encoding='utf-8-sig')
    print("✅ results/p1_analysis/common_features_comparison.csv")

# ========================================================================
# 6. 최종 요약 및 권장사항
# ========================================================================
print("\n" + "="*80)
print("📌 최종 요약 - 기존 vs Hurtlex 모델 비교")
print("="*80)

# 승자 집계
winners = comparison_df['Winner'].value_counts()
print(f"""
【전체 성과 비교】
- 기존 모델 우세: {winners.get('기존', 0)}개 지표
- Hurtlex 모델 우세: {winners.get('Hurtlex', 0)}개 지표
- 동점: {winners.get('동점', 0)}개 지표

【성능 상세】
""")

for _, row in comparison_df.iterrows():
    print(f"• {row['Metric']:10s}: 기존 {row['Old']:.3f} → Hurtlex {row['Hurtlex']:.3f} (차이: {row['Diff']:+.3f}) [{row['Winner']}]")

print(f"""
【주요 발견】

1. 정확도 (Accuracy):
   - 기존: {comparison_df[comparison_df['Metric']=='Accuracy']['Old'].values[0]:.3f}
   - Hurtlex: {comparison_df[comparison_df['Metric']=='Accuracy']['Hurtlex'].values[0]:.3f}
   - 해석: {'Hurtlex가 더 정확함' if comparison_df[comparison_df['Metric']=='Accuracy']['Diff'].values[0] > 0 else '기존 모델이 더 정확함'}

2. Precision (정밀도):
   - 기존: {comparison_df[comparison_df['Metric']=='Precision']['Old'].values[0]:.3f}
   - Hurtlex: {comparison_df[comparison_df['Metric']=='Precision']['Hurtlex'].values[0]:.3f}
   - 해석: {'Hurtlex가 False Positive를 더 잘 제어' if comparison_df[comparison_df['Metric']=='Precision']['Diff'].values[0] > 0 else '기존 모델이 False Positive를 더 잘 제어'}

3. Recall (재현율):
   - 기존: {comparison_df[comparison_df['Metric']=='Recall']['Old'].values[0]:.3f}
   - Hurtlex: {comparison_df[comparison_df['Metric']=='Recall']['Hurtlex'].values[0]:.3f}
   - 해석: {'Hurtlex가 Borderline을 더 많이 찾아냄' if comparison_df[comparison_df['Metric']=='Recall']['Diff'].values[0] > 0 else '기존 모델이 Borderline을 더 많이 찾아냄'}

4. F1-score (균형 지표):
   - 기존: {comparison_df[comparison_df['Metric']=='F1-score']['Old'].values[0]:.3f}
   - Hurtlex: {comparison_df[comparison_df['Metric']=='F1-score']['Hurtlex'].values[0]:.3f}
   - 해석: {'Hurtlex가 precision-recall 균형이 더 좋음' if comparison_df[comparison_df['Metric']=='F1-score']['Diff'].values[0] > 0 else '기존 모델이 precision-recall 균형이 더 좋음'}

【모델별 강점】

✓ 기존 모델 (11개 프레이밍):
  - Precision 우수: False Positive 낮음
  - 프레이밍 전략 분석에 유용
  - 문헌 근거 명확
  - 해석 용이 (음모론, 배제 등)

✓ Hurtlex 모델 (17개 카테고리):
  - Recall 우수: Borderline 더 많이 포착
  - 높은 커버리지 (83.7%)
  - 표준화된 분류
  - 세분화된 혐오 유형 분석

【권장사항】

1. 목적별 모델 선택:
   - Implicit Hate 발굴 우선 → Hurtlex (높은 Recall)
   - False Positive 최소화 우선 → 기존 (높은 Precision)

2. 앙상블 접근:
   - 두 모델의 예측을 결합 (투표 또는 확률 평균)
   - 각 모델의 강점 활용

3. 하이브리드 피처:
   - Hurtlex 17개 카테고리 + 기존 6개 프레이밍
   - 총 23개 피처로 재학습
   - Surface-level (Hurtlex) + Strategic (기존) 동시 포착

4. 다음 단계:
   - 앙상블 모델 구현 및 평가
   - 하이브리드 피처 모델 구현
   - 두 모델의 예측 불일치 케이스 심층 분석
""")

print("="*80)
print("✅ 모델 비교 분석 완료!")
print("="*80)
print("\n생성된 파일:")
print("  1. results/p1_analysis/model_comparison_old_vs_hurtlex.png")
print("  2. results/p1_analysis/model_performance_comparison.csv")
print("  3. results/p1_analysis/common_features_comparison.csv")
