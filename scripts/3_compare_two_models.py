#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""간단한 모델 비교 시각화"""

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

print("모델 비교 그래프 생성 중...")

# 데이터 로드
old_perf = pd.read_csv('results/p1_analysis/normal_model_performance.csv')
old_metrics = dict(zip(old_perf['지표'], old_perf['값']))

hurtlex_perf = pd.read_csv('results/p1_analysis/hurtlex_model_performance.csv')
hurtlex_metrics = dict(zip(hurtlex_perf['Metric'], hurtlex_perf['Value']))

old_features = pd.read_csv('results/p1_analysis/normal_feature_importance.csv')
hurtlex_features = pd.read_csv('results/p1_analysis/hurtlex_feature_importance.csv')

# 비교 데이터 생성
comparison_data = []
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
    old_val = float(old_metrics.get(metric, '0'))
    hurtlex_val = float(hurtlex_metrics.get(metric, '0'))
    diff = hurtlex_val - old_val
    
    comparison_data.append({
        'Metric': metric,
        'Old': old_val,
        'Hurtlex': hurtlex_val,
        'Diff': diff
    })

comparison_df = pd.DataFrame(comparison_data)

# 시각화
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. 모델 성능 비교
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

# 값 표시
for i, (old_val, new_val) in enumerate(zip(comparison_df['Old'], comparison_df['Hurtlex'])):
    ax1.text(i - width/2, old_val + 0.01, f'{old_val:.3f}', ha='center', va='bottom', fontsize=9)
    ax1.text(i + width/2, new_val + 0.01, f'{new_val:.3f}', ha='center', va='bottom', fontsize=9)

# 2. 기존 모델 Top 10
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

# 3. Hurtlex 모델 Top 10
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

# 4. 공통 피처 비교 (has_target만 - token_length는 너무 작아서 제외)
ax4 = fig.add_subplot(gs[2, 0])
common_features = ['has_target']
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
    
    bars1 = ax4.bar(x_pos - width/2, common_df['Old'], width, label='기존', color='#3498db', alpha=0.7)
    bars2 = ax4.bar(x_pos + width/2, common_df['Hurtlex'], width, label='Hurtlex', color='#e74c3c', alpha=0.7)
    
    # 값 표시
    for i, (old_val, hurtlex_val) in enumerate(zip(common_df['Old'], common_df['Hurtlex'])):
        ax4.text(i - width/2, old_val + 0.1, f'{old_val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax4.text(i + width/2, hurtlex_val + 0.1, f'{hurtlex_val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(common_df['Feature'], fontsize=11)
    ax4.set_ylabel('Coefficient', fontsize=11, fontweight='bold')
    ax4.set_title('공통 피처 계수 비교 (has_target)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 3.5)  # has_target 값을 잘 보이게

# 5. 성능 차이 - 텍스트 중심으로 표시
ax5 = fig.add_subplot(gs[2, 1])
metrics_list = comparison_df['Metric'].tolist()
diffs = comparison_df['Diff'].tolist()

# Y축 범위를 넓게 설정
ax5.set_ylim(-0.01, 0.01)
ax5.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.8)

# 각 지표마다 동점 표시
for i, (metric, d) in enumerate(zip(metrics_list, diffs)):
    if abs(d) < 0.001:
        # 0 위치에 점 표시
        ax5.plot(i, 0, 'o', markersize=15, color='#2ecc71', alpha=0.7)
        # 동점 텍스트
        ax5.text(i, 0.003, '완전\n동점', ha='center', va='bottom', 
                fontsize=11, fontweight='bold', color='#27ae60')
        ax5.text(i, -0.003, f'{d:.4f}', ha='center', va='top', 
                fontsize=9, color='#555')
    else:
        y_pos = d
        color = '#e74c3c' if d < 0 else '#3498db'
        ax5.plot(i, y_pos, 'o', markersize=12, color=color, alpha=0.7)
        ax5.text(i, y_pos + (0.002 if d > 0 else -0.002), f'{d:+.4f}', 
                ha='center', va='bottom' if d > 0 else 'top', 
                fontsize=9, fontweight='bold')

ax5.set_xticks(range(len(metrics_list)))
ax5.set_xticklabels(metrics_list, fontsize=11, fontweight='bold')
ax5.set_ylabel('Difference (Hurtlex - 기존)', fontsize=11, fontweight='bold')
ax5.set_title('성능 차이 (양수 = Hurtlex 우세)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

plt.savefig('results/p1_analysis/model_comparison_old_vs_hurtlex.png', dpi=300, bbox_inches='tight')
print("✅ 저장 완료: results/p1_analysis/model_comparison_old_vs_hurtlex.png")
plt.close()
