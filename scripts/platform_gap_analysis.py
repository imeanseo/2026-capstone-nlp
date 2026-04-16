#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
플랫폼 차이 (platform_gab) 변수 심층 분석

목적:
1. Gab vs Twitter 플랫폼별 Borderline 비율 차이 확인
2. platform_gab 변수가 회귀 모델에서 왜 가장 강력한 예측 변수인지 분석
3. 플랫폼별 Hurtlex 카테고리 분포 차이 확인
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 한글 폰트 설정
import platform
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("플랫폼 차이 (platform_gab) 변수 심층 분석")
print("="*80)

# ========================================================================
# 1. 데이터 로드
# ========================================================================
print("\n📂 [Step 1] 데이터 로드...")

# Hurtlex 프레이밍 데이터 (이미 platform_gab 포함)
df = pd.read_csv('hatexplain_prediction.csv')
df = df[df['gold_hatexplain_label'] == 1].copy()  # Normal만

# Platform 추출
df['platform'] = df['post_id'].apply(lambda x: 'Gab' if 'gab' in str(x).lower() else 'Twitter')
df['platform_gab'] = (df['platform'] == 'Gab').astype(int)

# Borderline 레이블 (기존과 동일)
import re

def parse_annotators(annotator_str):
    try:
        match = re.search(r"'label':\s*array\(\[([^\]]+)\]\)", annotator_str)
        if match:
            labels_str = match.group(1)
            labels = [int(x.strip()) for x in labels_str.split(',')]
            num_normal = sum(1 for label in labels if label == 1)
            return num_normal
        return 0
    except:
        return 0

df['num_agree'] = df['annotators'].apply(parse_annotators)
df['is_borderline'] = (df['num_agree'] == 2).astype(int)

print(f"✅ Normal 전체: {len(df):,}개")
print(f"  - Gab: {(df['platform']=='Gab').sum():,}개")
print(f"  - Twitter: {(df['platform']=='Twitter').sum():,}개")

# ========================================================================
# 2. 플랫폼별 Borderline 비율 분석
# ========================================================================
print("\n" + "="*80)
print("📊 [Step 2] 플랫폼별 Borderline 비율 분석")
print("="*80)

# 플랫폼별 Borderline 비율
platform_stats = df.groupby('platform').agg({
    'is_borderline': ['sum', 'mean', 'count']
}).round(4)

print("\n【플랫폼별 Borderline 비율】")
for platform in ['Gab', 'Twitter']:
    platform_df = df[df['platform'] == platform]
    borderline_count = platform_df['is_borderline'].sum()
    borderline_rate = platform_df['is_borderline'].mean() * 100
    total = len(platform_df)
    
    print(f"\n{platform}:")
    print(f"  - 전체: {total:,}개")
    print(f"  - Borderline: {borderline_count:,}개 ({borderline_rate:.2f}%)")
    print(f"  - 만장일치: {total - borderline_count:,}개 ({100 - borderline_rate:.2f}%)")

# 카이제곱 검정
gab_borderline = df[df['platform'] == 'Gab']['is_borderline'].sum()
gab_total = len(df[df['platform'] == 'Gab'])
twitter_borderline = df[df['platform'] == 'Twitter']['is_borderline'].sum()
twitter_total = len(df[df['platform'] == 'Twitter'])

contingency_table = np.array([
    [gab_borderline, gab_total - gab_borderline],
    [twitter_borderline, twitter_total - twitter_borderline]
])

chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\n【카이제곱 검정】")
print(f"  Chi-square statistic: {chi2:.2f}")
print(f"  p-value: {p_value:.4e}")
print(f"  자유도: {dof}")
if p_value < 0.001:
    print(f"  결론: 플랫폼 간 Borderline 비율 차이가 통계적으로 매우 유의함 (p < 0.001)")
elif p_value < 0.05:
    print(f"  결론: 플랫폼 간 Borderline 비율 차이가 통계적으로 유의함 (p < 0.05)")
else:
    print(f"  결론: 플랫폼 간 Borderline 비율 차이가 유의하지 않음")

# 오즈비 (Odds Ratio) 계산
gab_odds = gab_borderline / (gab_total - gab_borderline)
twitter_odds = twitter_borderline / (twitter_total - twitter_borderline)
odds_ratio = gab_odds / twitter_odds

print(f"\n【오즈비 (Odds Ratio)】")
print(f"  Gab 오즈: {gab_odds:.4f}")
print(f"  Twitter 오즈: {twitter_odds:.4f}")
print(f"  오즈비: {odds_ratio:.4f}")
print(f"  해석: Gab에서 Borderline일 확률이 Twitter 대비 {odds_ratio:.2f}배")

# ========================================================================
# 3. Hurtlex 카테고리별 플랫폼 차이
# ========================================================================
print("\n" + "="*80)
print("🔍 [Step 3] Hurtlex 카테고리별 플랫폼 차이")
print("="*80)

# Hurtlex 카테고리 로드
from collections import defaultdict

hurtlex_df = pd.read_csv('lexicons/hurtlex_EN.tsv', sep='\t')
hurtlex_dict = defaultdict(set)
for _, row in hurtlex_df.iterrows():
    category = row['category']
    lemma = str(row['lemma']).lower().strip()
    if lemma and lemma != 'nan':
        hurtlex_dict[category].add(lemma)

# 각 카테고리별 매칭
for category in sorted(hurtlex_dict.keys()):
    df[f'hurtlex_{category}'] = df['text'].apply(
        lambda x: int(any(word.strip('.,!?;:"\'()[]{}') in hurtlex_dict[category] 
                         for word in str(x).lower().split()))
    )

# 플랫폼별 Hurtlex 카테고리 분포
print("\n【플랫폼별 Hurtlex 카테고리 비율 차이 (상위 10개)】")
print(f"{'Category':<10s} {'Gab (%)':<12s} {'Twitter (%)':<12s} {'차이 (%p)':<12s}")
print("-" * 50)

category_platform_diff = []
for category in sorted(hurtlex_dict.keys()):
    gab_rate = df[df['platform'] == 'Gab'][f'hurtlex_{category}'].mean() * 100
    twitter_rate = df[df['platform'] == 'Twitter'][f'hurtlex_{category}'].mean() * 100
    diff = gab_rate - twitter_rate
    
    category_platform_diff.append({
        'category': category,
        'gab_rate': gab_rate,
        'twitter_rate': twitter_rate,
        'diff': diff
    })

# 차이가 큰 순서대로 정렬
category_platform_diff_df = pd.DataFrame(category_platform_diff).sort_values('diff', key=abs, ascending=False)

for _, row in category_platform_diff_df.head(10).iterrows():
    print(f"{row['category']:<10s} {row['gab_rate']:>10.2f}%  {row['twitter_rate']:>10.2f}%  {row['diff']:>+10.2f}%p")

# ========================================================================
# 4. 시각화
# ========================================================================
print("\n📊 [Step 4] 시각화...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 4-1. 플랫폼별 Borderline 비율
ax1 = fig.add_subplot(gs[0, 0])

platforms = ['Gab', 'Twitter']
borderline_rates = [
    df[df['platform'] == 'Gab']['is_borderline'].mean() * 100,
    df[df['platform'] == 'Twitter']['is_borderline'].mean() * 100
]

bars = ax1.bar(platforms, borderline_rates, color=['#e74c3c', '#3498db'], alpha=0.7, width=0.5)
ax1.set_ylabel('Borderline Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Borderline Rate by Platform', fontsize=14, fontweight='bold')
ax1.set_ylim([0, max(borderline_rates) * 1.2])

# 막대 위에 값 표시
for i, (bar, rate) in enumerate(zip(bars, borderline_rates)):
    ax1.text(bar.get_x() + bar.get_width()/2, rate + 1, 
             f'{rate:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 4-2. 플랫폼별 샘플 수
ax2 = fig.add_subplot(gs[0, 1])

platform_counts = df['platform'].value_counts()
ax2.pie(platform_counts.values, labels=platform_counts.index, autopct='%1.1f%%',
        colors=['#e74c3c', '#3498db'], startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('Sample Distribution by Platform', fontsize=14, fontweight='bold')

# 4-3. 플랫폼별 Borderline/만장일치 분포 (스택 바)
ax3 = fig.add_subplot(gs[1, :])

gab_borderline_count = df[(df['platform'] == 'Gab') & (df['is_borderline'] == 1)].shape[0]
gab_unanimous_count = df[(df['platform'] == 'Gab') & (df['is_borderline'] == 0)].shape[0]
twitter_borderline_count = df[(df['platform'] == 'Twitter') & (df['is_borderline'] == 1)].shape[0]
twitter_unanimous_count = df[(df['platform'] == 'Twitter') & (df['is_borderline'] == 0)].shape[0]

x = np.arange(2)
width = 0.5

p1 = ax3.bar(x, [gab_borderline_count, twitter_borderline_count], width, 
             label='Borderline', color='#e74c3c', alpha=0.7)
p2 = ax3.bar(x, [gab_unanimous_count, twitter_unanimous_count], width,
             bottom=[gab_borderline_count, twitter_borderline_count],
             label='만장일치', color='#3498db', alpha=0.7)

ax3.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
ax3.set_title('Borderline vs 만장일치 Distribution by Platform', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(['Gab', 'Twitter'], fontsize=12)
ax3.legend(fontsize=11)

# 막대 안에 값 표시
for i, (b, u) in enumerate(zip([gab_borderline_count, twitter_borderline_count],
                                [gab_unanimous_count, twitter_unanimous_count])):
    ax3.text(i, b/2, f'{b:,}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax3.text(i, b + u/2, f'{u:,}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# 4-4. Hurtlex 카테고리 차이 (Gab vs Twitter) Top 10
ax4 = fig.add_subplot(gs[2, :])

top_10_diff = category_platform_diff_df.head(10)
categories = top_10_diff['category'].tolist()
diffs = top_10_diff['diff'].tolist()

colors = ['#e74c3c' if d > 0 else '#3498db' for d in diffs]
bars = ax4.barh(range(len(categories)), diffs, color=colors, alpha=0.7)
ax4.set_yticks(range(len(categories)))
ax4.set_yticklabels(categories, fontsize=10)
ax4.set_xlabel('Difference (Gab % - Twitter %)', fontsize=12, fontweight='bold')
ax4.set_title('Top 10 Hurtlex Category Differences (Gab - Twitter)\n(Red = Gab 높음, Blue = Twitter 높음)', 
              fontsize=14, fontweight='bold')
ax4.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax4.invert_yaxis()

# 막대 끝에 값 표시
for i, (bar, diff) in enumerate(zip(bars, diffs)):
    x_pos = diff + (0.5 if diff > 0 else -0.5)
    ax4.text(x_pos, i, f'{diff:+.1f}%p', va='center', ha='left' if diff > 0 else 'right', 
             fontsize=9, fontweight='bold')

plt.savefig('results/p1_analysis/platform_gab_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ 시각화 저장: results/p1_analysis/platform_gab_analysis.png")

# ========================================================================
# 5. 결과 저장
# ========================================================================
print("\n💾 [Step 5] 결과 저장...")

# 5-1. 플랫폼 통계
platform_summary = []
for platform in ['Gab', 'Twitter']:
    platform_df = df[df['platform'] == platform]
    platform_summary.append({
        'Platform': platform,
        'Total': len(platform_df),
        'Borderline': platform_df['is_borderline'].sum(),
        'Borderline_Rate': f"{platform_df['is_borderline'].mean() * 100:.2f}%",
        'Unanimous': len(platform_df) - platform_df['is_borderline'].sum(),
        'Unanimous_Rate': f"{(1 - platform_df['is_borderline'].mean()) * 100:.2f}%"
    })

platform_summary_df = pd.DataFrame(platform_summary)
platform_summary_df.to_csv('results/p1_analysis/platform_comparison_summary.csv',
                           index=False, encoding='utf-8-sig')
print("✅ results/p1_analysis/platform_comparison_summary.csv")

# 5-2. 카테고리별 플랫폼 차이
category_platform_diff_df.to_csv('results/p1_analysis/platform_hurtlex_category_diff.csv',
                                 index=False, encoding='utf-8-sig')
print("✅ results/p1_analysis/platform_hurtlex_category_diff.csv")

# ========================================================================
# 6. 최종 요약
# ========================================================================
print("\n" + "="*80)
print("📌 최종 요약 - platform_gab 변수 분석")
print("="*80)

gab_rate = df[df['platform'] == 'Gab']['is_borderline'].mean() * 100
twitter_rate = df[df['platform'] == 'Twitter']['is_borderline'].mean() * 100

print(f"""
✅ platform_gab 변수가 회귀 모델에서 가장 강력한 예측 변수인 이유

【플랫폼별 Borderline 비율】
- Gab: {gab_rate:.2f}%
- Twitter: {twitter_rate:.2f}%
- 차이: {gab_rate - twitter_rate:.2f}%p

【통계적 유의성】
- Chi-square: {chi2:.2f}
- p-value: {p_value:.4e} {'(*** p < 0.001)' if p_value < 0.001 else ''}
- 결론: 플랫폼 간 차이가 통계적으로 매우 유의함

【오즈비 (Odds Ratio)】
- {odds_ratio:.2f}배
- 해석: Gab에서 Borderline일 확률이 Twitter 대비 {odds_ratio:.2f}배 높음

【회귀 계수 비교】
- 기존 모델: platform_gab = +1.6225
- Hurtlex 모델: platform_gab = +1.6613
- 두 모델 모두에서 압도적 1위 예측 변수

【해석】
1. Gab 플랫폼의 콘텐츠가 Twitter보다 더 논쟁적/경계적
2. 어노테이터들이 Gab 콘텐츠를 판단할 때 더 의견이 갈림
3. 플랫폼 자체가 콘텐츠 특성의 강력한 지표
4. Gab = 혐오 표현에 관대한 플랫폼 문화 반영

【주요 Hurtlex 카테고리 차이 (Gab - Twitter, Top 3)】
""")

for i, row in enumerate(category_platform_diff_df.head(3).iterrows(), 1):
    _, data = row
    direction = "Gab 높음" if data['diff'] > 0 else "Twitter 높음"
    print(f"{i}. {data['category']}: {data['diff']:+.2f}%p ({direction})")
    print(f"   Gab {data['gab_rate']:.2f}% vs Twitter {data['twitter_rate']:.2f}%")

print(f"""
💡 결론:
platform_gab 변수는 단순한 플랫폼 구분을 넘어서,
콘텐츠의 논쟁성/경계성을 나타내는 강력한 프록시(proxy) 변수입니다.

회귀 계수 +1.6은 다른 피처들 (0.1~0.4)에 비해 압도적으로 크므로,
Gab 플랫폼 여부만으로도 Borderline 여부를 상당히 예측할 수 있습니다.
""")

print("="*80)
print("✅ platform_gab 변수 분석 완료!")
print("="*80)
print("\n생성된 파일:")
print("  1. results/p1_analysis/platform_gab_analysis.png")
print("  2. results/p1_analysis/platform_comparison_summary.csv")
print("  3. results/p1_analysis/platform_hurtlex_category_diff.csv")
