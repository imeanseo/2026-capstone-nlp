#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HateXplain Normal 라벨 분석 (P0 필수 항목)
담당: Minseo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HateXplain Normal 라벨 분석 시작")
print("="*80)

# ============================================================
# 0. 설정
# ============================================================
# 한글 폰트
try:
    font_list = [f.name for f in fm.fontManager.ttflist]
    korean_fonts = ['AppleSDGothicNeo', 'AppleGothic', 'NanumGothic', 'Malgun Gothic']
    for font in korean_fonts:
        if any(font in f for f in font_list):
            plt.rcParams['font.family'] = font
            print(f"✅ 한글 폰트: {font}")
            break
except:
    print("⚠️ 한글 폰트 설정 실패")

plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1단계] 데이터 로드 중...")
df = pd.read_csv('hatexplain_prediction.csv', encoding='utf-8')
print(f"✅ 전체 데이터: {len(df):,}개")
print(f"\n라벨 분포:")
print(df['gold_hatexplain_label'].value_counts().sort_index())

# Normal (label=1) 필터링
df_normal = df[df['gold_hatexplain_label'] == 1].copy().reset_index(drop=True)
print(f"\n✅ Normal 샘플: {len(df_normal):,}개 ({len(df_normal)/len(df)*100:.2f}%)")

# ============================================================
# 2. 문장 길이 분석
# ============================================================
print("\n[2단계] 문장 길이 분석 중...")

def parse_tokens_length(token_str):
    """post_tokens에서 길이만 추출"""
    try:
        if pd.isna(token_str) or token_str == '':
            return 0
        
        # numpy array 문자열 형식: ['word1' 'word2' 'word3']
        # 정규표현식으로 'word' 패턴 개수 세기
        import re
        matches = re.findall(r"'([^']+)'", str(token_str))
        if matches:
            return len(matches)
        
        # 백업: eval 시도
        tokens = eval(token_str)
        if isinstance(tokens, (list, np.ndarray)):
            return len(tokens)
        return 0
    except:
        return 0

df_normal['token_length'] = df_normal['post_tokens'].apply(parse_tokens_length)

# 0 제거
zero_count = (df_normal['token_length'] == 0).sum()
if zero_count > 0:
    print(f"⚠️ 길이 0인 샘플 {zero_count}개 제거")
    df_normal = df_normal[df_normal['token_length'] > 0].copy().reset_index(drop=True)

# 통계
stats = df_normal['token_length'].describe()
print(f"\n📊 문장 길이 통계:")
print(f"  평균: {stats['mean']:.2f} 토큰")
print(f"  중앙값: {stats['50%']:.2f} 토큰")
print(f"  표준편차: {stats['std']:.2f}")
print(f"  범위: {int(stats['min'])} ~ {int(stats['max'])} 토큰")
print(f"  Q1-Q3: {int(stats['25%'])} ~ {int(stats['75%'])} 토큰")

# 히스토그램
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].hist(df_normal['token_length'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.1f}')
axes[0].axvline(stats['50%'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["50%"]:.1f}')
axes[0].set_xlabel('Token Count', fontsize=12)
axes[0].set_ylabel('Sample Count', fontsize=12)
axes[0].set_title('Token Length Distribution (Normal)', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 구간별
bins = [0, 10, 20, 30, 40, 50, 100, 500]
labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-100', '100+']
df_normal['length_bin'] = pd.cut(df_normal['token_length'], bins=bins, labels=labels, include_lowest=True)
bin_counts = df_normal['length_bin'].value_counts().sort_index()

axes[1].bar(range(len(bin_counts)), bin_counts.values, color='teal', alpha=0.7, edgecolor='black')
axes[1].set_xticks(range(len(bin_counts)))
axes[1].set_xticklabels(bin_counts.index, rotation=0)
axes[1].set_xlabel('Token Range', fontsize=12)
axes[1].set_ylabel('Sample Count', fontsize=12)
axes[1].set_title('Distribution by Range', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

for i, v in enumerate(bin_counts.values):
    axes[1].text(i, v + max(bin_counts.values)*0.02, f'{v:,}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('normal_length_distribution.png', dpi=300, bbox_inches='tight')
print("✅ 그래프 저장: normal_length_distribution.png")

# ============================================================
# 3. 타깃 집단 분석
# ============================================================
print("\n[3단계] 타깃 집단 분석 중...")

def parse_targets(target_str):
    """targets 파싱"""
    try:
        if pd.isna(target_str) or target_str == '' or target_str == '[]':
            return []
        
        # ['Hispanic', 'Refugee'] 형식
        # 정규표현식으로 'word' 패턴 추출
        import re
        matches = re.findall(r"'([^']+)'", str(target_str))
        if matches:
            # 'None' 문자열 제외
            return [m for m in matches if m != 'None']
        
        # 백업: eval 시도
        targets = eval(target_str)
        if isinstance(targets, list):
            return [t for t in targets if t != 'None'] if targets else []
        return []
    except:
        return []

df_normal['target_list'] = df_normal['targets'].apply(parse_targets)
df_normal['has_target'] = df_normal['target_list'].apply(lambda x: len(x) > 0)

no_target = (~df_normal['has_target']).sum()
has_target = df_normal['has_target'].sum()

print(f"📊 타깃 분포:")
print(f"  타깃 없음: {no_target:,}개 ({no_target/len(df_normal)*100:.2f}%)")
print(f"  타깃 있음: {has_target:,}개 ({has_target/len(df_normal)*100:.2f}%)")

if has_target > 0:
    all_targets = []
    for targets in df_normal[df_normal['has_target']]['target_list']:
        all_targets.extend(targets)
    target_counter = Counter(all_targets)
    print(f"\n  예외 케이스 타깃 Top 5:")
    for target, count in target_counter.most_common(5):
        print(f"    - {target}: {count}회")
else:
    target_counter = Counter()

# ============================================================
# 4. Annotator Agreement 분석
# ============================================================
print("\n[4단계] Annotator Agreement 분석 중...")

def parse_annotators_robust(ann_str):
    """정규표현식으로 label 배열 추출"""
    try:
        if pd.isna(ann_str) or ann_str == '':
            return []
        
        # 'label': array([0, 1, 1]) 패턴
        match = re.search(r"'label':\s*array\(\[([^\]]+)\]\)", str(ann_str))
        if match:
            numbers_str = match.group(1)
            labels = [int(x.strip()) for x in numbers_str.split(',')]
            return labels
        
        # 'label': [0, 1, 1] 패턴
        match = re.search(r"'label':\s*\[([^\]]+)\]", str(ann_str))
        if match:
            numbers_str = match.group(1)
            labels = [int(x.strip()) for x in numbers_str.split(',') if x.strip().isdigit()]
            return labels
        
        return []
    except:
        return []

df_normal['annotator_labels'] = df_normal['annotators'].apply(parse_annotators_robust)
valid = df_normal['annotator_labels'].apply(lambda x: len(x) > 0).sum()
print(f"  파싱 성공: {valid}/{len(df_normal)} ({valid/len(df_normal)*100:.1f}%)")

# Agreement 점수
def calculate_agreement(labels):
    if not labels or len(labels) == 0:
        return 0
    # Normal = 1
    normal_count = sum(1 for l in labels if l == 1)
    return normal_count / len(labels) if len(labels) > 0 else 0

df_normal['agreement_score'] = df_normal['annotator_labels'].apply(calculate_agreement)

def categorize_agreement(score):
    if score >= 0.99:
        return "3/3 (완전 일치)"
    elif score >= 0.5:
        return "2/3 (다수 일치)"
    else:
        return "1/3 이하 (불일치)"

df_normal['agreement_category'] = df_normal['agreement_score'].apply(categorize_agreement)

agreement_counts = df_normal['agreement_category'].value_counts()

print(f"\n📊 Agreement 분포:")
for cat in ["3/3 (완전 일치)", "2/3 (다수 일치)", "1/3 이하 (불일치)"]:
    count = agreement_counts.get(cat, 0)
    pct = count / len(df_normal) * 100 if len(df_normal) > 0 else 0
    print(f"  {cat}: {count:,}개 ({pct:.2f}%)")

print(f"\n  평균 Agreement: {df_normal['agreement_score'].mean():.3f}")

# 샘플 확인
print(f"\n  샘플 확인 (처음 5개):")
for i in range(min(5, len(df_normal))):
    row = df_normal.iloc[i]
    print(f"    {i+1}. Labels: {row['annotator_labels']}, Score: {row['agreement_score']:.2f}")

# 파이 차트
import os
font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
if os.path.exists(font_path):
    from matplotlib.font_manager import FontProperties
    font_prop = FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#2ecc71', '#f39c12', '#e74c3c']

ordered_cats = ["3/3 (완전 일치)", "2/3 (다수 일치)", "1/3 이하 (불일치)"]
ordered_counts = [agreement_counts.get(cat, 0) for cat in ordered_cats]
ordered_labels = [cat for cat, count in zip(ordered_cats, ordered_counts) if count > 0]
ordered_values = [count for count in ordered_counts if count > 0]

if len(ordered_values) > 0:
    explode_list = [0.05] + [0] * (len(ordered_values) - 1)
    
    wedges, texts, autotexts = ax.pie(
        ordered_values,
        labels=ordered_labels,
        autopct='%1.1f%%',
        colors=colors[:len(ordered_values)],
        explode=explode_list,
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)

ax.set_title('Annotator Agreement Distribution (Normal)', fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('normal_agreement_distribution.png', dpi=300, bbox_inches='tight')
print("\n✅ 그래프 저장: normal_agreement_distribution.png")

# ============================================================
# 5. Borderline 케이스
# ============================================================
print("\n[5단계] Borderline 케이스 분석 중...")

df_borderline = df_normal[df_normal['agreement_category'] == "2/3 (다수 일치)"].copy()
print(f"📊 Borderline 샘플: {len(df_borderline):,}개 ({len(df_borderline)/len(df_normal)*100:.2f}%)")

if len(df_borderline) > 0:
    minority_labels = []
    for labels in df_borderline['annotator_labels']:
        for label in labels:
            if label != 1:  # Normal이 아닌 것
                minority_labels.append(label)
    
    minority_counter = Counter(minority_labels)
    label_names = {0: 'Hate', 1: 'Normal', 2: 'Offensive'}
    
    print(f"\n  소수 의견 분포:")
    for label, count in minority_counter.most_common():
        name = label_names.get(label, f'Label {label}')
        pct = count/len(minority_labels)*100 if len(minority_labels) > 0 else 0
        print(f"    {name}: {count}회 ({pct:.2f}%)")
    
    # CSV 저장
    export = df_borderline[['post_id', 'text', 'annotator_labels', 'target_list', 'token_length']].head(50)
    export.to_csv('borderline_samples.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ Borderline 샘플 저장: borderline_samples.csv")
else:
    print("  → Borderline 케이스 없음")
    minority_counter = Counter()

# ============================================================
# 6. 요약
# ============================================================
print("\n" + "="*100)
print("📋 P0 분석 결과 요약")
print("="*100)

print(f"\n【1. 기본 통계】")
print(f"  • Normal 샘플: {len(df_normal):,}개 (전체의 {len(df_normal)/len(df)*100:.1f}%)")
print(f"  • 평균 길이: {stats['mean']:.1f} 토큰")
print(f"  • 중앙값 길이: {stats['50%']:.1f} 토큰")

print(f"\n【2. Agreement 분포】")
for cat in ["3/3 (완전 일치)", "2/3 (다수 일치)", "1/3 이하 (불일치)"]:
    count = agreement_counts.get(cat, 0)
    pct = count / len(df_normal) * 100 if len(df_normal) > 0 else 0
    print(f"  • {cat}: {pct:.1f}%")

print(f"\n【3. Borderline】")
print(f"  • 2/3 일치: {len(df_borderline):,}개 ({len(df_borderline)/len(df_normal)*100:.1f}%)")

print(f"\n【4. 예외 케이스】")
print(f"  • 타깃 있음: {has_target:,}개 ({has_target/len(df_normal)*100:.2f}%)")

print("\n" + "="*100)
print("💡 한 줄 결론")
print("="*100)

q1 = int(stats['25%'])
q3 = int(stats['75%'])
mean_len = stats['mean']
median_len = stats['50%']
perfect_pct = agreement_counts.get('3/3 (완전 일치)', 0)/len(df_normal)*100
borderline_pct = agreement_counts.get('2/3 (다수 일치)', 0)/len(df_normal)*100

print(f"\n1. Normal 샘플은 평균 {mean_len:.0f} 토큰, 대부분 {q1}-{q3} 토큰 구간")
print(f"2. {perfect_pct:.1f}%가 완전 일치, {borderline_pct:.1f}%가 borderline")
print(f"3. 타깃 예외 {has_target/len(df_normal)*100:.2f}%")
print(f"4. 매칭 시 길이 ±5 토큰 권장 (중앙값 {median_len:.0f} 기준)")

# 요약 CSV
summary = pd.DataFrame({
    '항목': [
        'Normal 샘플 수',
        '평균 길이',
        '중앙값 길이',
        '완전 일치 비율',
        'Borderline 비율',
        '타깃 예외 비율'
    ],
    '값': [
        f"{len(df_normal):,}개",
        f"{mean_len:.1f} 토큰",
        f"{median_len:.1f} 토큰",
        f"{perfect_pct:.2f}%",
        f"{borderline_pct:.2f}%",
        f"{has_target/len(df_normal)*100:.2f}%"
    ]
})
summary.to_csv('normal_analysis_summary.csv', index=False, encoding='utf-8-sig')

print("\n" + "="*100)
print("✅ 분석 완료!")
print("\n생성된 파일:")
print("  1. normal_length_distribution.png")
print("  2. normal_agreement_distribution.png")
if len(df_borderline) > 0:
    print("  3. borderline_samples.csv")
print("  4. normal_analysis_summary.csv")
print("="*100)
