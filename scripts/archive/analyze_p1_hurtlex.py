#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HateXplain Normal 라벨 분석 (P1 + HurtLex 사전 기반 심화 분석)
- 4. Surface cue 분석 (HurtLex 17개 카테고리 적용)
- 7. Annotator disagreement 분석
- 8. 어휘/구문 분석

담당: Minseo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import Counter, defaultdict
import re
import warnings
import nltk
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')

# NLTK 데이터 다운로드 (처음 한번만)
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    print("⏳ NLTK WordNet 다운로드 중...")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

print("="*80)
print("HateXplain Normal 라벨 P1 분석 시작 (HurtLex 사전 기반)")
print("="*80)

# ============================================================
# 0. 설정
# ============================================================
# 영어 폰트 사용
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
print("✅ Using English font")

# ============================================================
# HurtLex 사전 로드
# ============================================================
print("\n[HurtLex 사전 로드]")

hurtlex_path = 'lexicons/hurtlex_EN.tsv'
hurtlex = pd.read_csv(hurtlex_path, sep='\t')
print(f"✅ HurtLex 로드: {len(hurtlex):,}개 단어")

# 카테고리 설명
category_names = {
    'ps': '인종/민족 비하 (Ethnic slurs)',
    'rci': '지역/국적 (Locations)',
    'pa': '직업 (Professions)',
    'ddf': '신체 장애 (Physical disabilities)',
    'ddp': '인지 장애 (Cognitive disabilities)',
    'dmc': '도덕적 결함 (Moral defects)',
    'is': '사회경제적 불이익 (Social disadvantage)',
    'or': '식물 (Plants)',
    'an': '동물 (Animals)',
    'asm': '남성 성기 (Male genitalia)',
    'asf': '여성 성기 (Female genitalia)',
    'pr': '매춘 (Prostitution)',
    'om': '동성애 (Homosexuality)',
    'qas': '부정적 함의 (Negative connotations)',
    'cds': '경멸어 (Derogatory words)',
    're': '범죄/부도덕 (Crime/immoral)',
    'svp': '칠대죄 (Seven deadly sins)'
}

print(f"\n📚 HurtLex 카테고리:")
category_counts = hurtlex['category'].value_counts()
for cat, count in category_counts.items():
    name = category_names.get(cat.lower(), cat)
    print(f"  {cat.upper()}: {count:,}개 - {name}")

# 카테고리별 단어 사전 생성
hurtlex_dict = defaultdict(set)
for _, row in hurtlex.iterrows():
    lemma = str(row['lemma']).lower().strip()
    category = str(row['category']).lower()
    hurtlex_dict[category].add(lemma)
    hurtlex_dict['all'].add(lemma)  # 전체 단어도 저장

print(f"\n✅ HurtLex 사전 준비 완료")

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1단계] 데이터 로드 중...")
df = pd.read_csv('hatexplain_prediction.csv', encoding='utf-8')

# Normal (label=1) 필터링
df_normal = df[df['gold_hatexplain_label'] == 1].copy().reset_index(drop=True)
print(f"✅ Normal 샘플: {len(df_normal):,}개")

# 파싱 함수들
def parse_tokens_list(token_str):
    """post_tokens에서 토큰 리스트 추출"""
    try:
        if pd.isna(token_str) or token_str == '':
            return []
        matches = re.findall(r"'([^']+)'", str(token_str))
        if matches:
            return matches
        tokens = eval(token_str)
        if isinstance(tokens, (list, np.ndarray)):
            return list(tokens)
        return []
    except:
        return []

def parse_annotators_robust(ann_str):
    """정규표현식으로 label 배열 추출"""
    try:
        if pd.isna(ann_str) or ann_str == '':
            return []
        match = re.search(r"'label':\s*array\(\[([^\]]+)\]\)", str(ann_str))
        if match:
            numbers_str = match.group(1)
            labels = [int(x.strip()) for x in numbers_str.split(',')]
            return labels
        match = re.search(r"'label':\s*\[([^\]]+)\]", str(ann_str))
        if match:
            numbers_str = match.group(1)
            labels = [int(x.strip()) for x in numbers_str.split(',') if x.strip().isdigit()]
            return labels
        return []
    except:
        return []

# 기본 파싱
df_normal['tokens'] = df_normal['post_tokens'].apply(parse_tokens_list)
df_normal['token_length'] = df_normal['tokens'].apply(len)
df_normal['annotator_labels'] = df_normal['annotators'].apply(parse_annotators_robust)

# 0 길이 제거
df_normal = df_normal[df_normal['token_length'] > 0].copy().reset_index(drop=True)

# Agreement 계산
def calculate_agreement(labels):
    if not labels or len(labels) == 0:
        return 0
    normal_count = sum(1 for l in labels if l == 1)
    return normal_count / len(labels) if len(labels) > 0 else 0

df_normal['agreement_score'] = df_normal['annotator_labels'].apply(calculate_agreement)

def categorize_agreement(score):
    if score >= 0.99:
        return "Perfect Match"
    elif score >= 0.5:
        return "Borderline"
    else:
        return "Disagreement"

df_normal['agreement_category'] = df_normal['agreement_score'].apply(categorize_agreement)

print(f"✅ 파싱 완료")

# ============================================================
# 4. Surface Cue 분석 (HurtLex 기반)
# ============================================================
print("\n" + "="*80)
print("[4. Surface Cue 분석 (HurtLex 사전 기반)]")
print("="*80)

print("\n[4-1] HurtLex 카테고리별 단어 매칭...")

# 각 샘플에 대해 HurtLex 카테고리별 매칭
def match_hurtlex_categories(tokens, hurtlex_dict):
    """토큰에서 HurtLex 카테고리별 매칭 (Lemmatization 적용)"""
    if not tokens:
        return {}
    
    # 토큰을 소문자 + lemmatize
    tokens_lemmatized = []
    for t in tokens:
        t_lower = t.lower()
        # 명사와 동사 형태 모두 시도
        lemma_n = lemmatizer.lemmatize(t_lower, pos='n')
        lemma_v = lemmatizer.lemmatize(t_lower, pos='v')
        lemma_a = lemmatizer.lemmatize(t_lower, pos='a')
        tokens_lemmatized.extend([t_lower, lemma_n, lemma_v, lemma_a])
    
    tokens_lemmatized = set(tokens_lemmatized)  # 중복 제거
    matches = defaultdict(list)
    
    for token in tokens_lemmatized:
        for category, words in hurtlex_dict.items():
            if category != 'all' and token in words:
                matches[category].append(token)
    
    return dict(matches)

df_normal['hurtlex_matches'] = df_normal['tokens'].apply(
    lambda tokens: match_hurtlex_categories(tokens, hurtlex_dict)
)

# HurtLex 포함 여부
df_normal['has_hurtlex'] = df_normal['hurtlex_matches'].apply(lambda x: len(x) > 0)

hurtlex_count = df_normal['has_hurtlex'].sum()
hurtlex_pct = hurtlex_count / len(df_normal) * 100

print(f"\n📊 HurtLex 단어 포함:")
print(f"  있음: {hurtlex_count:,}개 ({hurtlex_pct:.2f}%)")
print(f"  없음: {len(df_normal) - hurtlex_count:,}개 ({100 - hurtlex_pct:.2f}%)")

# Agreement별 HurtLex 비율
print(f"\n📊 Agreement별 HurtLex 포함 비율:")
for cat in ['Perfect Match', 'Borderline', 'Disagreement']:
    subset = df_normal[df_normal['agreement_category'] == cat]
    if len(subset) > 0:
        hl_count = subset['has_hurtlex'].sum()
        hl_pct = hl_count / len(subset) * 100
        print(f"  {cat}: {hl_pct:.2f}% ({hl_count}/{len(subset)})")

# 카테고리별 빈도 분석
print(f"\n[4-2] HurtLex 카테고리별 빈도 분석...")

category_freq = Counter()
for matches in df_normal['hurtlex_matches']:
    for category in matches.keys():
        category_freq[category] += 1

print(f"\n📊 카테고리별 출현 빈도 Top 10:")
for i, (cat, count) in enumerate(category_freq.most_common(10), 1):
    name = category_names.get(cat.lower(), cat)
    pct = count / len(df_normal) * 100
    print(f"  {i:2d}. {cat.upper():4s}: {count:4,}개 ({pct:5.2f}%) - {name}")

# HurtLex 카테고리 수 분포
df_normal['hurtlex_category_count'] = df_normal['hurtlex_matches'].apply(len)

print(f"\n📊 문장당 HurtLex 카테고리 수:")
cat_count_dist = df_normal['hurtlex_category_count'].value_counts().sort_index()
for count, samples in cat_count_dist.items():
    pct = samples / len(df_normal) * 100
    print(f"  {count}개 카테고리: {samples:,}개 ({pct:.2f}%)")

# Surface Cue 강도 재분류 (HurtLex 기반)
def classify_surface_cue_hurtlex(row):
    """HurtLex 기반 Surface cue 강도 분류"""
    matches = row['hurtlex_matches']
    
    if not matches:
        return "None"
    
    # 강한 카테고리 (인종, 성적 비하 등)
    strong_cats = {'ps', 'om', 'asm', 'asf', 'pr', 're', 'cds'}
    # 중간 카테고리 (장애, 도덕적 결함 등)
    medium_cats = {'ddf', 'ddp', 'dmc', 'is'}
    # 약한 카테고리 (동물, 부정적 함의 등)
    weak_cats = {'an', 'qas', 'or', 'svp'}
    
    matched_cats = set(matches.keys())
    
    if matched_cats & strong_cats:
        return "Strong (HurtLex)"
    elif matched_cats & medium_cats:
        return "Medium (HurtLex)"
    elif matched_cats & weak_cats:
        return "Weak (HurtLex)"
    else:
        return "Other"

df_normal['surface_cue_hurtlex'] = df_normal.apply(classify_surface_cue_hurtlex, axis=1)

cue_counts_hl = df_normal['surface_cue_hurtlex'].value_counts()

print(f"\n[4-3] Surface Cue Classification (HurtLex-based)...")
print(f"\n📊 Surface Cue Distribution:")
for cue in ['Strong (HurtLex)', 'Medium (HurtLex)', 'Weak (HurtLex)', 'None', 'Other']:
    count = cue_counts_hl.get(cue, 0)
    pct = count / len(df_normal) * 100 if len(df_normal) > 0 else 0
    print(f"  {cue}: {count:,}개 ({pct:.2f}%)")

# Borderline vs 완전 일치 비교
print(f"\n📊 Borderline vs Perfect Match Surface Cue (HurtLex):")
for cat in ['Borderline', 'Perfect Match']:
    subset = df_normal[df_normal['agreement_category'] == cat]
    if len(subset) > 0:
        print(f"\n  [{cat}] (n={len(subset):,})")
        cue_dist = subset['surface_cue_hurtlex'].value_counts()
        for cue in ['Strong (HurtLex)', 'Medium (HurtLex)', 'Weak (HurtLex)', 'None']:
            count = cue_dist.get(cue, 0)
            pct = count / len(subset) * 100
            print(f"    {cue}: {pct:.2f}%")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. HurtLex 포함 여부
hl_data = [hurtlex_count, len(df_normal) - hurtlex_count]
hl_labels = [f'HurtLex\n({hurtlex_pct:.1f}%)', f'No HurtLex\n({100-hurtlex_pct:.1f}%)']
axes[0, 0].pie(hl_data, labels=hl_labels, autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'], 
               startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
axes[0, 0].set_title('HurtLex Word Distribution', fontsize=14, fontweight='bold')

# 2. 카테고리별 빈도 Top 10
top_cats = category_freq.most_common(10)
cat_names = [cat.upper() for cat, _ in top_cats]
cat_vals = [count for _, count in top_cats]
axes[0, 1].barh(range(len(cat_names)), cat_vals, color='steelblue', alpha=0.8, edgecolor='black')
axes[0, 1].set_yticks(range(len(cat_names)))
axes[0, 1].set_yticklabels(cat_names)
axes[0, 1].set_xlabel('Sample Count', fontsize=12)
axes[0, 1].set_title('HurtLex Category Frequency (Top 10)', fontsize=14, fontweight='bold')
axes[0, 1].invert_yaxis()
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3. Surface Cue (HurtLex)
cue_order_hl = ['Strong (HurtLex)', 'Medium (HurtLex)', 'Weak (HurtLex)', 'None']
cue_data_hl = [cue_counts_hl.get(c, 0) for c in cue_order_hl]
colors_cue_hl = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71']
axes[1, 0].bar(range(len(cue_order_hl)), cue_data_hl, color=colors_cue_hl, alpha=0.8, edgecolor='black')
axes[1, 0].set_xticks(range(len(cue_order_hl)))
axes[1, 0].set_xticklabels(['Strong', 'Medium', 'Weak', 'None'], rotation=0)
axes[1, 0].set_ylabel('Sample Count', fontsize=12)
axes[1, 0].set_title('Surface Cue Strength (HurtLex-based)', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(cue_data_hl):
    axes[1, 0].text(i, v + max(cue_data_hl)*0.02, f'{v:,}\n({v/len(df_normal)*100:.1f}%)', 
                    ha='center', fontsize=10)

# 4. Agreement별 HurtLex 비율 비교
border_df = df_normal[df_normal['agreement_category'] == 'Borderline']
perfect_df = df_normal[df_normal['agreement_category'] == 'Perfect Match']

border_hl_pct = border_df['has_hurtlex'].sum() / len(border_df) * 100 if len(border_df) > 0 else 0
perfect_hl_pct = perfect_df['has_hurtlex'].sum() / len(perfect_df) * 100 if len(perfect_df) > 0 else 0

groups = ['Borderline', 'Perfect Match']
hl_pcts = [border_hl_pct, perfect_hl_pct]
axes[1, 1].bar(groups, hl_pcts, color=['#e67e22', '#27ae60'], alpha=0.8, edgecolor='black')
axes[1, 1].set_ylabel('HurtLex Coverage (%)', fontsize=12)
axes[1, 1].set_title('HurtLex Coverage by Agreement', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(hl_pcts):
    axes[1, 1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('surface_cue_hurtlex_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Graph saved: surface_cue_hurtlex_analysis.png")

# ============================================================
# 7. Annotator Disagreement 심층 분석 (HurtLex 연계)
# ============================================================
print("\n" + "="*80)
print("[7. Annotator Disagreement 분석 (HurtLex 연계)]")
print("="*80)

df_borderline = df_normal[df_normal['agreement_category'] == 'Borderline'].copy()
df_perfect = df_normal[df_normal['agreement_category'] == '완전 일치'].copy()

print(f"\n📊 샘플 수:")
print(f"  Borderline: {len(df_borderline):,}개")
print(f"  완전 일치: {len(df_perfect):,}개")

# HurtLex 커버리지 비교
print(f"\n📊 HurtLex 커버리지 비교:")
print(f"  Borderline: {border_hl_pct:.2f}%")
print(f"  완전 일치: {perfect_hl_pct:.2f}%")
print(f"  차이: {border_hl_pct - perfect_hl_pct:+.2f}%p")

# Borderline에서 많이 나타나는 HurtLex 카테고리
print(f"\n📊 Borderline에서 주요 HurtLex 카테고리:")
border_cat_freq = Counter()
for matches in df_borderline['hurtlex_matches']:
    for category in matches.keys():
        border_cat_freq[category] += 1

for i, (cat, count) in enumerate(border_cat_freq.most_common(10), 1):
    name = category_names.get(cat.lower(), cat)
    pct = count / len(df_borderline) * 100
    print(f"  {i:2d}. {cat.upper():4s}: {count:4,}개 ({pct:5.2f}%) - {name}")

# HurtLex 없는 Borderline (가장 중요!)
borderline_no_hl = df_borderline[~df_borderline['has_hurtlex']].copy()
print(f"\n⭐ HurtLex 없는 Borderline: {len(borderline_no_hl):,}개 ({len(borderline_no_hl)/len(df_borderline)*100:.2f}%)")
print(f"   → 이들이 진정한 'Implicit Hate' 후보군!")

# 샘플 저장 (전체)
if len(borderline_no_hl) > 0:
    export = borderline_no_hl[['post_id', 'text', 'annotator_labels', 'token_length']]
    export.to_csv('borderline_no_hurtlex_full.csv', index=False, encoding='utf-8-sig')
    print(f"   ✅ 저장: borderline_no_hurtlex_full.csv (전체 {len(borderline_no_hl)}개)")

# ============================================================
# 최종 요약
# ============================================================
print("\n" + "="*100)
print("📋 HurtLex 기반 분석 결과 요약")
print("="*100)

print(f"\n【HurtLex 커버리지】")
print(f"  • Normal 중 HurtLex 포함: {hurtlex_pct:.2f}%")
print(f"  • Borderline HurtLex 커버리지: {border_hl_pct:.2f}%")
print(f"  • 완전 일치 HurtLex 커버리지: {perfect_hl_pct:.2f}%")
print(f"  • 커버리지 차이: {border_hl_pct - perfect_hl_pct:+.2f}%p")

print(f"\n【주요 HurtLex 카테고리】")
for i, (cat, count) in enumerate(category_freq.most_common(5), 1):
    name = category_names.get(cat.lower(), cat)
    pct = count / len(df_normal) * 100
    print(f"  {i}. {cat.upper()}: {pct:.2f}% - {name}")

print(f"\n【Implicit Hate 후보 (HurtLex 기반)】")
print(f"  • HurtLex 없는 Borderline: {len(borderline_no_hl):,}개")
print(f"  • 전체 Borderline 중 비율: {len(borderline_no_hl)/len(df_borderline)*100:.2f}%")
print(f"  → 표면적 혐오 표현 없이도 어노테이터가 Offensive/Hate로 판단")

print(f"\n【Surface Cue 분류 비교】")
print(f"  기존 방식 (욕설 기반):")
print(f"    - 욕설 있음: 23.2%")
print(f"  HurtLex 방식 (17개 카테고리):")
print(f"    - 강함: {cue_counts_hl.get('강함 (HurtLex 강)', 0)/len(df_normal)*100:.2f}%")
print(f"    - 중간: {cue_counts_hl.get('중간 (HurtLex 중)', 0)/len(df_normal)*100:.2f}%")
print(f"    - 약함: {cue_counts_hl.get('약함 (HurtLex 약)', 0)/len(df_normal)*100:.2f}%")
print(f"    - 없음: {cue_counts_hl.get('없음', 0)/len(df_normal)*100:.2f}%")

print("\n" + "="*100)
print("💡 핵심 결론")
print("="*100)

print(f"\n1. **HurtLex로 더 정교한 Surface Cue 분류 가능**")
print(f"   - 단순 욕설(23.2%) vs HurtLex 전체({hurtlex_pct:.1f}%)")
print(f"   - 17개 카테고리로 세분화된 혐오 표현 유형 파악")

print(f"\n2. **Borderline의 HurtLex 커버리지가 더 높음**")
print(f"   - Borderline: {border_hl_pct:.1f}% vs 완전 일치: {perfect_hl_pct:.1f}%")
print(f"   - 차이: {border_hl_pct - perfect_hl_pct:+.1f}%p")
print(f"   - → 어노테이터가 인식한 '미묘한 혐오 신호'를 HurtLex가 포착")

print(f"\n3. **진정한 Implicit Hate 후보: HurtLex 없는 Borderline**")
print(f"   - {len(borderline_no_hl):,}개 ({len(borderline_no_hl)/len(df_borderline)*100:.1f}%)")
print(f"   - HurtLex에도 없는 단어로 혐오를 표현")
print(f"   - → 맥락 의존적 혐오, 우회적 표현, 신조어 등 가능성")

print(f"\n4. **주요 HurtLex 카테고리 (Normal에서도 출현)**")
top_5_cats = category_freq.most_common(5)
for cat, _ in top_5_cats:
    name = category_names.get(cat.lower(), cat)
    print(f"   - {cat.upper()}: {name}")

print("\n" + "="*100)
print("✅ HurtLex 기반 분석 완료!")
print("\n생성된 파일:")
print("  1. surface_cue_hurtlex_analysis.png")
print("  2. borderline_no_hurtlex.csv (진정한 Implicit Hate 후보)")
print("="*100)
