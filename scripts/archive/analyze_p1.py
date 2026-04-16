#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HateXplain Normal 라벨 분석 (P1: 실험 설계 직결)
- 4. Surface cue 분석
- 7. Annotator disagreement 분석
- 8. 어휘/구문 분석

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
print("HateXplain Normal 라벨 P1 분석 시작")
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
# 1. 데이터 로드 (P0에서 생성한 파싱 함수 재사용)
# ============================================================
print("\n[1단계] 데이터 로드 중...")
df = pd.read_csv('hatexplain_prediction.csv', encoding='utf-8')

# Normal (label=1) 필터링
df_normal = df[df['gold_hatexplain_label'] == 1].copy().reset_index(drop=True)
print(f"✅ Normal 샘플: {len(df_normal):,}개")

# 파싱 함수들
def parse_tokens_length(token_str):
    """post_tokens에서 길이만 추출"""
    try:
        if pd.isna(token_str) or token_str == '':
            return 0
        matches = re.findall(r"'([^']+)'", str(token_str))
        if matches:
            return len(matches)
        tokens = eval(token_str)
        if isinstance(tokens, (list, np.ndarray)):
            return len(tokens)
        return 0
    except:
        return 0

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
df_normal['token_length'] = df_normal['post_tokens'].apply(parse_tokens_length)
df_normal['tokens'] = df_normal['post_tokens'].apply(parse_tokens_list)
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
# 4. Surface Cue 분석
# ============================================================
print("\n" + "="*80)
print("[4. Surface Cue 분석]")
print("="*80)

# 4-1. 욕설/비속어 포함 비율
print("\n[4-1] 욕설/비속어 포함 비율 분석...")

# 욕설/비속어 사전 (확장 가능)
profanity_words = {
    'fuck', 'shit', 'damn', 'bitch', 'ass', 'hell', 'crap', 
    'bastard', 'asshole', 'dick', 'piss', 'cock', 'pussy',
    'motherfucker', 'fucker', 'fucking', 'fucked', 'shitty',
    'bullshit', 'goddamn', 'retard', 'retarded', 'stupid', 'idiot',
    'moron', 'dumb', 'dumbass', 'loser', 'suck', 'sucks',
    'faggot', 'fag', 'nigger', 'nigga', 'cunt', 'whore', 'slut'
}

def contains_profanity(text):
    """욕설/비속어 포함 여부"""
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    for word in profanity_words:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            return True
    return False

df_normal['has_profanity'] = df_normal['text'].apply(contains_profanity)

profanity_count = df_normal['has_profanity'].sum()
profanity_pct = profanity_count / len(df_normal) * 100

print(f"\n📊 욕설/비속어 포함:")
print(f"  있음: {profanity_count:,}개 ({profanity_pct:.2f}%)")
print(f"  없음: {len(df_normal) - profanity_count:,}개 ({100 - profanity_pct:.2f}%)")

# Agreement 카테고리별 욕설 비율
print(f"\n📊 Agreement별 욕설 비율:")
for cat in ['완전 일치', 'Borderline', '불일치']:
    subset = df_normal[df_normal['agreement_category'] == cat]
    if len(subset) > 0:
        prof_count = subset['has_profanity'].sum()
        prof_pct = prof_count / len(subset) * 100
        print(f"  {cat}: {prof_pct:.2f}% ({prof_count}/{len(subset)})")

# 4-2. Intensifier / 감정어 분석
print("\n[4-2] Intensifier / 감정어 분석...")

# Intensifier 사전
intensifiers = {
    'very', 'really', 'so', 'extremely', 'incredibly', 'absolutely',
    'totally', 'completely', 'utterly', 'quite', 'pretty', 'fairly',
    'super', 'highly', 'deeply', 'strongly', 'badly', 'seriously'
}

# 부정 감정어 사전
negative_emotion = {
    'hate', 'disgust', 'disgusting', 'terrible', 'horrible', 'awful',
    'bad', 'worse', 'worst', 'evil', 'wrong', 'sick', 'mad', 'angry',
    'rage', 'furious', 'annoyed', 'annoying', 'irritating', 'frustrating',
    'upset', 'sad', 'depressed', 'miserable', 'pathetic', 'loser',
    'fear', 'afraid', 'scared', 'terrified', 'worried', 'anxious',
    'ugly', 'nasty', 'gross', 'vile', 'disgusted', 'appalled'
}

def contains_intensifier(text):
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    for word in intensifiers:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            return True
    return False

def contains_negative_emotion(text):
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    for word in negative_emotion:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            return True
    return False

df_normal['has_intensifier'] = df_normal['text'].apply(contains_intensifier)
df_normal['has_negative_emotion'] = df_normal['text'].apply(contains_negative_emotion)

int_count = df_normal['has_intensifier'].sum()
neg_count = df_normal['has_negative_emotion'].sum()

print(f"\n📊 Intensifier:")
print(f"  있음: {int_count:,}개 ({int_count/len(df_normal)*100:.2f}%)")
print(f"\n📊 부정 감정어:")
print(f"  있음: {neg_count:,}개 ({neg_count/len(df_normal)*100:.2f}%)")

# Intensifier + 부정 감정어 조합
both = df_normal['has_intensifier'] & df_normal['has_negative_emotion']
both_count = both.sum()
print(f"\n📊 Intensifier + 부정 감정어 (조합):")
print(f"  있음: {both_count:,}개 ({both_count/len(df_normal)*100:.2f}%)")

# 4-3. Surface Cue 종합 분류
print("\n[4-3] Surface Cue 종합 분류...")

def classify_surface_cue(row):
    """표면 큐 강도 분류"""
    if row['has_profanity']:
        return "Strong (Profanity)"
    elif row['has_intensifier'] and row['has_negative_emotion']:
        return "Medium (Emotion+Intensifier)"
    elif row['has_negative_emotion']:
        return "Weak (Emotion only)"
    else:
        return "None"

df_normal['surface_cue'] = df_normal.apply(classify_surface_cue, axis=1)

cue_counts = df_normal['surface_cue'].value_counts()
print(f"\n📊 Surface Cue 분류:")
for cue in ['Strong (Profanity)', 'Medium (Emotion+Intensifier)', 'Weak (Emotion only)', 'None']:
    count = cue_counts.get(cue, 0)
    pct = count / len(df_normal) * 100 if len(df_normal) > 0 else 0
    print(f"  {cue}: {count:,}개 ({pct:.2f}%)")

# Borderline vs 완전 일치 비교
print(f"\n📊 Borderline vs Perfect Match Surface Cue:")
for cat in ['Borderline', 'Perfect Match']:
    subset = df_normal[df_normal['agreement_category'] == cat]
    if len(subset) > 0:
        print(f"\n  [{cat}] (n={len(subset):,})")
        cue_dist = subset['surface_cue'].value_counts()
        for cue in ['Strong (Profanity)', 'Medium (Emotion+Intensifier)', 'Weak (Emotion only)', 'None']:
            count = cue_dist.get(cue, 0)
            pct = count / len(subset) * 100
            print(f"    {cue}: {pct:.2f}%")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 왼쪽: Surface Cue 전체 분포
cue_order = ['Strong (Profanity)', 'Medium (Emotion+Intensifier)', 'Weak (Emotion only)', 'None']
cue_data = [cue_counts.get(c, 0) for c in cue_order]
colors_cue = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71']

axes[0].bar(range(len(cue_order)), cue_data, color=colors_cue, alpha=0.8, edgecolor='black')
axes[0].set_xticks(range(len(cue_order)))
axes[0].set_xticklabels(['Strong', 'Medium', 'Weak', 'None'], rotation=0)
axes[0].set_ylabel('Sample Count', fontsize=12)
axes[0].set_title('Surface Cue Strength Distribution', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

for i, v in enumerate(cue_data):
    axes[0].text(i, v + max(cue_data)*0.02, f'{v:,}\n({v/len(df_normal)*100:.1f}%)', 
                 ha='center', fontsize=10)

# 오른쪽: Agreement별 비교 (스택 바)
border_cue = df_normal[df_normal['agreement_category'] == 'Borderline']['surface_cue'].value_counts()
perfect_cue = df_normal[df_normal['agreement_category'] == 'Perfect Match']['surface_cue'].value_counts()

border_data = [border_cue.get(c, 0) / df_normal[df_normal['agreement_category'] == 'Borderline'].shape[0] * 100 for c in cue_order]
perfect_data = [perfect_cue.get(c, 0) / df_normal[df_normal['agreement_category'] == 'Perfect Match'].shape[0] * 100 for c in cue_order]

x = np.arange(len(cue_order))
width = 0.35

axes[1].bar(x - width/2, border_data, width, label='Borderline', color='#e67e22', alpha=0.8, edgecolor='black')
axes[1].bar(x + width/2, perfect_data, width, label='Perfect Match', color='#27ae60', alpha=0.8, edgecolor='black')

axes[1].set_xticks(x)
axes[1].set_xticklabels(['Strong', 'Medium', 'Weak', 'None'], rotation=0)
axes[1].set_ylabel('Percentage (%)', fontsize=12)
axes[1].set_title('Surface Cue by Agreement Category', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('surface_cue_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Graph saved: surface_cue_analysis.png")

# ============================================================
# 8. 어휘/구문 분석 (파생 데이터 템플릿용)
# ============================================================
print("\n" + "="*80)
print("[8. 어휘/구문 분석]")
print("="*80)

print("\n[8-1] 상위 빈출 단어 분석...")

# 전체 토큰 수집
all_tokens = []
for tokens in df_normal['tokens']:
    if tokens:
        all_tokens.extend([t.lower() for t in tokens if len(t) > 2])  # 2글자 이상만

# 불용어 제거
stopwords = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'is',
    'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'should', 'could', 'may', 'might', 'can',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
    'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
    'its', 'our', 'their', 'am', 'as', 'if', 'when', 'where', 'who', 'what'
}

filtered_tokens = [t for t in all_tokens if t not in stopwords and not t.startswith('http')]

token_counter = Counter(filtered_tokens)

print(f"\n📊 상위 빈출 단어 Top 30:")
for i, (word, count) in enumerate(token_counter.most_common(30), 1):
    print(f"  {i:2d}. '{word}': {count:,}회")

# 8-2. Bi-gram 분석
print("\n[8-2] Bi-gram 빈출 패턴...")

bigrams = []
for tokens in df_normal['tokens']:
    if tokens and len(tokens) > 1:
        clean_tokens = [t.lower() for t in tokens if len(t) > 2]
        for i in range(len(clean_tokens) - 1):
            bigram = (clean_tokens[i], clean_tokens[i+1])
            if clean_tokens[i] not in stopwords or clean_tokens[i+1] not in stopwords:
                bigrams.append(' '.join(bigram))

bigram_counter = Counter(bigrams)

print(f"\n📊 상위 Bi-gram Top 20:")
for i, (bigram, count) in enumerate(bigram_counter.most_common(20), 1):
    print(f"  {i:2d}. '{bigram}': {count:,}회")

# 8-3. 집단 언급 패턴 (일반화 vs 개인 지시)
print("\n[8-3] 집단 일반화 vs 개인 지시 표현 패턴...")

# 일반화 패턴 (복수형, all, every 등)
generalization_patterns = [
    r'\b(all|every|most|many|some|few)\s+(of\s+)?(the\s+)?(women|men|blacks|whites|muslims|christians|gays|immigrants|refugees)',
    r'\b(women|men|blacks|whites|muslims|christians|gays|immigrants|refugees)\s+(are|were|do|have|always|never|tend)',
    r'\bthey\s+(are|were|do|always|never)',
    r'\bpeople\s+(like|who|that)',
]

# 개인 지시 패턴
individual_patterns = [
    r'\b(a|an|one|this|that)\s+(woman|man|person|guy|girl|boy)',
    r'\b(he|she|him|her)\s+(is|was|did|does)',
    r'\bi\s+(know|met|saw|heard)',
]

def contains_generalization(text):
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    for pattern in generalization_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

def contains_individual(text):
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    for pattern in individual_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

df_normal['has_generalization'] = df_normal['text'].apply(contains_generalization)
df_normal['has_individual'] = df_normal['text'].apply(contains_individual)

gen_count = df_normal['has_generalization'].sum()
ind_count = df_normal['has_individual'].sum()

print(f"\n📊 집단 일반화 표현:")
print(f"  있음: {gen_count:,}개 ({gen_count/len(df_normal)*100:.2f}%)")
print(f"\n📊 개인 지시 표현:")
print(f"  있음: {ind_count:,}개 ({ind_count/len(df_normal)*100:.2f}%)")

# ============================================================
# 7. Annotator Disagreement 심층 분석
# ============================================================
print("\n" + "="*80)
print("[7. Annotator Disagreement 심층 분석]")
print("="*80)

print("\n[7-1] Borderline 케이스 특성 분석...")

df_borderline = df_normal[df_normal['agreement_category'] == 'Borderline'].copy()
df_perfect = df_normal[df_normal['agreement_category'] == '완전 일치'].copy()

print(f"\n📊 샘플 수:")
print(f"  Borderline: {len(df_borderline):,}개")
print(f"  완전 일치: {len(df_perfect):,}개")

# 길이 비교
print(f"\n📊 문장 길이 비교:")
print(f"  Borderline 평균: {df_borderline['token_length'].mean():.2f} 토큰")
print(f"  완전 일치 평균: {df_perfect['token_length'].mean():.2f} 토큰")

# Surface cue 비교 (이미 위에서 출력)

# 소수 의견 분석
print(f"\n[7-2] 소수 의견 상세 분석...")

minority_labels = []
for labels in df_borderline['annotator_labels']:
    for label in labels:
        if label != 1:  # Normal이 아닌 것
            minority_labels.append(label)

minority_counter = Counter(minority_labels)
label_names = {0: 'Hate', 2: 'Offensive'}

print(f"\n📊 Borderline Normal → 다른 라벨로 판단한 비율:")
total_minority = len(minority_labels)
for label, count in minority_counter.most_common():
    name = label_names.get(label, f'Label {label}')
    pct = count / total_minority * 100 if total_minority > 0 else 0
    print(f"  {name}: {count}회 ({pct:.2f}%)")

# 7-3. Borderline 샘플 예시 (Surface cue별)
print(f"\n[7-3] Borderline 샘플 저장 (Surface cue별 - 전체)...")

print("\n✅ Surface Cue별 Borderline 전체 저장 중...")
for cue in ['Strong (Profanity)', 'Medium (Emotion+Intensifier)', 'Weak (Emotion only)', 'None']:
    subset = df_borderline[df_borderline['surface_cue'] == cue].copy()
    if len(subset) > 0:
        print(f"  {cue}: {len(subset)}개")
        cue_short = cue.split()[0]  # 'Strong', 'Medium', 'Weak', 'None'
        export = subset[['post_id', 'text', 'annotator_labels', 'surface_cue', 'token_length']]
        filename = f'borderline_{cue_short.lower()}_full.csv'
        export.to_csv(filename, index=False, encoding='utf-8-sig')

# 8-3의 마지막: Borderline에서 일반화 표현 비율 비교
border_gen = df_borderline['has_generalization'].sum() if len(df_borderline) > 0 else 0
border_gen_pct = border_gen / len(df_borderline) * 100 if len(df_borderline) > 0 else 0

perfect_gen = df_perfect['has_generalization'].sum() if len(df_perfect) > 0 else 0
perfect_gen_pct = perfect_gen / len(df_perfect) * 100 if len(df_perfect) > 0 else 0

print(f"\n[8-3 계속] 일반화 표현 비율 비교:")
print(f"  Borderline: {border_gen_pct:.2f}%")
print(f"  완전 일치: {perfect_gen_pct:.2f}%")

# ============================================================
# 9. 파생 데이터 템플릿 생성
# ============================================================
print("\n" + "="*80)
print("[파생 데이터 템플릿 생성]")
print("="*80)

print("\n[템플릿 1] Surface Cue 'None' + Perfect Match → Cell D 후보")
template_cell_d = df_normal[
    (df_normal['surface_cue'] == 'None') & 
    (df_normal['agreement_category'] == 'Perfect Match')
].copy()
print(f"  샘플 수: {len(template_cell_d):,}개")

print("\n[템플릿 2] Surface Cue 'Weak/None' + Borderline → Cell C 후보")
template_cell_c = df_normal[
    (df_normal['surface_cue'].isin(['Weak (Emotion only)', 'None'])) & 
    (df_normal['agreement_category'] == 'Borderline')
].copy()
print(f"  샘플 수: {len(template_cell_c):,}개")

print("\n[템플릿 3] Surface Cue 'Strong' + Perfect Match → 일반 부정문 (Cell B 후보)")
template_cell_b = df_normal[
    (df_normal['surface_cue'] == 'Strong (Profanity)') & 
    (df_normal['agreement_category'] == 'Perfect Match')
].copy()
print(f"  샘플 수: {len(template_cell_b):,}개")

# 템플릿 저장 (전체)
if len(template_cell_d) > 0:
    template_cell_d[['post_id', 'text', 'surface_cue', 'token_length']].to_csv(
        'template_cell_d_full.csv', index=False, encoding='utf-8-sig'
    )
    print(f"  ✅ template_cell_d_full.csv 저장 (전체 {len(template_cell_d)}개)")

if len(template_cell_c) > 0:
    template_cell_c[['post_id', 'text', 'surface_cue', 'annotator_labels', 'token_length']].to_csv(
        'template_cell_c_full.csv', index=False, encoding='utf-8-sig'
    )
    print(f"  ✅ template_cell_c_full.csv 저장 (전체 {len(template_cell_c)}개)")

if len(template_cell_b) > 0:
    template_cell_b[['post_id', 'text', 'surface_cue', 'token_length']].to_csv(
        'template_cell_b_full.csv', index=False, encoding='utf-8-sig'
    )
    print(f"  ✅ template_cell_b_full.csv 저장 (전체 {len(template_cell_b)}개)")

# ============================================================
# 10. 최종 요약
# ============================================================
print("\n" + "="*100)
print("📋 P1 분석 결과 요약")
print("="*100)

print(f"\n【4. Surface Cue】")
print(f"  • 욕설/비속어: {profanity_pct:.2f}%")
print(f"  • 부정 감정어: {neg_count/len(df_normal)*100:.2f}%")
print(f"  • Surface Cue '없음': {cue_counts.get('없음', 0)/len(df_normal)*100:.2f}%")
print(f"  • Borderline 중 욕설 포함: {df_borderline['has_profanity'].sum()/len(df_borderline)*100:.2f}%")

print(f"\n【7. Annotator Disagreement】")
print(f"  • Borderline: {len(df_borderline):,}개 ({len(df_borderline)/len(df_normal)*100:.2f}%)")
print(f"  • 소수 의견 Offensive: {minority_counter.get(2, 0)}회 ({minority_counter.get(2, 0)/total_minority*100:.2f}%)")
print(f"  • 소수 의견 Hate: {minority_counter.get(0, 0)}회 ({minority_counter.get(0, 0)/total_minority*100:.2f}%)")

print(f"\n【8. 어휘/구문】")
print(f"  • 일반화 표현: {gen_count/len(df_normal)*100:.2f}%")
print(f"  • 개인 지시 표현: {ind_count/len(df_normal)*100:.2f}%")
print(f"  • Borderline 일반화 비율: {border_gen_pct:.2f}% (완전 일치 대비 {border_gen_pct - perfect_gen_pct:+.2f}%p)")

print(f"\n【파생 데이터 템플릿】")
print(f"  • Cell D 후보 (중립): {len(template_cell_d):,}개")
print(f"  • Cell C 후보 (Implicit Hate): {len(template_cell_c):,}개")
print(f"  • Cell B 후보 (일반 부정): {len(template_cell_b):,}개")

print("\n" + "="*100)
print("💡 핵심 결론")
print("="*100)

print(f"\n1. **Surface Cue와 Borderline의 상관관계**")
print(f"   - Normal 중 욕설 포함은 {profanity_pct:.1f}%에 불과")
print(f"   - Borderline 샘플 중 {df_borderline['has_profanity'].sum()/len(df_borderline)*100:.1f}%만 욕설 포함")
print(f"   - → 대부분의 Borderline은 '표면 cue 약함'으로 Cell C(Implicit Hate) 특성 확인")

print(f"\n2. **암묵적 혐오 표현 패턴 (Borderline 분석)**")
print(f"   - 소수 의견의 {minority_counter.get(2, 0)/total_minority*100:.0f}%가 Offensive로 판단")
print(f"   - 일반화 표현 비율: Borderline {border_gen_pct:.1f}% vs 완전 일치 {perfect_gen_pct:.1f}%")
print(f"   - → 집단 일반화 표현이 암묵적 편향 신호로 작용 가능")

print(f"\n3. **파생 데이터 생성 전략**")
print(f"   - Cell C(Implicit) 후보: {len(template_cell_c):,}개 (Surface cue 약함 + Borderline)")
print(f"   - Cell D(중립) 후보: {len(template_cell_d):,}개 (Surface cue 없음 + 완전 일치)")
print(f"   - → 실험 설계 시 이 두 그룹을 대조하여 암묵적 혐오 탐지 가능성 검증")

print(f"\n4. **다음 단계 제안**")
print(f"   - Borderline 샘플 {len(df_borderline):,}개의 육안 검수 (Cell C 후보 정제)")
print(f"   - 상위 빈출 단어 기반 파생 데이터 생성 템플릿 구축")
print(f"   - Hate/Offensive 데이터와 매칭하여 minimal pair 구성")

# 요약 저장
summary_p1 = pd.DataFrame({
    '항목': [
        'Surface Cue 없음',
        'Borderline 욕설 비율',
        '일반화 표현 (Borderline)',
        '일반화 표현 (완전 일치)',
        'Cell C 후보 (Implicit)',
        'Cell D 후보 (중립)',
    ],
    '값': [
        f"{cue_counts.get('없음', 0)/len(df_normal)*100:.2f}%",
        f"{df_borderline['has_profanity'].sum()/len(df_borderline)*100:.2f}%",
        f"{border_gen_pct:.2f}%",
        f"{perfect_gen_pct:.2f}%",
        f"{len(template_cell_c):,}개",
        f"{len(template_cell_d):,}개",
    ]
})
summary_p1.to_csv('p1_analysis_summary.csv', index=False, encoding='utf-8-sig')

print("\n" + "="*100)
print("✅ P1 분석 완료!")
print("\n생성된 파일:")
print("  1. surface_cue_analysis.png")
print("  2. borderline_강함.csv, borderline_중간.csv, borderline_약함.csv, borderline_없음.csv")
print("  3. template_cell_d.csv (Cell D 후보)")
print("  4. template_cell_c.csv (Cell C 후보)")
print("  5. template_cell_b.csv (Cell B 후보)")
print("  6. p1_analysis_summary.csv")
print("="*100)
