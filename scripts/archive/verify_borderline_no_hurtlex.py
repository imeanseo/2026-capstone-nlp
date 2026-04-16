#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
343개 HurtLex 없는 Borderline 샘플 재검증 스크립트

목적:
1. 명시적 슬러/비하어가 실제로 포함되어 있는지 확인
2. 진정한 Implicit Hate 후보를 3갈래로 분류
   - (가) 명시적 슬러 누락 (HurtLex/매칭 한계)
   - (나) 혐오 아님/애매 (비혐오 일상어, 밈, 중립 논의)
   - (다) 진정한 맥락형 Implicit Hate (프레이밍, 일반화, 암묵적 비하)
"""

import pandas as pd
import numpy as np
from collections import Counter
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
print("343개 HurtLex 없는 Borderline 샘플 재검증")
print("="*80)

# 1. 데이터 로드
df = pd.read_csv('results/p1_analysis/borderline_no_hurtlex.csv')
print(f"\n✅ 데이터 로드 완료: {len(df)}개 샘플\n")

# 2. 2차 슬러 리스트 (HurtLex에서 누락되었거나 변형된 슬러)
# 팀원 분석에서 언급된 것들 + 육안 확인된 것들
EXPLICIT_SLURS = {
    # 성소수자 비하
    'dyke', 'dykes', 'faggot', 'faggots', 'fag', 'fags',
    'lesbo', 'lesbos', 'homo', 'homos', 'queer', 'queers',
    
    # 인종/민족 비하
    'nigger', 'niggers', 'nigga', 'niggas', 'nigg', 
    'chink', 'chinks', 'gook', 'gooks',
    'ching chong', 'chinaman', 'chinamen',
    'towelhead', 'towelheads', 'raghead', 'ragheads',
    'camel jockey', 'sandnigger', 'shitskin', 'shitskins',
    'kike', 'kikes', 'heeb', 'heebs', 'zhid',
    'beaner', 'beaners', 'wetback', 'wetbacks',
    'spic', 'spics', 'spook', 'spooks',
    
    # 종교 비하 (철자 변형)
    'muzzie', 'muzzies', 'muzzy', 'moslem', 'moslems',
    'mohammaden', 'mohammedan',
    
    # 여성 비하
    'bitch', 'bitches', 'hoe', 'hoes', 'whore', 'whores',
    'slut', 'sluts', 'cunt', 'cunts', 'thot', 'thots',
    
    # 정치적 비하 (변형어)
    'feminazi', 'feminazis', 'libtard', 'libtards',
    
    # 기타
    'retard', 'retarded', 'retards',
    'tranny', 'trannies',
    'coon', 'coons',
    'dike', 'dikes',  # dyke의 철자 오류
    'dindu', 'dindu nuffin',  # 밈 형태의 인종 비하
    'negrophobe',  # 변형어
}

# 3. 비하 밈/구문 패턴
DEROGATORY_PATTERNS = [
    'ching chong',
    'dindu nuffin',
    'camel jockey',
    'towelhead',
    'sand nigger',
    'shitskin',
]

# 4. 명시적 슬러 검출 함수
def check_explicit_slur(text):
    """텍스트에서 명시적 슬러 검출"""
    if pd.isna(text):
        return False, []
    
    text_lower = text.lower()
    found_slurs = []
    
    # 단어 단위 매칭
    words = text_lower.split()
    for word in words:
        # 구두점 제거
        word_clean = word.strip('.,!?;:"\'()[]{}')
        if word_clean in EXPLICIT_SLURS:
            found_slurs.append(word_clean)
    
    # 구문 패턴 매칭
    for pattern in DEROGATORY_PATTERNS:
        if pattern in text_lower:
            found_slurs.append(pattern)
    
    return len(found_slurs) > 0, found_slurs

# 5. 일상어/밈 패턴 (혐오가 아닌 경우)
CASUAL_PATTERNS = {
    'i hate you': '친구 간 농담',
    'ghetto': '허름하다/저렴하다 의미로 사용',
    'redneck': '중립적 지역/문화 지칭',
    'immigrant': '중립적 정책 논의',
    'refugee': '중립적 정책 논의',
    'muslim': '중립적 종교 언급',
    'islam': '중립적 종교 언급',
}

def check_casual_pattern(text):
    """비혐오 일상어/논의 패턴 체크"""
    if pd.isna(text):
        return False, None
    
    text_lower = text.lower()
    for pattern, desc in CASUAL_PATTERNS.items():
        if pattern in text_lower:
            return True, desc
    return False, None

# 6. 전체 샘플 분석
print("명시적 슬러 재검증 중...")
df['has_explicit_slur'] = df['text'].apply(lambda x: check_explicit_slur(x)[0])
df['found_slurs'] = df['text'].apply(lambda x: check_explicit_slur(x)[1])
df['is_casual'] = df['text'].apply(lambda x: check_casual_pattern(x)[0])
df['casual_reason'] = df['text'].apply(lambda x: check_casual_pattern(x)[1])

# 7. 3갈래 분류
def classify_sample(row):
    """샘플을 3갈래로 분류"""
    if row['has_explicit_slur']:
        return '(가) 명시적 슬러 누락'
    elif row['is_casual']:
        return '(나) 비혐오/애매'
    else:
        return '(다) 진정한 맥락형'

df['classification'] = df.apply(classify_sample, axis=1)

# 8. 결과 요약
print("\n" + "="*80)
print("📊 재분류 결과 요약")
print("="*80)

classification_counts = df['classification'].value_counts()
for cat, count in classification_counts.items():
    pct = count / len(df) * 100
    print(f"{cat:30s}: {count:4d}개 ({pct:5.1f}%)")

print(f"\n{'총 샘플':30s}: {len(df):4d}개")

# 9. (가) 명시적 슬러 누락 샘플 분석
print("\n" + "="*80)
print("(가) 명시적 슬러 누락 - 상위 10개 슬러")
print("="*80)

explicit_df = df[df['classification'] == '(가) 명시적 슬러 누락']
all_slurs = []
for slurs_list in explicit_df['found_slurs']:
    all_slurs.extend(slurs_list)

slur_counts = Counter(all_slurs)
for slur, count in slur_counts.most_common(10):
    print(f"  {slur:20s}: {count:3d}회")

# 10. (나) 비혐오/애매 샘플 분석
print("\n" + "="*80)
print("(나) 비혐오/애매 - 주요 패턴")
print("="*80)

casual_df = df[df['classification'] == '(나) 비혐오/애매']
if len(casual_df) > 0:
    casual_reasons = casual_df['casual_reason'].value_counts()
    for reason, count in casual_reasons.items():
        if reason:
            print(f"  {reason:40s}: {count:3d}개")

# 11. (다) 진정한 맥락형 샘플 분석
print("\n" + "="*80)
print("(다) 진정한 맥락형 Implicit Hate - 샘플 예시 (5개)")
print("="*80)

implicit_df = df[df['classification'] == '(다) 진정한 맥락형']
for i, row in implicit_df.head(5).iterrows():
    print(f"\n{i+1}. {row['text'][:120]}")
    print(f"   토큰 수: {row['token_length']}")

# 12. 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 12-1. 분류 비율 파이 차트
colors = ['#e74c3c', '#f39c12', '#2ecc71']
wedges, texts, autotexts = axes[0].pie(
    classification_counts.values,
    labels=classification_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    textprops={'fontsize': 11, 'weight': 'bold'}
)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(13)

axes[0].set_title('HurtLex 없는 Borderline 343개 재분류', 
                  fontsize=14, fontweight='bold', pad=20)

# 12-2. 명시적 슬러 Top 10 막대 그래프
if len(slur_counts) > 0:
    top_slurs = dict(slur_counts.most_common(10))
    axes[1].barh(list(top_slurs.keys()), list(top_slurs.values()), 
                 color='#e74c3c', alpha=0.7)
    axes[1].set_xlabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Top 10 Explicit Slurs Found', 
                      fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    
    # 값 표시
    for i, (k, v) in enumerate(top_slurs.items()):
        axes[1].text(v + 0.5, i, str(v), va='center', fontsize=10)

plt.tight_layout()
plt.savefig('results/p1_analysis/borderline_reclassification.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ 시각화 저장: results/p1_analysis/borderline_reclassification.png")

# 13. 각 분류별 CSV 저장
for classification in df['classification'].unique():
    subset = df[df['classification'] == classification]
    
    # 파일명 생성
    if '가' in classification:
        filename = 'borderline_explicit_slurs.csv'
    elif '나' in classification:
        filename = 'borderline_casual_nonhate.csv'
    else:
        filename = 'borderline_true_implicit.csv'
    
    filepath = f'results/p1_analysis/{filename}'
    subset[['post_id', 'text', 'found_slurs', 'casual_reason', 'token_length']].to_csv(
        filepath, index=False, encoding='utf-8-sig'
    )
    print(f"✅ CSV 저장: {filepath} ({len(subset)}개)")

# 14. 요약 통계 CSV
summary_data = {
    '분류': list(classification_counts.index),
    '샘플 수': list(classification_counts.values),
    '비율 (%)': [f"{v/len(df)*100:.1f}" for v in classification_counts.values]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('results/p1_analysis/borderline_reclassification_summary.csv',
                  index=False, encoding='utf-8-sig')
print(f"\n✅ 요약 저장: results/p1_analysis/borderline_reclassification_summary.csv")

# 15. 최종 결론
print("\n" + "="*80)
print("📌 최종 결론")
print("="*80)

implicit_count = len(implicit_df)
implicit_pct = implicit_count / len(df) * 100
explicit_count = len(explicit_df)
explicit_pct = explicit_count / len(df) * 100

print(f"""
✅ 343개 재검증 완료

1. **명시적 슬러 누락 ({explicit_count}개, {explicit_pct:.1f}%)**
   - HurtLex/레마타이제이션의 한계로 인한 누락
   - dyke, muzzie, ching chong 등 변형어·밈
   - → 이들은 "명시적 비하"로 재분류 필요

2. **비혐오/애매 ({len(casual_df)}개, {len(casual_df)/len(df)*100:.1f}%)**
   - 일상어, 친구 간 농담, 중립적 정책 논의
   - → Cell C 후보에서 제외

3. **진정한 맥락형 Implicit Hate ({implicit_count}개, {implicit_pct:.1f}%)**
   - 표면적 슬러 없음 + HurtLex에도 없음
   - 프레이밍, 일반화, 구조적 편향으로 혐오 전달
   - → **연구의 핵심 대상**

💡 권장사항:
- 논문/보고서에서는 "343개 전체"가 아닌 "{implicit_count}개"를 Implicit Hate 후보로 명시
- 명시적 슬러 {explicit_count}개는 별도 분석 (HurtLex 개선 방안)
- 팀원의 "Cue 없는 Hate 1,158개"와 교차 분석 제안
""")

print("="*80)
print("✅ 스크립트 완료!")
print("="*80)
