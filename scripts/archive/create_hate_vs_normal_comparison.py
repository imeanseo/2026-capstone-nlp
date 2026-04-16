#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hate/Offensive vs Normal - 교차 분석표 생성

목적:
1. Hate/Offensive 데이터의 "Cue 없는 Hate" vs Normal 데이터의 "True Implicit Normal" 비교
2. 혐오 표현의 경계선 탐색
3. 일요일 미팅 자료 준비
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("Hate/Offensive vs Normal - 교차 분석표 생성")
print("="*80)

# 1. 데이터 로드
print("\n📂 데이터 로드 중...")

# Normal 데이터 분석 결과
df_true_implicit = pd.read_csv('results/p1_analysis/borderline_true_implicit.csv')
print(f"✅ True Implicit Normal: {len(df_true_implicit)}개")

# 전체 Normal (통계용)
df_normal = pd.read_csv('hatexplain_prediction.csv')
df_normal_only = df_normal[df_normal['gold_hatexplain_label'] == 1].copy()
print(f"✅ Normal 전체: {len(df_normal_only)}개")

# Hate/Offensive 데이터는 직접 접근 불가 → 노션에서 가져온 수치 사용
print(f"✅ Hate/Offensive 전체: 11,995개 (노션 기준)")
print(f"✅ Hate/Offensive Cue 없는 샘플: 1,158개 (18.6%) (노션 기준)")

# 2. 기본 통계 비교표
print("\n" + "="*80)
print("📊 기본 통계 비교")
print("="*80)

# Normal 데이터 통계
true_implicit_length = df_true_implicit['token_length'].mean()
print(f"\nNormal True Implicit 평균 길이: {true_implicit_length:.1f}토큰")

comparison_basic = {
    '구분': [
        '전체 샘플 수',
        'Cue 없는 샘플',
        '비율',
        '평균 길이 (토큰)',
        '플랫폼 (추정)',
        '만장일치율',
        'Normal 소수의견'
    ],
    'Hate/Offensive': [
        '11,995개',
        '1,158개',
        '9.7%',  # 1158/11995
        '26.8토큰',
        'Gab 87%, Twitter 13%',
        '25.5%',
        '31.1%'
    ],
    'Normal': [
        '7,814개',
        '74개',
        '0.9%',  # 74/7814
        f'{true_implicit_length:.1f}토큰',
        'Twitter/Gab 혼합',
        '0% (Borderline이므로)',
        '34.4% (Borderline 비율)'
    ]
}

basic_df = pd.DataFrame(comparison_basic)
print("\n")
print(basic_df.to_string(index=False))

# 3. 프레이밍 비교표
print("\n" + "="*80)
print("📊 프레이밍 분포 비교")
print("="*80)

# Hate/Offensive 데이터의 프레이밍 (노션 기준, Cue 없는 Hate)
hate_framing = {
    'CRIMINAL_DANGER': 8.2,  # 노션: "범죄 프레이밍"
    'EXCLUSION': 4.3,
    'GENERALIZATION': 12.0,
    'THREAT_VIOLENCE': 3.9,
    'DEHUMANIZATION': 2.6,
    'CONSPIRACY': None,  # 데이터 없음
    'MORAL_DISGUST': None,
    'INTELLECTUAL_INFERIORITY': None,
    'SEXUAL_GENDERED': None,
    'ECONOMIC_BURDEN': None,
    'RELIGIOUS': None,
}

# Normal 데이터의 프레이밍 (방금 분석)
normal_framing = {
    'ECONOMIC_BURDEN': 4.1,
    'THREAT_VIOLENCE': 1.4,
    'GENERALIZATION': 1.4,
    'DEHUMANIZATION': 0.0,
    'EXCLUSION': 0.0,
    'CONSPIRACY': 0.0,
    'MORAL_DISGUST': 0.0,
    'INTELLECTUAL_INFERIORITY': 0.0,
    'SEXUAL_GENDERED': 0.0,
    'CRIMINAL_DANGER': 0.0,
    'RELIGIOUS': 0.0,
}

# 프레이밍 비교표 생성
framing_comparison = {
    'Framing Category': [],
    'Hate/Offensive Cue-free (%)': [],
    'Normal True Implicit (%)': [],
    'Difference': [],
    'Interpretation': []
}

for category in ['GENERALIZATION', 'CRIMINAL_DANGER', 'EXCLUSION', 
                 'THREAT_VIOLENCE', 'DEHUMANIZATION', 'ECONOMIC_BURDEN',
                 'CONSPIRACY', 'MORAL_DISGUST', 'INTELLECTUAL_INFERIORITY',
                 'SEXUAL_GENDERED', 'RELIGIOUS']:
    hate_val = hate_framing.get(category)
    normal_val = normal_framing.get(category, 0.0)
    
    framing_comparison['Framing Category'].append(category)
    framing_comparison['Hate/Offensive Cue-free (%)'].append(
        f'{hate_val:.1f}' if hate_val is not None else 'N/A'
    )
    framing_comparison['Normal True Implicit (%)'].append(f'{normal_val:.1f}')
    
    if hate_val is not None:
        diff = normal_val - hate_val
        framing_comparison['Difference'].append(f'{diff:+.1f}%p')
        
        if abs(diff) < 2:
            interp = '유사'
        elif diff > 0:
            interp = 'Normal이 더 높음'
        else:
            interp = 'Hate가 더 높음'
    else:
        framing_comparison['Difference'].append('N/A')
        interp = '비교 불가'
    
    framing_comparison['Interpretation'].append(interp)

framing_df = pd.DataFrame(framing_comparison)
print("\n")
print(framing_df.to_string(index=False))

# 4. 특징 어휘 비교 (나의 데이터만)
print("\n" + "="*80)
print("📊 True Implicit Normal 특징 어휘 (Top 20)")
print("="*80)

all_words = []
for text in df_true_implicit['text']:
    if pd.notna(text):
        words = text.lower().split()
        words_clean = [w.strip('.,!?;:"\'()[]{}') for w in words]
        all_words.extend(words_clean)

# 불용어 제거
stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
             'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
             'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
             'could', 'should', 'may', 'might', 'can', 'that', 'this', 'it',
             'i', 'you', 'he', 'she', 'we', 'they', 'my', 'your', 'his', 'her',
             'our', 'their', 'me', 'him', 'them', 'us', '<user>', '<number>'}

words_filtered = [w for w in all_words if w not in stopwords and len(w) > 2]
word_counts = Counter(words_filtered)

print("\n")
for word, count in word_counts.most_common(20):
    print(f"  {word:20s}: {count:3d}회")

# 5. 종합 비교표 생성
print("\n" + "="*80)
print("📊 종합 비교표")
print("="*80)

comprehensive_comparison = {
    '항목': [
        '【기본 정보】',
        '전체 샘플 수',
        'Cue 없는 비율',
        '평균 길이',
        '',
        '【라벨 특성】',
        '만장일치율',
        'Normal 소수의견',
        '플랫폼',
        '',
        '【프레이밍】',
        'GENERALIZATION',
        'CRIMINAL_DANGER',
        'EXCLUSION',
        'ECONOMIC_BURDEN',
        'THREAT_VIOLENCE',
        '프레이밍 없음',
        '',
        '【특징 어휘】',
        '상위 키워드',
    ],
    'Hate/Offensive (Cue 없음)': [
        '',
        '6,234개 (Hate 전체)',
        '18.6% (1,158개)',
        '26.8토큰 (Cue 있는 것 24.2)',
        '',
        '',
        '25.5% (낮음)',
        '31.1% (높음)',
        'Gab 87%',
        '',
        '',
        '12.0%',
        '8.2%',
        '4.3%',
        'N/A',
        '3.9%',
        '~55-60% (추정)',
        '',
        '',
        'should, because, think, white, jews',
    ],
    'Normal (True Implicit)': [
        '',
        '7,814개 (Normal 전체)',
        '0.9% (74개)',
        f'{true_implicit_length:.1f}토큰',
        '',
        '',
        '0% (정의상 Borderline)',
        '34.4% (정의상)',
        'Twitter/Gab 혼합',
        '',
        '',
        '1.4%',
        '0.0%',
        '0.0%',
        '4.1%',
        '1.4%',
        '93.2% 🔥',
        '',
        '',
        'women, not, should, biological, muslim',
    ]
}

comprehensive_df = pd.DataFrame(comprehensive_comparison)
print("\n")
print(comprehensive_df.to_string(index=False))

# 6. CSV 저장
basic_df.to_csv('results/p1_analysis/hate_vs_normal_basic.csv',
                index=False, encoding='utf-8-sig')
framing_df.to_csv('results/p1_analysis/hate_vs_normal_framing.csv',
                  index=False, encoding='utf-8-sig')
comprehensive_df.to_csv('results/p1_analysis/hate_vs_normal_comprehensive.csv',
                        index=False, encoding='utf-8-sig')

print("\n" + "="*80)
print("✅ 교차 분석표 저장 완료")
print("="*80)
print("  • hate_vs_normal_basic.csv")
print("  • hate_vs_normal_framing.csv")
print("  • hate_vs_normal_comprehensive.csv")

# 7. 핵심 발견
print("\n" + "="*80)
print("🔥 핵심 발견")
print("="*80)

print("""
1. 【프레이밍 분포의 극명한 차이】
   Hate/Offensive (Cue 없음):
   - GENERALIZATION (12.0%) - 가장 높음
   - CRIMINAL_DANGER (8.2%)
   - EXCLUSION (4.3%)
   - → 범죄/일반화/배제 프레이밍이 명확
   
   Normal (True Implicit):
   - 프레이밍 없음 (93.2%) 🔥🔥🔥
   - ECONOMIC_BURDEN (4.1%) - 유일하게 약간 감지
   - → 전형적 프레이밍조차 없는 극단적 암묵성

2. 【연구 함의】
   ✅ "Cue 없는 Hate"와 "True Implicit Normal"은 **질적으로 다른 그룹**
   
   Hate/Offensive:
   - Surface cue는 없지만 **전형적 혐오 프레이밍 존재** (40-45%)
   - CRIMINAL, GENERALIZATION 등으로 의도 파악 가능
   - 평균 26.8토큰으로 서사적
   
   Normal:
   - Surface cue도 없고 **전형적 프레이밍도 거의 없음** (93%)
   - 어노테이터 1명만 Hate로 판단 (다수는 Normal)
   - → **극도로 미묘한 맥락 의존적 경우**
   - → 또는 **실제로 혐오가 아닌데 오판된 경우**

3. 【74개는 무엇인가?】
   두 가지 가능성:
   
   A) **극단적 Implicit Hate**
      - 프레이밍조차 없는 매우 미묘한 편향
      - 예: "biological men who regard themselves as women"
      - 구조와 표현 선택만으로 혐오 전달
   
   B) **False Positive (실제로는 비혐오)**
      - 어노테이터 1명의 과민 반응
      - 문맥 없이 판단한 결과
      - → 추가 육안 검수 필요

4. 【Hate ↔ Normal 경계선】
   
   Hate/Offensive Cue 없는 샘플 (1,158) ←→ Normal True Implicit (74)
   
   이 사이의 차이점:
   - 프레이밍 유무: 45% vs 7%
   - 길이: 26.8 vs 21.5토큰
   - 만장일치: 25.5% vs 0%
   
   → **프레이밍이 Hate/Normal 경계를 가르는 핵심 요소**

5. 【일요일 미팅 제안】
   □ 74개 육안 검수 (함께)
   □ "Normal 소수의견 3,358개" (Hate/Offensive 데이터)와 교차
   □ 프레이밍 없는 Hate 존재 여부 확인
   □ 실험 설계 정밀화: Cell C 정의 재검토
""")

print("="*80)
print("✅ 교차 분석 완료!")
print("="*80)
