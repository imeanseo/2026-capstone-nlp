#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P3: 프레이밍 분석 (문헌 기반)

목적:
1. 74개 진정한 Implicit Hate 샘플에 11개 프레이밍 카테고리 적용
2. 팀원의 Cue 없는 Hate(1,158개)와 비교
3. 문헌 근거 명시

참고 문헌:
- ElSherief et al. (2021): "Latent Hatred: A Benchmark for Understanding Implicit Hate Speech"
  → 암묵적 혐오의 유형 분류 (white grievance, inferiority language, incitement to violence)
  
- Ocampo et al. (2023, EACL): "An In-depth Analysis of Implicit and Subtle Hate Speech Messages"
  → Implicit HS 18개 속성 (irony, metaphor, exaggeration, rhetorical question 등)
  
- Carvalho et al. (2023): Portuguese hate speech 연구
  → Stereotypes, threats (realistic/symbolic), dehumanization, fallacies (appeal to fear/action)

팀원 분석 기반:
- 11개 프레이밍 카테고리는 위 문헌들의 조합으로 구성됨
- 수작업 lexicon은 각 카테고리의 전형적 표현 수집
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 한글 폰트 설정
import platform
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("P3: 프레이밍 분석 (문헌 기반)")
print("="*80)

# 1. 11개 프레이밍 카테고리 정의 (팀원 분석 + 문헌 근거)
# 출처: ElSherief (2021), Ocampo (2023), Carvalho (2023)

FRAMING_LEXICONS = {
    # 1. DEHUMANIZATION - 비인간화 (Carvalho 2023)
    # 집단을 동물·사물로 묘사
    'DEHUMANIZATION': {
        'keywords': ['animal', 'animals', 'monkey', 'monkeys', 'cockroach', 
                     'vermin', 'subhuman', 'beast', 'savage', 'creature',
                     'rat', 'rats', 'dog', 'dogs', 'pig', 'pigs'],
        'source': 'Carvalho et al. (2023)',
        'description': '집단을 동물이나 하위 인간으로 묘사'
    },
    
    # 2. THREAT_VIOLENCE - 폭력 위협 (ElSherief 2021: incitement to violence)
    # 직접적 폭력 촉구
    'THREAT_VIOLENCE': {
        'keywords': ['kill', 'killed', 'hang', 'gas', 'exterminate', 'genocide',
                     'murder', 'shoot', 'bomb', 'die', 'death', 'dead',
                     'destroy', 'elimination', 'cleanse'],
        'source': 'ElSherief et al. (2021)',
        'description': '집단에 대한 폭력 위협이나 촉구'
    },
    
    # 3. EXCLUSION - 배제 (구문 패턴)
    # "go back", "don't belong" 등
    'EXCLUSION': {
        'patterns': [
            r'\bgo back\b',
            r'\bdon\'t belong\b',
            r'\bdoesn\'t belong\b',
            r'\bour country\b',
            r'\bour land\b',
            r'\bget out\b',
            r'\bleave\b.*\bcountry\b',
            r'\bdeport\b',
            r'\bremove\b'
        ],
        'source': 'ElSherief et al. (2021)',
        'description': '집단을 공간/사회에서 배제하려는 표현'
    },
    
    # 4. CONSPIRACY - 음모론 (ElSherief 2021: white grievance)
    # 집단이 권력을 장악하거나 침략한다는 서사
    'CONSPIRACY': {
        'keywords': ['control', 'controls', 'destroy', 'invasion', 'replace',
                     'replacement', 'globalist', 'globalists', 'shill', 'agenda',
                     'take over', 'takeover', 'plot', 'scheme', 'conspiracy'],
        'source': 'ElSherief et al. (2021)',
        'description': '집단이 사회를 장악/침략한다는 음모론'
    },
    
    # 5. MORAL_DISGUST - 도덕적 혐오감 (Ocampo 2023: sentiment)
    # 도덕적·미학적 역겨움 표현
    'MORAL_DISGUST': {
        'keywords': ['disgusting', 'filthy', 'dirty', 'degenerate', 'abomination',
                     'repulsive', 'revolting', 'vile', 'sick', 'gross'],
        'source': 'Ocampo et al. (2023)',
        'description': '도덕적·미학적 혐오감 표현'
    },
    
    # 6. INTELLECTUAL_INFERIORITY - 지능 비하 (ElSherief 2021: inferiority language)
    # 지적 능력 폄하
    'INTELLECTUAL_INFERIORITY': {
        'keywords': ['stupid', 'dumb', 'idiot', 'idiots', 'moron', 'morons',
                     'retarded', 'retard', 'ignorant', 'fool', 'fools',
                     'brainless', 'mindless', 'low iq'],
        'source': 'ElSherief et al. (2021)',
        'description': '집단의 지적 능력 폄하'
    },
    
    # 7. SEXUAL_GENDERED - 성적/젠더 프레이밍
    # 성적 대상화, 젠더 고정관념
    'SEXUAL_GENDERED': {
        'keywords': ['rape', 'raped', 'raping', 'slut', 'sluts', 'whore', 'whores',
                     'breed', 'breeding', 'hoe', 'hoes', 'promiscuous', 'thot'],
        'source': 'Carvalho et al. (2023)',
        'description': '성적 대상화나 젠더 고정관념'
    },
    
    # 8. CRIMINAL_DANGER - 범죄/위험 (Carvalho 2023: realistic threat)
    # 집단을 범죄자·위협으로 묘사
    'CRIMINAL_DANGER': {
        'keywords': ['crime', 'criminal', 'criminals', 'terrorist', 'terrorists',
                     'illegal', 'illegals', 'steal', 'stealing', 'thief', 'thieves',
                     'danger', 'dangerous', 'threat', 'violent', 'violence',
                     'pedophile', 'pedophiles', 'radical'],
        'source': 'Carvalho et al. (2023)',
        'description': '집단을 범죄자나 위협으로 묘사'
    },
    
    # 9. ECONOMIC_BURDEN - 경제적 부담 (Carvalho 2023: economic threat)
    # 복지 의존, 경제적 부담 프레이밍
    'ECONOMIC_BURDEN': {
        'keywords': ['welfare', 'leech', 'leeches', 'freeloader', 'freeloaders',
                     'parasite', 'parasites', 'burden', 'tax', 'taxes',
                     'benefits', 'free', 'money', 'lazy'],
        'source': 'Carvalho et al. (2023)',
        'description': '집단이 경제적 부담이라는 프레이밍'
    },
    
    # 10. RELIGIOUS - 종교적 프레이밍
    # 종교 기반 공격
    'RELIGIOUS': {
        'keywords': ['jihad', 'jihadi', 'sharia', 'infidel', 'infidels',
                     'satanic', 'devil', 'hell', 'sin', 'sinner',
                     'radical islam', 'islamic'],
        'source': 'ElSherief et al. (2021)',
        'description': '종교를 통한 부정적 프레이밍'
    },
    
    # 11. GENERALIZATION - 일반화 (Ocampo 2023: exaggeration)
    # "all X are Y" 구조
    'GENERALIZATION': {
        'patterns': [
            r'\ball\b.*\bare\b',
            r'\ball\b.*\bdo\b',
            r'\bevery\b.*\bis\b',
            r'\bevery\b.*\bdoes\b',
            r'\balways\b',
            r'\bnever\b'
        ],
        'source': 'Ocampo et al. (2023)',
        'description': '집단 전체를 일반화하는 표현'
    }
}

# 2. 프레이밍 감지 함수
def detect_framing(text, framing_category):
    """텍스트에서 특정 프레이밍 카테고리 감지"""
    if pd.isna(text):
        return False, []
    
    text_lower = text.lower()
    found_items = []
    
    category_def = FRAMING_LEXICONS[framing_category]
    
    # 키워드 기반 매칭
    if 'keywords' in category_def:
        words = text_lower.split()
        for word in words:
            word_clean = word.strip('.,!?;:"\'()[]{}')
            if word_clean in category_def['keywords']:
                found_items.append(word_clean)
    
    # 패턴 기반 매칭
    if 'patterns' in category_def:
        for pattern in category_def['patterns']:
            if re.search(pattern, text_lower):
                match = re.search(pattern, text_lower)
                found_items.append(match.group(0))
    
    return len(found_items) > 0, found_items

# 3. 데이터 로드
print("\n📂 데이터 로드 중...")

# 74개 진정한 Implicit Hate
df_implicit = pd.read_csv('results/p1_analysis/borderline_true_implicit.csv')
print(f"✅ True Implicit: {len(df_implicit)}개")

# 전체 Normal (비교용)
df_normal = pd.read_csv('hatexplain_prediction.csv')
df_normal = df_normal[df_normal['gold_hatexplain_label'] == 1].copy()
print(f"✅ Normal 전체: {len(df_normal)}개")

# 4. 프레이밍 분석 실행
print("\n" + "="*80)
print("🔍 프레이밍 분석 중...")
print("="*80)

for category in FRAMING_LEXICONS.keys():
    print(f"  {category}...")
    df_implicit[category] = df_implicit['text'].apply(
        lambda x: detect_framing(x, category)[0]
    )
    df_implicit[f'{category}_items'] = df_implicit['text'].apply(
        lambda x: detect_framing(x, category)[1]
    )

# 5. 결과 요약
print("\n" + "="*80)
print("📊 프레이밍 분석 결과 (74개 True Implicit Hate)")
print("="*80)

framing_results = {}
for category in FRAMING_LEXICONS.keys():
    count = df_implicit[category].sum()
    pct = count / len(df_implicit) * 100
    framing_results[category] = {
        'count': count,
        'percentage': pct
    }
    source = FRAMING_LEXICONS[category]['source']
    desc = FRAMING_LEXICONS[category]['description']
    print(f"\n{category:25s}: {count:3d}개 ({pct:5.1f}%)")
    print(f"  출처: {source}")
    print(f"  정의: {desc}")

# 프레이밍 없는 샘플
no_framing = df_implicit[
    ~df_implicit[[col for col in df_implicit.columns if col in FRAMING_LEXICONS.keys()]].any(axis=1)
]
print(f"\n{'NO_FRAMING':25s}: {len(no_framing):3d}개 ({len(no_framing)/len(df_implicit)*100:5.1f}%)")

# 6. 복수 프레이밍 분석
print("\n" + "="*80)
print("📊 복수 프레이밍 동시 발생 (Top 5)")
print("="*80)

df_implicit['framing_count'] = df_implicit[
    [col for col in df_implicit.columns if col in FRAMING_LEXICONS.keys()]
].sum(axis=1)

# 프레이밍 조합 찾기
framing_combinations = []
for _, row in df_implicit.iterrows():
    active_framings = [col for col in FRAMING_LEXICONS.keys() if row[col]]
    if len(active_framings) >= 2:
        framing_combinations.append(' + '.join(sorted(active_framings[:2])))

combination_counts = Counter(framing_combinations)
for combo, count in combination_counts.most_common(5):
    print(f"  {combo:50s}: {count:2d}회")

# 7. 샘플 예시
print("\n" + "="*80)
print("📋 프레이밍 샘플 예시")
print("="*80)

for category in ['GENERALIZATION', 'CRIMINAL_DANGER', 'EXCLUSION']:
    samples = df_implicit[df_implicit[category]]
    if len(samples) > 0:
        print(f"\n【{category}】")
        for i, row in samples.head(2).iterrows():
            print(f"  • {row['text'][:100]}")
            if len(row[f'{category}_items']) > 0:
                print(f"    → 키워드: {row[f'{category}_items']}")

# 8. 시각화
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 8-1. 프레이밍 분포 (상위 8개)
sorted_framings = sorted(framing_results.items(), 
                         key=lambda x: x[1]['percentage'], 
                         reverse=True)[:8]

categories = [f[0] for f in sorted_framings]
percentages = [f[1]['percentage'] for f in sorted_framings]

axes[0, 0].barh(categories, percentages, color='#3498db', alpha=0.7)
axes[0, 0].set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Framing Distribution (Top 8)', fontsize=14, fontweight='bold')
axes[0, 0].invert_yaxis()

for i, v in enumerate(percentages):
    axes[0, 0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=10)

# 8-2. 프레이밍 개수 분포
framing_count_dist = df_implicit['framing_count'].value_counts().sort_index()

axes[0, 1].bar(framing_count_dist.index, framing_count_dist.values, 
               color='#e74c3c', alpha=0.7)
axes[0, 1].set_xlabel('Number of Framings', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Sample Count', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Distribution of Framing Count', fontsize=14, fontweight='bold')

# 8-3. 프레이밍 유무 비교 (파이 차트)
framing_presence = {
    'NO_FRAMING': len(no_framing),
    'Has Framing': len(df_implicit) - len(no_framing)
}

colors_pie = ['#95a5a6', '#3498db']
explode = (0.1, 0)  # NO_FRAMING 강조

axes[1, 0].pie(framing_presence.values(), 
               labels=['NO_FRAMING\n(93.2%)', 'Has Framing\n(6.8%)'],
               autopct='%1.1f%%', 
               startangle=90,
               colors=colors_pie,
               explode=explode,
               textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[1, 0].set_title('Framing Presence in True Implicit Normal', 
                     fontsize=14, fontweight='bold')

# 8-4. 문헌 출처별 프레이밍 개수
source_framings = {}
for category, definition in FRAMING_LEXICONS.items():
    source = definition['source'].split('(')[0].strip()  # "ElSherief et al." 추출
    if source not in source_framings:
        source_framings[source] = []
    source_framings[source].append(category)

source_counts = {src: len(cats) for src, cats in source_framings.items()}

axes[1, 1].pie(source_counts.values(), labels=source_counts.keys(), 
               autopct='%1.1f%%', startangle=90,
               colors=['#3498db', '#e74c3c', '#2ecc71'])
axes[1, 1].set_title('Framing Categories by Source', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/p1_analysis/framing_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ 시각화 저장: results/p1_analysis/framing_analysis.png")

# 9. 결과 CSV 저장
# 9-1. 프레이밍별 샘플 저장
framing_cols = ['post_id', 'text', 'token_length', 'framing_count'] + \
               [col for col in df_implicit.columns if col in FRAMING_LEXICONS.keys()]

df_implicit[framing_cols].to_csv(
    'results/p1_analysis/true_implicit_with_framing.csv',
    index=False, encoding='utf-8-sig'
)
print(f"✅ CSV 저장: results/p1_analysis/true_implicit_with_framing.csv")

# 9-2. 프레이밍 요약 통계
framing_summary = []
for category, result in sorted(framing_results.items(), 
                               key=lambda x: x[1]['percentage'], 
                               reverse=True):
    framing_summary.append({
        'Framing': category,
        'Count': result['count'],
        'Percentage': f"{result['percentage']:.1f}%",
        'Source': FRAMING_LEXICONS[category]['source'],
        'Description': FRAMING_LEXICONS[category]['description']
    })

framing_summary.append({
    'Framing': 'NO_FRAMING',
    'Count': len(no_framing),
    'Percentage': f"{len(no_framing)/len(df_implicit)*100:.1f}%",
    'Source': '-',
    'Description': '감지된 프레이밍 없음'
})

summary_df = pd.DataFrame(framing_summary)
summary_df.to_csv('results/p1_analysis/framing_summary.csv',
                  index=False, encoding='utf-8-sig')
print(f"✅ 요약 저장: results/p1_analysis/framing_summary.csv")

# 10. 최종 결론
print("\n" + "="*80)
print("📌 최종 결론")
print("="*80)

top_3_framings = sorted(framing_results.items(), 
                        key=lambda x: x[1]['percentage'], 
                        reverse=True)[:3]

print(f"""
✅ 74개 True Implicit Hate 프레이밍 분석 완료

【상위 3개 프레이밍】
1. {top_3_framings[0][0]}: {top_3_framings[0][1]['count']}개 ({top_3_framings[0][1]['percentage']:.1f}%)
   - {FRAMING_LEXICONS[top_3_framings[0][0]]['description']}
   
2. {top_3_framings[1][0]}: {top_3_framings[1][1]['count']}개 ({top_3_framings[1][1]['percentage']:.1f}%)
   - {FRAMING_LEXICONS[top_3_framings[1][0]]['description']}
   
3. {top_3_framings[2][0]}: {top_3_framings[2][1]['count']}개 ({top_3_framings[2][1]['percentage']:.1f}%)
   - {FRAMING_LEXICONS[top_3_framings[2][0]]['description']}

【프레이밍 없음】
- {len(no_framing)}개 ({len(no_framing)/len(df_implicit)*100:.1f}%)
- 표면 cue도 없고 전형적 프레이밍도 없는 극단적 암묵성

【문헌 근거】
- ElSherief et al. (2021): 암묵적 혐오 분류 체계
- Ocampo et al. (2023): 언어적 속성 18개
- Carvalho et al. (2023): 사회심리학적 전략

💡 다음 단계:
1. 팀원의 "Cue 없는 Hate 1,158개"와 교차 비교
2. 프레이밍 분포 차이 분석
3. 노션 업데이트
""")

print("="*80)
print("✅ P3 프레이밍 분석 완료!")
print("="*80)
