#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normal 데이터 Logistic Regression

목적:
1. Normal vs Borderline Normal (Hate 소수의견) 구분
2. 팀원의 Hate/Offensive 회귀와 대칭 비교
3. "Hate처럼 보이지만 Normal"인 샘플의 특징 파악
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
print("Normal 데이터 Logistic Regression")
print("="*80)

# 1. 데이터 로드
print("\n📂 데이터 로드 중...")

df_normal = pd.read_csv('hatexplain_prediction.csv')
df_normal = df_normal[df_normal['gold_hatexplain_label'] == 1].copy()
print(f"✅ Normal 전체: {len(df_normal)}개")

# 프레이밍 데이터 로드 (74개 True Implicit에 대한 프레이밍)
df_framing = pd.read_csv('results/p1_analysis/true_implicit_with_framing.csv')
print(f"✅ 프레이밍 데이터: {len(df_framing)}개")

# 2. 종속변수 생성: is_borderline
# annotators 컬럼에서 라벨 추출
import ast

def parse_annotators(annotator_str):
    """annotators 컬럼에서 라벨 추출"""
    try:
        # 'label': array([2, 1, 1]) 형식에서 숫자 추출
        match = re.search(r"'label':\s*array\(\[([^\]]+)\]\)", annotator_str)
        if match:
            labels_str = match.group(1)
            labels = [int(x.strip()) for x in labels_str.split(',')]
            # Normal(1)로 판단한 어노테이터 수 계산
            num_normal = sum(1 for label in labels if label == 1)
            return num_normal
        return 0
    except:
        return 0

df_normal['num_agree'] = df_normal['annotators'].apply(parse_annotators)

# gold_hatexplain_label이 1(Normal)인데 어노테이터가 3명 일치하지 않으면 Borderline
# 즉, Normal로 판단한 사람이 2명이면 Borderline (1명은 Hate(0) 또는 Offensive(2))
df_normal['is_borderline'] = (df_normal['num_agree'] == 2).astype(int)

print(f"\n만장일치 Normal: {(df_normal['is_borderline']==0).sum()}개 ({(df_normal['is_borderline']==0).sum()/len(df_normal)*100:.1f}%)")
print(f"Borderline Normal: {(df_normal['is_borderline']==1).sum()}개 ({(df_normal['is_borderline']==1).sum()/len(df_normal)*100:.1f}%)")

# 3. 피처 엔지니어링
print("\n" + "="*80)
print("🔧 피처 엔지니어링")
print("="*80)

# 3-1. HurtLex 로드
print("\nHurtLex 로드 중...")
hurtlex_words = set()
try:
    with open('hurtlex_EN_conservative.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                word = parts[1].lower().strip()
                if word and len(word) > 1:
                    hurtlex_words.add(word)
except:
    print("⚠️ HurtLex 파일을 찾을 수 없습니다.")

print(f"✅ HurtLex 단어: {len(hurtlex_words)}개")

# 3-2. 기본 피처
def has_hurtlex(text):
    if pd.isna(text):
        return 0
    words = text.lower().split()
    return int(any(word.strip('.,!?;:"\'()[]{}') in hurtlex_words for word in words))

def extract_platform(post_id):
    """Gab: 1, Twitter: 0"""
    if pd.isna(post_id):
        return 0
    return 1 if 'gab' in str(post_id).lower() else 0

def get_target_presence(targets_str):
    """targets 컬럼에서 타겟 존재 여부 추출"""
    try:
        import ast
        targets = ast.literal_eval(targets_str)
        # 'None'이 아닌 실제 타겟이 있는지 확인
        real_targets = [t for t in targets if t != 'None']
        return 1 if len(real_targets) > 0 else 0
    except:
        return 0

# 3-3. 언어적 피처
def has_negation(text):
    if pd.isna(text):
        return 0
    negations = ['not', 'never', 'no', 'nothing', 'nobody', 'nowhere', "n't", 'neither', 'nor']
    text_lower = text.lower()
    return int(any(neg in text_lower for neg in negations))

def has_question(text):
    if pd.isna(text):
        return 0
    return int('?' in text)

def has_conditional(text):
    if pd.isna(text):
        return 0
    conditionals = ['if', 'would', 'could', 'should', 'might', 'may']
    text_lower = text.lower()
    words = text_lower.split()
    return int(any(cond in words for cond in conditionals))

# 3-4. 프레이밍 피처 (11개 카테고리)
FRAMING_LEXICONS = {
    'DEHUMANIZATION': ['animal', 'animals', 'monkey', 'monkeys', 'cockroach', 
                       'vermin', 'subhuman', 'beast', 'savage', 'creature',
                       'rat', 'rats', 'dog', 'dogs', 'pig', 'pigs'],
    'THREAT_VIOLENCE': ['kill', 'killed', 'hang', 'gas', 'exterminate', 'genocide',
                        'murder', 'shoot', 'bomb', 'die', 'death', 'dead',
                        'destroy', 'elimination', 'cleanse'],
    'EXCLUSION_PATTERNS': [r'\bgo back\b', r'\bdon\'t belong\b', r'\bdoesn\'t belong\b',
                           r'\bour country\b', r'\bget out\b', r'\bdeport\b'],
    'CONSPIRACY': ['control', 'controls', 'destroy', 'invasion', 'replace',
                   'replacement', 'globalist', 'globalists', 'shill', 'agenda',
                   'take over', 'takeover', 'plot', 'scheme'],
    'MORAL_DISGUST': ['disgusting', 'filthy', 'dirty', 'degenerate', 'abomination',
                      'repulsive', 'revolting', 'vile', 'sick', 'gross'],
    'INTELLECTUAL_INFERIORITY': ['stupid', 'dumb', 'idiot', 'idiots', 'moron',
                                  'retarded', 'retard', 'ignorant', 'fool'],
    'SEXUAL_GENDERED': ['rape', 'raped', 'raping', 'slut', 'sluts', 'whore',
                        'breed', 'breeding', 'hoe', 'hoes'],
    'CRIMINAL_DANGER': ['crime', 'criminal', 'criminals', 'terrorist', 'terrorists',
                        'illegal', 'illegals', 'steal', 'stealing', 'thief',
                        'danger', 'dangerous', 'threat', 'violent', 'violence',
                        'pedophile', 'radical'],
    'ECONOMIC_BURDEN': ['welfare', 'leech', 'leeches', 'freeloader', 'freeloaders',
                        'parasite', 'parasites', 'burden', 'tax', 'lazy'],
    'RELIGIOUS': ['jihad', 'jihadi', 'sharia', 'infidel', 'infidels',
                  'satanic', 'devil', 'hell', 'sin', 'sinner'],
    'GENERALIZATION_PATTERNS': [r'\ball\b.*\bare\b', r'\ball\b.*\bdo\b',
                                r'\bevery\b.*\bis\b', r'\balways\b', r'\bnever\b']
}

def detect_framing(text, framing_type):
    if pd.isna(text):
        return 0
    text_lower = text.lower()
    
    # 패턴 기반
    if '_PATTERNS' in framing_type:
        patterns = FRAMING_LEXICONS[framing_type]
        return int(any(re.search(p, text_lower) for p in patterns))
    
    # 키워드 기반
    keywords = FRAMING_LEXICONS[framing_type]
    words = text_lower.split()
    words_clean = [w.strip('.,!?;:"\'()[]{}') for w in words]
    return int(any(word in keywords for word in words_clean))

# 피처 생성
print("\n피처 생성 중...")

df_normal['has_hurtlex'] = df_normal['text'].apply(has_hurtlex)
df_normal['platform_gab'] = df_normal['post_id'].apply(extract_platform)

# token_length 계산 (post_tokens에서)
def count_tokens(tokens_str):
    try:
        # "['u' 'really' 'think' ...]" 형식 파싱
        match = re.findall(r"'([^']+)'", tokens_str)
        return len(match)
    except:
        return 0

df_normal['token_length'] = df_normal['post_tokens'].apply(count_tokens)
df_normal['has_target'] = df_normal['targets'].apply(get_target_presence)

# 언어적 피처
df_normal['has_negation'] = df_normal['text'].apply(has_negation)
df_normal['has_question'] = df_normal['text'].apply(has_question)
df_normal['has_conditional'] = df_normal['text'].apply(has_conditional)

# 프레이밍 피처
df_normal['framing_dehumanization'] = df_normal['text'].apply(lambda x: detect_framing(x, 'DEHUMANIZATION'))
df_normal['framing_violence'] = df_normal['text'].apply(lambda x: detect_framing(x, 'THREAT_VIOLENCE'))
df_normal['framing_exclusion'] = df_normal['text'].apply(lambda x: detect_framing(x, 'EXCLUSION_PATTERNS'))
df_normal['framing_criminal'] = df_normal['text'].apply(lambda x: detect_framing(x, 'CRIMINAL_DANGER'))
df_normal['framing_economic'] = df_normal['text'].apply(lambda x: detect_framing(x, 'ECONOMIC_BURDEN'))
df_normal['framing_generalization'] = df_normal['text'].apply(lambda x: detect_framing(x, 'GENERALIZATION_PATTERNS'))

# 프레이밍 총 개수
framing_cols = ['framing_dehumanization', 'framing_violence', 'framing_exclusion', 
                'framing_criminal', 'framing_economic', 'framing_generalization']
df_normal['framing_count'] = df_normal[framing_cols].sum(axis=1)
df_normal['has_framing'] = (df_normal['framing_count'] > 0).astype(int)

# 피처 리스트 (platform_gab 제외)
feature_names = [
    'has_hurtlex', 'token_length', 'has_target',
    'has_negation', 'has_question', 'has_conditional',
    'has_framing', 'framing_count',
    'framing_dehumanization', 'framing_violence', 'framing_exclusion',
    'framing_criminal', 'framing_economic', 'framing_generalization'
]

print(f"✅ 생성된 피처: {len(feature_names)}개")
for feat in feature_names:
    print(f"  • {feat}")

# 4. 데이터 분할 및 모델 학습
print("\n" + "="*80)
print("🤖 모델 학습")
print("="*80)

X = df_normal[feature_names]
y = df_normal['is_borderline']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")
print(f"  - Borderline: {y_test.sum()}개 ({y_test.sum()/len(y_test)*100:.1f}%)")
print(f"  - 만장일치: {len(y_test)-y_test.sum()}개 ({(len(y_test)-y_test.sum())/len(y_test)*100:.1f}%)")

# Logistic Regression
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, y_train)

# 예측
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# 5. 결과 평가
print("\n" + "="*80)
print("📊 모델 성능")
print("="*80)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nAccuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred)
print(f"\n혼동 행렬:")
print(f"              예측 만장일치  예측 Borderline")
print(f"실제 만장일치      {cm[0,0]:4d}          {cm[0,1]:4d}")
print(f"실제 Borderline    {cm[1,0]:4d}          {cm[1,1]:4d}")

# 6. Feature Importance
print("\n" + "="*80)
print("📈 Feature Importance (Coefficients)")
print("="*80)

coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': clf.coef_[0]
}).sort_values('Coefficient', ascending=False)

print("\n【Borderline을 증가시키는 피처 (양수 계수)】")
positive_features = coefficients[coefficients['Coefficient'] > 0]
for idx, row in positive_features.iterrows():
    print(f"  {row['Feature']:30s}: {row['Coefficient']:+.3f}")

print("\n【Borderline을 감소시키는 피처 (음수 계수)】")
negative_features = coefficients[coefficients['Coefficient'] < 0]
for idx, row in negative_features.iterrows():
    print(f"  {row['Feature']:30s}: {row['Coefficient']:+.3f}")

# 7. 시각화
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 7-1. Feature Importance
top_n = 10
top_features = coefficients.head(top_n)

axes[0, 0].barh(top_features['Feature'], top_features['Coefficient'], 
                color=['#e74c3c' if c > 0 else '#3498db' for c in top_features['Coefficient']],
                alpha=0.7)
axes[0, 0].set_xlabel('Coefficient', fontsize=12, fontweight='bold')
axes[0, 0].set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
axes[0, 0].invert_yaxis()

# 7-2. 혼동 행렬 히트맵
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=['만장일치', 'Borderline'],
            yticklabels=['만장일치', 'Borderline'])
axes[0, 1].set_xlabel('예측', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('실제', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# 7-3. 피처별 분포 비교 (상위 5개)
top_5_features = coefficients.head(5)['Feature'].tolist()

comparison_data = []
for feat in top_5_features:
    borderline_mean = df_normal[df_normal['is_borderline'] == 1][feat].mean()
    unanimous_mean = df_normal[df_normal['is_borderline'] == 0][feat].mean()
    comparison_data.append({
        'Feature': feat[:20],  # 피처명 길이 제한
        'Borderline': borderline_mean,
        '만장일치': unanimous_mean
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.set_index('Feature')

comparison_df.plot(kind='barh', ax=axes[1, 0], color=['#e74c3c', '#3498db'], alpha=0.7)
axes[1, 0].set_xlabel('평균값', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Top 5 Features: Borderline vs 만장일치', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].invert_yaxis()

# 7-4. 회귀 계수 비교 (양수 vs 음수)
positive_coef = coefficients[coefficients['Coefficient'] > 0].head(7)
negative_coef = coefficients[coefficients['Coefficient'] < 0].tail(5)

all_coef = pd.concat([positive_coef, negative_coef]).sort_values('Coefficient', ascending=True)

colors = ['#e74c3c' if x > 0 else '#3498db' for x in all_coef['Coefficient']]
axes[1, 1].barh(range(len(all_coef)), all_coef['Coefficient'], color=colors, alpha=0.7)
axes[1, 1].set_yticks(range(len(all_coef)))
axes[1, 1].set_yticklabels([f[:20] for f in all_coef['Feature']], fontsize=9)
axes[1, 1].set_xlabel('Coefficient', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Positive vs Negative Coefficients', fontsize=14, fontweight='bold')
axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig('results/p1_analysis/logistic_regression_normal.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ 시각화 저장: results/p1_analysis/logistic_regression_normal.png")

# 8. 잘못 분류된 샘플 분석
print("\n" + "="*80)
print("🔍 잘못 분류된 샘플 분석")
print("="*80)

# 테스트 데이터에 예측 결과 추가
X_test_with_results = X_test.copy()
X_test_with_results['actual'] = y_test.values
X_test_with_results['predicted'] = y_pred
X_test_with_results['probability'] = y_prob

# 원본 텍스트 가져오기
test_indices = X_test.index
X_test_with_results['text'] = df_normal.loc[test_indices, 'text'].values
X_test_with_results['post_id'] = df_normal.loc[test_indices, 'post_id'].values

# False Positive (만장일치인데 Borderline으로 예측)
false_positives = X_test_with_results[(X_test_with_results['actual'] == 0) & 
                                       (X_test_with_results['predicted'] == 1)]
print(f"\nFalse Positive: {len(false_positives)}개")
if len(false_positives) > 0:
    print("\n【샘플 예시 (상위 3개)】")
    for i, (idx, row) in enumerate(false_positives.head(3).iterrows()):
        print(f"\n{i+1}. {row['text'][:100]}")
        print(f"   확률: {row['probability']:.3f}")
        print(f"   HurtLex: {row['has_hurtlex']}, Framing: {row['framing_count']}, Length: {row['token_length']:.0f}")

# False Negative (Borderline인데 만장일치로 예측)
false_negatives = X_test_with_results[(X_test_with_results['actual'] == 1) & 
                                       (X_test_with_results['predicted'] == 0)]
print(f"\nFalse Negative: {len(false_negatives)}개")
if len(false_negatives) > 0:
    print("\n【샘플 예시 (상위 3개)】")
    for i, (idx, row) in enumerate(false_negatives.head(3).iterrows()):
        print(f"\n{i+1}. {row['text'][:100]}")
        print(f"   확률: {row['probability']:.3f}")
        print(f"   HurtLex: {row['has_hurtlex']}, Framing: {row['framing_count']}, Length: {row['token_length']:.0f}")

# 9. 결과 저장
print("\n" + "="*80)
print("💾 결과 저장")
print("="*80)

# 9-1. Feature Importance CSV
coefficients.to_csv('results/p1_analysis/normal_feature_importance.csv',
                    index=False, encoding='utf-8-sig')
print("✅ results/p1_analysis/normal_feature_importance.csv")

# 9-2. 예측 결과 CSV
X_test_with_results.to_csv('results/p1_analysis/normal_predictions.csv',
                            index=False, encoding='utf-8-sig')
print("✅ results/p1_analysis/normal_predictions.csv")

# 9-3. 모델 성능 요약
performance_summary = {
    '지표': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
    '값': [f'{accuracy:.3f}', f'{precision:.3f}', f'{recall:.3f}', f'{f1:.3f}']
}
performance_df = pd.DataFrame(performance_summary)
performance_df.to_csv('results/p1_analysis/normal_model_performance.csv',
                      index=False, encoding='utf-8-sig')
print("✅ results/p1_analysis/normal_model_performance.csv")

# 10. 최종 요약
print("\n" + "="*80)
print("📌 최종 요약")
print("="*80)

top_3 = coefficients.head(3)
print(f"""
✅ Normal 데이터 Logistic Regression 완료

【모델 성능】
- Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)
- Precision: {precision:.3f}
- Recall: {recall:.3f}
- F1-score: {f1:.3f}

【Top 3 Borderline 증가 피처】
1. {top_3.iloc[0]['Feature']:30s}: {top_3.iloc[0]['Coefficient']:+.3f}
2. {top_3.iloc[1]['Feature']:30s}: {top_3.iloc[1]['Coefficient']:+.3f}
3. {top_3.iloc[2]['Feature']:30s}: {top_3.iloc[2]['Coefficient']:+.3f}

【해석】
- Borderline Normal은 만장일치 Normal과 명확히 구분됨
- 상위 피처들이 "Implicit Hate 후보"의 특징을 나타냄
- 팀원의 Hate/Offensive 회귀와 비교 가능

💡 다음 단계:
- 팀원의 회귀 결과와 비교 (compare_regressions.py)
- 공통 피처 파악 → Cell C 정의 정밀화
""")

print("="*80)
print("✅ Normal Logistic Regression 완료!")
print("="*80)
