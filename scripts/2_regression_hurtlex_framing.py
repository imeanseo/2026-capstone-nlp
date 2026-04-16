#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hurtlex 카테고리 기반 Normal 데이터 Logistic Regression

목적:
1. 기존 임의 11개 프레이밍 대신 Hurtlex 17개 카테고리 사용
2. Normal vs Borderline Normal 구분
3. 기존 모델과 성능 비교

Hurtlex 17 카테고리:
- cds: Cognitive Disabilities
- an: Animal Names
- dmc: Mental Disabilities
- re: Negative Moral
- svp: Sexual Violence
- qas: LGBTQ Slurs
- asm: Racial Slurs
- ps: Physical/Mental Diseases
- om: Crimes/Immoral
- pr: Prostitution
- pa: Physical Disabilities
- or: Plants/Food
- asf: Anti-Women
- is: Low Socio-Economic
- ddf: Anti-Men
- rci: Religious
- ddp: Body Parts
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
import platform
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("Hurtlex 카테고리 기반 Normal 데이터 Logistic Regression")
print("="*80)

# ========================================================================
# 1. Hurtlex 사전 로드 및 카테고리별 단어 딕셔너리 생성
# ========================================================================
print("\n📂 [Step 1] Hurtlex 사전 로드...")

hurtlex_df = pd.read_csv('lexicons/hurtlex_EN.tsv', sep='\t')
print(f"✅ Hurtlex 로드: {len(hurtlex_df):,}개 단어")

# 카테고리 설명
CATEGORY_DESCRIPTIONS = {
    'cds': 'Cognitive Disabilities',
    'an': 'Animal Names',
    'dmc': 'Mental Disabilities',
    're': 'Negative Moral',
    'svp': 'Sexual Violence',
    'qas': 'LGBTQ Slurs',
    'asm': 'Racial Slurs',
    'ps': 'Physical/Mental Diseases',
    'om': 'Crimes/Immoral',
    'pr': 'Prostitution',
    'pa': 'Physical Disabilities',
    'or': 'Plants/Food',
    'asf': 'Anti-Women',
    'is': 'Low Socio-Economic',
    'ddf': 'Anti-Men',
    'rci': 'Religious',
    'ddp': 'Body Parts'
}

# 카테고리별 단어 딕셔너리 생성 (lemma 기반)
hurtlex_dict = defaultdict(set)
for _, row in hurtlex_df.iterrows():
    category = row['category']
    lemma = str(row['lemma']).lower().strip()
    if lemma and lemma != 'nan':
        hurtlex_dict[category].add(lemma)

print(f"✅ {len(hurtlex_dict)}개 카테고리 단어 딕셔너리 생성 완료")

# ========================================================================
# 2. 데이터 로드
# ========================================================================
print("\n📂 [Step 2] 데이터 로드...")

df_normal = pd.read_csv('hatexplain_prediction.csv')
df_normal = df_normal[df_normal['gold_hatexplain_label'] == 1].copy()
print(f"✅ Normal 전체: {len(df_normal):,}개")

# ========================================================================
# 3. 종속변수 생성: is_borderline
# ========================================================================
print("\n🏷️ [Step 3] 레이블 생성...")

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

print(f"✅ 레이블 생성 완료")
print(f"  만장일치 Normal: {(df_normal['is_borderline']==0).sum():,}개 ({(df_normal['is_borderline']==0).sum()/len(df_normal)*100:.1f}%)")
print(f"  Borderline Normal: {(df_normal['is_borderline']==1).sum():,}개 ({(df_normal['is_borderline']==1).sum()/len(df_normal)*100:.1f}%)")

# ========================================================================
# 4. 피처 엔지니어링: Hurtlex 17개 카테고리
# ========================================================================
print("\n🔧 [Step 4] Hurtlex 카테고리 피처 생성...")

# 각 카테고리별 매칭
for category in sorted(hurtlex_dict.keys()):
    print(f"  {category}...", end=' ')
    df_normal[f'hurtlex_{category}'] = df_normal['text'].apply(
        lambda x: int(any(word.strip('.,!?;:"\'()[]{}') in hurtlex_dict[category] 
                         for word in str(x).lower().split()))
    )
    count = df_normal[f'hurtlex_{category}'].sum()
    print(f"{count:,}개")

# 전체 Hurtlex 포함 여부
df_normal['has_hurtlex'] = df_normal[[f'hurtlex_{cat}' for cat in hurtlex_dict.keys()]].max(axis=1)

# 카테고리 개수
hurtlex_cols = [f'hurtlex_{cat}' for cat in sorted(hurtlex_dict.keys())]
df_normal['hurtlex_category_count'] = df_normal[hurtlex_cols].sum(axis=1)

print(f"\n✅ Hurtlex 피처 생성 완료")
print(f"  Hurtlex 포함: {df_normal['has_hurtlex'].sum():,}개 ({df_normal['has_hurtlex'].mean()*100:.2f}%)")
print(f"  평균 카테고리 수: {df_normal['hurtlex_category_count'].mean():.2f}개")

# ========================================================================
# 5. 추가 피처 생성 (플랫폼 구분 제외)
# ========================================================================
print("\n🔧 [Step 5] 추가 피처 생성...")

# Token length
def count_tokens(tokens_str):
    try:
        match = re.findall(r"'([^']+)'", tokens_str)
        return len(match)
    except:
        return 0

df_normal['token_length'] = df_normal['post_tokens'].apply(count_tokens)

# Target presence
def get_target_presence(targets_str):
    try:
        import ast
        targets = ast.literal_eval(targets_str)
        # 'None'이 아닌 실제 타겟이 있는지 확인
        real_targets = [t for t in targets if t != 'None']
        return 1 if len(real_targets) > 0 else 0
    except:
        return 0

df_normal['has_target'] = df_normal['targets'].apply(get_target_presence)

print(f"✅ 추가 피처 생성 완료 (플랫폼 변수 제외)")
print(f"  평균 토큰 길이: {df_normal['token_length'].mean():.2f}")
print(f"  Target 포함: {df_normal['has_target'].sum():,}개")

# ========================================================================
# 6. 회귀 모델: Hurtlex 카테고리 기반 (플랫폼 변수 제외)
# ========================================================================
print("\n📈 [Step 6] 회귀 모델 학습 (플랫폼 구분 없이)...")

# 피처 선택: Hurtlex 17개 카테고리 + 기타 피처 (platform_gab 제외)
feature_names = hurtlex_cols + ['token_length', 'has_target', 'hurtlex_category_count']

X = df_normal[feature_names]
y = df_normal['is_borderline']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n학습 데이터: {len(X_train):,}개")
print(f"테스트 데이터: {len(X_test):,}개")
print(f"  - Borderline: {y_test.sum()}개 ({y_test.sum()/len(y_test)*100:.1f}%)")
print(f"  - 만장일치: {len(y_test)-y_test.sum()}개 ({(len(y_test)-y_test.sum())/len(y_test)*100:.1f}%)")

# 모델 학습
clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
clf.fit(X_train, y_train)

# 예측
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# ========================================================================
# 7. 모델 평가
# ========================================================================
print("\n📊 [Step 7] 모델 성능 평가...")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n✅ Hurtlex 모델 성능:")
print(f"  Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f}")
print(f"  F1-score:  {f1:.3f}")
print(f"  AUC:       {auc:.3f}")

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred)
print(f"\n혼동 행렬:")
print(f"              예측 만장일치  예측 Borderline")
print(f"실제 만장일치      {cm[0,0]:4d}          {cm[0,1]:4d}")
print(f"실제 Borderline    {cm[1,0]:4d}          {cm[1,1]:4d}")

# ========================================================================
# 8. Feature Importance
# ========================================================================
print("\n📈 [Step 8] Feature Importance 분석...")

coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': clf.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n【상위 10개 중요 피처】")
for i, row in coefficients.head(10).iterrows():
    feat = row['Feature'].replace('hurtlex_', '')
    desc = CATEGORY_DESCRIPTIONS.get(feat, feat)
    direction = "→ Borderline 증가" if row['Coefficient'] > 0 else "→ 만장일치 증가"
    print(f"  {feat:15s} ({desc:25s}): {row['Coefficient']:+.4f} {direction}")

# ========================================================================
# 9. 시각화
# ========================================================================
print("\n📊 [Step 9] 시각화...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 9-1. Top 15 Feature Importance
ax1 = fig.add_subplot(gs[0, :])
top_15_features = coefficients.head(15).copy()
top_15_features['feature_label'] = top_15_features['Feature'].apply(
    lambda x: f"{x.replace('hurtlex_', '')[:10]}\n({CATEGORY_DESCRIPTIONS.get(x.replace('hurtlex_', ''), x)[:20]})" 
    if x.startswith('hurtlex_') else x
)

colors = ['#e74c3c' if coef > 0 else '#3498db' for coef in top_15_features['Coefficient']]
ax1.barh(range(len(top_15_features)), top_15_features['Coefficient'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(top_15_features)))
ax1.set_yticklabels(top_15_features['feature_label'], fontsize=9)
ax1.set_xlabel('Coefficient', fontsize=12, fontweight='bold')
ax1.set_title('Top 15 Feature Importance (Hurtlex Model)\n(Red = Borderline 증가, Blue = 만장일치 증가)', 
              fontsize=14, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax1.invert_yaxis()

# 9-2. Confusion Matrix
ax2 = fig.add_subplot(gs[1, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['만장일치', 'Borderline'],
            yticklabels=['만장일치', 'Borderline'])
ax2.set_ylabel('실제', fontsize=11, fontweight='bold')
ax2.set_xlabel('예측', fontsize=11, fontweight='bold')
ax2.set_title(f'Confusion Matrix\n(AUC: {auc:.3f})', fontsize=12, fontweight='bold')

# 9-3. Borderline vs 만장일치 - Hurtlex 커버리지
ax3 = fig.add_subplot(gs[1, 1])
border_hl_pct = df_normal[df_normal['is_borderline']==1]['has_hurtlex'].mean() * 100
unanimous_hl_pct = df_normal[df_normal['is_borderline']==0]['has_hurtlex'].mean() * 100

ax3.bar(['Borderline', '만장일치'], [border_hl_pct, unanimous_hl_pct], 
        color=['#e74c3c', '#3498db'], alpha=0.7)
ax3.set_ylabel('Hurtlex Coverage (%)', fontsize=11, fontweight='bold')
ax3.set_title('Hurtlex Coverage\nby Agreement Type', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 100])

for i, v in enumerate([border_hl_pct, unanimous_hl_pct]):
    ax3.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax4 = fig.add_subplot(gs[1, 2])
cat_count_borderline = df_normal.groupby('hurtlex_category_count')['is_borderline'].mean() * 100
cat_count_sizes = df_normal.groupby('hurtlex_category_count').size()

# 샘플 수가 충분한 경우만 표시 (10개 이상)
valid_counts = cat_count_sizes[cat_count_sizes >= 10].index
cat_count_borderline_valid = cat_count_borderline[valid_counts]

ax4.plot(valid_counts, cat_count_borderline_valid, marker='o', linewidth=2, 
         markersize=8, color='#e74c3c')
ax4.set_xlabel('Number of Categories', fontsize=11, fontweight='bold')
ax4.set_ylabel('Borderline Rate (%)', fontsize=11, fontweight='bold')
ax4.set_title('Borderline Rate by\nCategory Count', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 9-5. 상위 카테고리 분포
ax5 = fig.add_subplot(gs[2, :])
category_results = {}
for category in hurtlex_dict.keys():
    count = df_normal[f'hurtlex_{category}'].sum()
    pct = count / len(df_normal) * 100
    category_results[category] = {'count': count, 'percentage': pct}

top_10 = sorted(category_results.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
top_10_cats = [cat for cat, _ in top_10]
top_10_counts = [category_results[cat]['count'] for cat in top_10_cats]
top_10_labels = [f"{cat}\n({CATEGORY_DESCRIPTIONS.get(cat, cat)[:15]})" for cat in top_10_cats]

ax5.barh(range(len(top_10_cats)), top_10_counts, color='#3498db', alpha=0.7)
ax5.set_yticks(range(len(top_10_cats)))
ax5.set_yticklabels(top_10_labels, fontsize=9)
ax5.set_xlabel('Sample Count', fontsize=12, fontweight='bold')
ax5.set_title('Top 10 Hurtlex Categories in Normal Samples', fontsize=14, fontweight='bold')
ax5.invert_yaxis()

for i, v in enumerate(top_10_counts):
    ax5.text(v + 30, i, f'{v:,}', va='center', fontsize=10)

plt.savefig('results/p1_analysis/logistic_regression_hurtlex.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ 시각화 저장: results/p1_analysis/logistic_regression_hurtlex.png")

# ========================================================================
# 10. 결과 저장
# ========================================================================
print("\n💾 [Step 10] 결과 저장...")

# 10-1. Feature Importance
coefficients.to_csv('results/p1_analysis/hurtlex_feature_importance.csv',
                    index=False, encoding='utf-8-sig')
print("✅ results/p1_analysis/hurtlex_feature_importance.csv")

# 10-2. 예측 결과
X_test_with_results = X_test.copy()
X_test_with_results['actual'] = y_test.values
X_test_with_results['predicted'] = y_pred
X_test_with_results['probability'] = y_prob
test_indices = X_test.index
X_test_with_results['text'] = df_normal.loc[test_indices, 'text'].values
X_test_with_results['post_id'] = df_normal.loc[test_indices, 'post_id'].values

X_test_with_results.to_csv('results/p1_analysis/hurtlex_predictions.csv',
                            index=False, encoding='utf-8-sig')
print("✅ results/p1_analysis/hurtlex_predictions.csv")

# 10-3. 모델 성능 요약
performance_summary = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC'],
    'Value': [f'{accuracy:.3f}', f'{precision:.3f}', f'{recall:.3f}', f'{f1:.3f}', f'{auc:.3f}']
}
performance_df = pd.DataFrame(performance_summary)
performance_df.to_csv('results/p1_analysis/hurtlex_model_performance.csv',
                      index=False, encoding='utf-8-sig')
print("✅ results/p1_analysis/hurtlex_model_performance.csv")

# 10-4. 카테고리별 통계
category_stats = []
for category in sorted(hurtlex_dict.keys()):
    result = category_results.get(category, {'count': 0, 'percentage': 0})
    borderline_rate = df_normal[df_normal[f'hurtlex_{category}']==1]['is_borderline'].mean() * 100 if df_normal[f'hurtlex_{category}'].sum() > 0 else 0
    category_stats.append({
        'category': category,
        'description': CATEGORY_DESCRIPTIONS.get(category, category),
        'count': result['count'],
        'percentage': f"{result['percentage']:.2f}%",
        'borderline_rate': f"{borderline_rate:.2f}%"
    })

stats_df = pd.DataFrame(category_stats).sort_values('count', ascending=False)
stats_df.to_csv('results/p1_analysis/hurtlex_category_stats.csv',
                index=False, encoding='utf-8-sig')
print("✅ results/p1_analysis/hurtlex_category_stats.csv")

# ========================================================================
# 11. 기존 모델과 비교
# ========================================================================
print("\n" + "="*80)
print("📊 [Step 11] 기존 모델과 비교")
print("="*80)

try:
    # 기존 모델 성능 로드
    old_performance = pd.read_csv('results/p1_analysis/normal_model_performance.csv')
    old_metrics = dict(zip(old_performance['지표'], old_performance['값']))
    
    print("\n【모델 성능 비교】")
    print(f"{'Metric':<15s} {'기존 (11개 프레이밍)':<20s} {'Hurtlex (17개 카테고리)':<20s} {'차이':<10s}")
    print("-" * 70)
    
    metrics_map = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }
    
    for metric_name, new_value in metrics_map.items():
        old_value = float(old_metrics.get(metric_name, '0'))
        diff = new_value - old_value
        diff_str = f"{diff:+.3f}" if diff != 0 else "0.000"
        print(f"{metric_name:<15s} {old_value:<20.3f} {new_value:<20.3f} {diff_str:<10s}")
    
    print(f"{'AUC':<15s} {'N/A':<20s} {auc:<20.3f} {'NEW':<10s}")
    
except FileNotFoundError:
    print("⚠️ 기존 모델 결과를 찾을 수 없습니다.")

# ========================================================================
# 12. 최종 요약
# ========================================================================
print("\n" + "="*80)
print("📌 최종 요약 - Hurtlex 카테고리 기반 회귀 분석")
print("="*80)

top_3 = coefficients.head(3)
print(f"""
✅ Hurtlex 카테고리 기반 Logistic Regression 완료

【데이터】
- Normal 샘플: {len(df_normal):,}개
  • Borderline: {df_normal['is_borderline'].sum():,}개
  • 만장일치: {(~df_normal['is_borderline'].astype(bool)).sum():,}개

【Hurtlex 프레이밍】
- Hurtlex 포함: {df_normal['has_hurtlex'].sum():,}개 ({df_normal['has_hurtlex'].mean()*100:.2f}%)
- 평균 카테고리 수: {df_normal['hurtlex_category_count'].mean():.2f}개
- Borderline Hurtlex 커버리지: {border_hl_pct:.1f}%
- 만장일치 Hurtlex 커버리지: {unanimous_hl_pct:.1f}%

【모델 성능】
- Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)
- Precision: {precision:.3f}
- Recall:    {recall:.3f}
- F1-score:  {f1:.3f}
- AUC:       {auc:.3f}

【Top 3 중요 피처】
""")

for i, (feat, coef) in enumerate(top_3[['Feature', 'Coefficient']].values, 1):
    if feat.startswith('hurtlex_'):
        cat = feat.replace('hurtlex_', '')
        desc = CATEGORY_DESCRIPTIONS.get(cat, cat)
        direction = "Borderline 증가" if coef > 0 else "만장일치 증가"
        print(f"  {i}. {cat:10s} ({desc:25s}): {coef:+.4f} → {direction}")
    else:
        direction = "Borderline 증가" if coef > 0 else "만장일치 증가"
        print(f"  {i}. {feat:10s}: {coef:+.4f} → {direction}")

print(f"""
【기존 임의 프레이밍 vs Hurtlex 카테고리】

✓ 기존 방식 (11개 임의 프레이밍):
  - 장점: 특정 프레이밍 전략에 집중 (음모론, 배제, 비인간화 등)
  - 장점: 문헌 근거 명확
  - 단점: 커버리지 낮음
  - 피처: 6개 주요 프레이밍 (has_framing, framing_count 등)

✓ Hurtlex 방식 (17개 표준 카테고리):
  - 장점: 높은 커버리지 ({df_normal['has_hurtlex'].mean()*100:.1f}%)
  - 장점: 표준화된 분류 체계 (8,000+ 단어)
  - 장점: 세분화된 혐오 유형 분석 가능
  - 피처: 17개 카테고리 + 메타 피처

💡 권장사항:
1. Hurtlex는 표면적 cue 감지에 강점
2. 기존 프레이밍은 전략적 분석에 강점
3. 두 접근법을 함께 사용하는 것이 최적:
   - Surface-level detection: Hurtlex
   - Strategic framing analysis: 기존 11개 프레이밍
4. 모델 성능 비교 후 앙상블 가능성 검토
""")

print("="*80)
print("✅ Hurtlex 카테고리 기반 회귀 분석 완료!")
print("="*80)
print("\n생성된 파일:")
print("  1. results/p1_analysis/logistic_regression_hurtlex.png")
print("  2. results/p1_analysis/hurtlex_feature_importance.csv")
print("  3. results/p1_analysis/hurtlex_predictions.csv")
print("  4. results/p1_analysis/hurtlex_model_performance.csv")
print("  5. results/p1_analysis/hurtlex_category_stats.csv")
