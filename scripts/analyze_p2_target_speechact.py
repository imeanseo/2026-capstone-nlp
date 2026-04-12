"""
P2 분석: Target 그룹 심화 + Speech Act 분석
- P1에서 생성된 데이터를 활용
- 3. Target 그룹 중심 분석 (implicitness, toxicity profile)
- 4-3. 문장 기능별 패턴 (Speech Act)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("P2 분석: Target 심화 + Speech Act")
print("=" * 60)

# P1에서 생성한 데이터들 로드
borderline_strong = pd.read_csv('results/p1_analysis/borderline_strong.csv')
borderline_weak = pd.read_csv('results/p1_analysis/borderline_weak.csv')
borderline_none = pd.read_csv('results/p1_analysis/borderline_none.csv')
borderline_no_hurtlex = pd.read_csv('results/p1_analysis/borderline_no_hurtlex.csv')
cell_c = pd.read_csv('results/p1_analysis/template_cell_c.csv')
cell_d = pd.read_csv('results/p1_analysis/template_cell_d.csv')

# Borderline 통합
borderline_all = pd.concat([borderline_strong, borderline_weak, borderline_none], ignore_index=True)

# 전체 Normal 데이터 로드 (타겟 정보를 위해)
df_full = pd.read_csv('hatexplain_prediction.csv')
df_normal = df_full[df_full['gold_hatexplain_label'] == 1].copy()

print(f"\n전체 Normal 샘플: {len(df_normal):,}개")
print(f"  - Borderline: {len(borderline_all):,}개 ({len(borderline_all)/len(df_normal)*100:.1f}%)")
print(f"  - Perfect Match: {len(cell_d):,}개 ({len(cell_d)/len(df_normal)*100:.1f}%)")
print(f"  - HurtLex 없는 Borderline: {len(borderline_no_hurtlex):,}개\n")

# ============================================================
# Target 파싱
# ============================================================

def parse_targets(target_str):
    """타겟 리스트 파싱"""
    try:
        if pd.isna(target_str):
            return []
        if isinstance(target_str, str):
            target_str = target_str.replace("'", '').replace('[', '').replace(']', '')
            targets = [t.strip() for t in target_str.split() if t.strip()]
            return targets
        return []
    except:
        return []

# 모든 데이터프레임에 target 파싱 적용
for df in [df_normal, borderline_all, cell_d, borderline_no_hurtlex]:
    df['targets_parsed'] = df['targets'].apply(parse_targets) if 'targets' in df.columns else df.apply(lambda x: [], axis=1)
    df['target_count'] = df['targets_parsed'].apply(len)

print("✅ 데이터 파싱 완료\n")

# ============================================================
# 3. Target 그룹 중심 분석
# ============================================================

print("=" * 60)
print("3. Target 그룹 중심 분석")
print("=" * 60)

target_groups = ['African', 'Women', 'Islam', 'Jewish', 'Homosexual', 'Arab', 
                 'Caucasian', 'Refugee', 'Hispanic', 'Asian', 'Men', 'Other']

# 타겟별 통계 수집
target_stats = []

for target in target_groups:
    # 전체 Normal에서 해당 타겟 포함한 샘플
    target_df = df_normal[df_normal['targets_parsed'].apply(lambda x: target in x)]
    n = len(target_df)
    
    if n == 0:
        continue
    
    # token_length 계산
    if 'token_length' not in target_df.columns:
        target_df = target_df.copy()
        target_df['token_length'] = target_df['post_tokens'].apply(lambda x: len(str(x).split()))
    
    mean_len = target_df['token_length'].mean()
    median_len = target_df['token_length'].median()
    
    # Borderline 중 해당 타겟 포함한 샘플 (post_id 기준 매칭)
    target_post_ids = set(target_df['post_id'].values)
    borderline_post_ids = set(borderline_all['post_id'].values)
    borderline_target_ids = target_post_ids & borderline_post_ids
    borderline_count = len(borderline_target_ids)
    borderline_pct = (borderline_count / n) * 100 if n > 0 else 0
    
    # Perfect Match 중 해당 타겟 포함한 샘플
    perfect_post_ids = set(cell_d['post_id'].values)
    perfect_target_ids = target_post_ids & perfect_post_ids
    perfect_count = len(perfect_target_ids)
    perfect_pct = (perfect_count / n) * 100 if n > 0 else 0
    
    # 복수 타겟 비율
    multi_target_pct = (target_df['target_count'] >= 2).sum() / n * 100
    
    target_stats.append({
        'target': target,
        'n': n,
        'mean_len': mean_len,
        'median_len': median_len,
        'perfect_pct': perfect_pct,
        'borderline_pct': borderline_pct,
        'multi_target_pct': multi_target_pct
    })

target_stats_df = pd.DataFrame(target_stats)
target_stats_df = target_stats_df.sort_values('n', ascending=False)

print("\n【타겟별 프로필】")
print(target_stats_df.to_string(index=False))

# 타겟별 Implicitness (판단 난이도)
print("\n【타겟별 판단 난이도 (Implicitness)】")
print("Implicitness = Borderline 비율이 높을수록 판단이 어려움")
print(f"\n{'타겟':<15} {'완전일치%':<10} {'Borderline%':<12} {'평균길이':<10}")
print("-" * 55)

target_stats_sorted = target_stats_df.sort_values('borderline_pct', ascending=False)
for _, row in target_stats_sorted.iterrows():
    print(f"{row['target']:<15} {row['perfect_pct']:>8.1f}%  {row['borderline_pct']:>10.1f}%  {row['mean_len']:>8.1f}")

# ============================================================
# 4-3. Speech Act 분석
# ============================================================

print("\n" + "=" * 60)
print("4-3. Speech Act 분석 (문장 기능별 패턴)")
print("=" * 60)

def classify_speech_act(text):
    """문장의 기능(Speech Act) 분류"""
    if pd.isna(text) or not text:
        return 'UNCLEAR'
    
    text_lower = text.lower()
    tokens_lower = text_lower.split()
    
    # 1. DIRECT_ADDRESS (직접 호칭)
    if 'you' in tokens_lower or 'your' in tokens_lower:
        return 'DIRECT_ADDRESS'
    
    # 2. IMPERATIVE (명령형)
    imperative_verbs = ['go', 'kill', 'deport', 'stop', 'shut', 'get', 'fuck', 'leave', 'take']
    if len(tokens_lower) >= 1:
        first_three = tokens_lower[:min(3, len(tokens_lower))]
        if any(verb in first_three for verb in imperative_verbs):
            return 'IMPERATIVE'
    
    # 3. WISH_DESIRE (소망/당위)
    wish_markers = ['should', 'need to', 'deserve', 'must', 'ought', 'have to', 'better']
    if any(marker in text_lower for marker in wish_markers):
        return 'WISH_DESIRE'
    
    # 4. MOCKERY_SARCASM (조롱)
    mockery_markers = ['lol', 'lmao', 'imagine', 'lmfao', 'haha', 'hahaha']
    if any(marker in tokens_lower for marker in mockery_markers):
        return 'MOCKERY_SARCASM'
    
    # 5. SHORT_LABEL (짧은 낙인) - 6토큰 이하
    if len(tokens_lower) <= 6:
        return 'SHORT_LABEL'
    
    # 6. GROUP_STATEMENT (집단 진술)
    group_markers = ['all', 'every', 'always', 'never', 'most', 'many']
    if any(marker in tokens_lower for marker in group_markers):
        return 'GROUP_STATEMENT'
    
    # 7. QUESTION (질문)
    if '?' in text:
        return 'QUESTION'
    
    # 8. HATE_DECLARATION (혐오 선언)
    hate_markers = ['i hate', 'we hate', 'hate all', 'hate these']
    if any(marker in text_lower for marker in hate_markers):
        return 'HATE_DECLARATION'
    
    # 9. NARRATIVE (서사) - 과거형이 많고 긴 문장
    narrative_markers = ['was', 'were', 'had', 'been', 'did']
    if any(marker in tokens_lower for marker in narrative_markers) and len(tokens_lower) > 15:
        return 'NARRATIVE'
    
    return 'UNCLEAR'

# Speech Act 분류 적용
borderline_all['speech_act'] = borderline_all['text'].apply(classify_speech_act)
cell_d['speech_act'] = cell_d['text'].apply(classify_speech_act)
borderline_no_hurtlex['speech_act'] = borderline_no_hurtlex['text'].apply(classify_speech_act)

# 전체 Borderline 분포
print("\n【Speech Act 분포: Borderline (2/3 일치)】")
speech_act_borderline = borderline_all['speech_act'].value_counts()
for act, count in speech_act_borderline.items():
    pct = count / len(borderline_all) * 100
    print(f"{act:<20} {count:>5,}개 ({pct:>5.1f}%)")

# Perfect Match 분포
print("\n【Speech Act 분포: Perfect Match (3/3 일치)】")
speech_act_perfect = cell_d['speech_act'].value_counts()
for act, count in speech_act_perfect.items():
    pct = count / len(cell_d) * 100
    print(f"{act:<20} {count:>5,}개 ({pct:>5.1f}%)")

# Borderline vs Perfect 비교
print("\n【Speech Act 비교: Borderline vs Perfect Match】")
print(f"{'Speech Act':<20} {'Borderline':<12} {'Perfect':<12} {'차이':<10}")
print("-" * 60)

all_acts = set(speech_act_borderline.index) | set(speech_act_perfect.index)
for act in sorted(all_acts):
    borderline_pct = (borderline_all['speech_act'] == act).sum() / len(borderline_all) * 100
    perfect_pct = (cell_d['speech_act'] == act).sum() / len(cell_d) * 100
    diff = borderline_pct - perfect_pct
    
    print(f"{act:<20} {borderline_pct:>10.1f}%  {perfect_pct:>10.1f}%  {diff:>+8.1f}%p")

# HurtLex 없는 Borderline의 Speech Act
print("\n【Speech Act 분포: HurtLex 없는 Borderline (343개)】")
speech_act_implicit = borderline_no_hurtlex['speech_act'].value_counts()
for act, count in speech_act_implicit.items():
    pct = count / len(borderline_no_hurtlex) * 100
    print(f"{act:<20} {count:>5,}개 ({pct:>5.1f}%)")

# ============================================================
# 시각화
# ============================================================

print("\n" + "=" * 60)
print("시각화 생성 중...")
print("=" * 60)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 타겟별 Borderline 비율 (상위 10개)
ax1 = fig.add_subplot(gs[0, :2])
target_stats_top10 = target_stats_df.head(10).sort_values('borderline_pct', ascending=True)
bars = ax1.barh(target_stats_top10['target'], target_stats_top10['borderline_pct'], color='coral')
ax1.set_xlabel('Borderline Rate (%)', fontsize=11)
ax1.set_title('Borderline Rate by Target Group (Top 10)', fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(target_stats_top10.iterrows()):
    ax1.text(row['borderline_pct'] + 0.5, i, f"{row['borderline_pct']:.1f}%", 
             va='center', fontsize=9)

# 2. 타겟별 샘플 수
ax2 = fig.add_subplot(gs[0, 2])
target_stats_top8 = target_stats_df.head(8).sort_values('n', ascending=True)
ax2.barh(target_stats_top8['target'], target_stats_top8['n'], color='skyblue')
ax2.set_xlabel('Count', fontsize=11)
ax2.set_title('Sample Count by Target', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. Speech Act 분포 (Borderline)
ax3 = fig.add_subplot(gs[1, 0])
speech_act_sorted = speech_act_borderline.sort_values(ascending=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(speech_act_sorted)))
ax3.barh(speech_act_sorted.index, speech_act_sorted.values, color=colors)
ax3.set_xlabel('Count', fontsize=10)
ax3.set_title('Speech Act (Borderline)', fontsize=11, fontweight='bold')
ax3.tick_params(axis='y', labelsize=8)
ax3.grid(axis='x', alpha=0.3)

# 4. Speech Act 분포 (Perfect)
ax4 = fig.add_subplot(gs[1, 1])
speech_act_perfect_sorted = speech_act_perfect.sort_values(ascending=True)
ax4.barh(speech_act_perfect_sorted.index, speech_act_perfect_sorted.values, color=colors)
ax4.set_xlabel('Count', fontsize=10)
ax4.set_title('Speech Act (Perfect Match)', fontsize=11, fontweight='bold')
ax4.tick_params(axis='y', labelsize=8)
ax4.grid(axis='x', alpha=0.3)

# 5. Speech Act 비교 (상위 6개)
ax5 = fig.add_subplot(gs[1, 2])
top_acts = list(speech_act_borderline.head(6).index)
borderline_pcts = [(borderline_all['speech_act'] == act).sum() / len(borderline_all) * 100 for act in top_acts]
perfect_pcts = [(cell_d['speech_act'] == act).sum() / len(cell_d) * 100 for act in top_acts]

x = np.arange(len(top_acts))
width = 0.35

ax5.bar(x - width/2, borderline_pcts, width, label='Borderline', color='coral', alpha=0.8)
ax5.bar(x + width/2, perfect_pcts, width, label='Perfect', color='lightblue', alpha=0.8)

ax5.set_ylabel('Percentage (%)', fontsize=10)
ax5.set_title('Speech Act Comparison', fontsize=11, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(top_acts, rotation=45, ha='right', fontsize=8)
ax5.legend(fontsize=9)
ax5.grid(axis='y', alpha=0.3)

# 6. HurtLex 없는 Borderline의 Speech Act
ax6 = fig.add_subplot(gs[2, :])
speech_act_implicit_sorted = speech_act_implicit.sort_values(ascending=False)
bars6 = ax6.bar(range(len(speech_act_implicit_sorted)), speech_act_implicit_sorted.values, 
                color='mediumpurple', alpha=0.7)
ax6.set_xlabel('Speech Act Type', fontsize=11)
ax6.set_ylabel('Count', fontsize=11)
ax6.set_title('Speech Act Distribution: HurtLex-free Borderline (343 Implicit Hate Candidates)', 
              fontsize=13, fontweight='bold')
ax6.set_xticks(range(len(speech_act_implicit_sorted)))
ax6.set_xticklabels(speech_act_implicit_sorted.index, rotation=45, ha='right', fontsize=10)
ax6.grid(axis='y', alpha=0.3)

for i, (act, count) in enumerate(speech_act_implicit_sorted.items()):
    pct = count / len(borderline_no_hurtlex) * 100
    ax6.text(i, count + 2, f"{count}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9)

plt.savefig('results/p2_analysis/target_speechact_analysis.png', dpi=300, bbox_inches='tight')
print("✅ 저장: results/p2_analysis/target_speechact_analysis.png")

# ============================================================
# CSV 저장
# ============================================================

# 1. 타겟별 통계
target_stats_df.to_csv('results/p2_analysis/target_profile.csv', index=False, encoding='utf-8-sig')
print("✅ 저장: results/p2_analysis/target_profile.csv")

# 2. Speech Act 분포
speech_act_summary = pd.DataFrame({
    'Speech_Act': speech_act_borderline.index,
    'Borderline_Count': speech_act_borderline.values,
    'Borderline_Pct': (speech_act_borderline.values / len(borderline_all) * 100).round(2),
    'Perfect_Count': [speech_act_perfect.get(act, 0) for act in speech_act_borderline.index],
    'Perfect_Pct': [(speech_act_perfect.get(act, 0) / len(cell_d) * 100) for act in speech_act_borderline.index]
})
speech_act_summary.to_csv('results/p2_analysis/speech_act_comparison.csv', index=False, encoding='utf-8-sig')
print("✅ 저장: results/p2_analysis/speech_act_comparison.csv")

# 3. HurtLex 없는 Borderline의 Speech Act
borderline_no_hurtlex[['post_id', 'text', 'speech_act']].to_csv(
    'results/p2_analysis/borderline_no_hurtlex_speechact.csv', index=False, encoding='utf-8-sig')
print("✅ 저장: results/p2_analysis/borderline_no_hurtlex_speechact.csv")

print("\n" + "=" * 60)
print("✅ P2 분석 완료!")
print("=" * 60)
