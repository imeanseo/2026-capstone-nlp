# A-B / A-C steering으로 implicit hate 탐지 성능 향상

날짜: ___

---

## 0. 한 줄 요약

> **두 minimal-pair steering vector(v_AB target / v_AC cue)가 두 평가셋(Latent Hatred, ToxiGen)에서 B0를 일관되게 이긴다.** 다만 (1) 두 벡터의 효과 크기가 거의 같고(표상 prior d=1.13 vs 0.30의 4배 비대칭이 성능에는 재현 안 됨), (2) ToxiGen에서는 단순 v_harm baseline이 우리 두 벡터를 살짝 앞선다. 핵심 성과는 mid-layer(L13–14) 일관 최적과 평균 32% 수준의 FN recovery.
> 

---

## 1. 핵심 수치 — Best 셋업
### Latent Hatred (B0 macro_F1 = 0.5882, hate_recall = 0.545)

| Setup | layer | α | macro_F1 | hate_recall | FN_recovery | ΔF1 |
| --- | --- | --- | --- | --- | --- | --- |
| B0 No steering | — | — | 0.5882 | 0.545 | — | — |
| B1 Random | 14 | 4.0 | 0.6070 | 0.602 | 0.143 | +0.019 |
| B2 v_harm | 10 | 2.0 | 0.6050 | 0.668 | 0.270 | +0.017 |
| **E1 v_AB (target)** | **14** | **4.0** | **0.6185** | **0.710** | **0.374** | **+0.030** |
| **E2 v_AC (cue)** | **14** | **4.0** | **0.6239** | **0.690** | **0.369** | **+0.036** |

### ToxiGen (B0 macro_F1 = 0.5854, hate_recall = 0.423)

| Setup | layer | α | macro_F1 | hate_recall | FN_recovery | ΔF1 |
| --- | --- | --- | --- | --- | --- | --- |
| B0 No steering | — | — | 0.5854 | 0.423 | — | — |
| B1 Random | 21 | 4.0 | 0.6361 | 0.600 | 0.307 | +0.051 |
| **B2 v_harm** | **13** | **4.0** | **0.6600** | **0.742** | **0.553** | **+0.075** |
| E1 v_AB (target) | 13 | 4.5 | 0.6514 | 0.594 | 0.311 | +0.066 |
| E2 v_AC (cue) | 13 | 4.5 | 0.6544 | 0.581 | 0.298 | +0.069 |

---

## 2. 무엇이 보였나

**(1) Steering은 작동한다.** 모든 vector가 B0를 이긴다. dose-response 깔끔(α 증가 → F1 단조 증가, α≈4에서 최적). sign flip(−α)에서 F1 하락 (Latent에서 −0.017, ToxiGen에서 −0.026~−0.038) → 방향이 진짜 의미 있음.

**(2) Mid-layer(L13–14) 일관 최적.** 두 평가셋, 두 메인 벡터(v_AB, v_AC) 모두 L13–14 부근에서 최적. late layer(20–27)는 오히려 약함. axis attribution에서 본 "harm-dominant 후반 우세"와 어긋남 — **steering 최적점 ≠ 표상이 가장 분화된 지점**.

**(3) FN recovery가 macro F1보다 더 강한 메시지.** B0가 놓치는 implicit hate를 v_AB는 37%, v_AC는 37% 회복 (Latent). ToxiGen에서는 30%. macro F1 +0.03~0.07은 작아 보이지만, FN recovery 30~37%는 "B0가 놓친 케이스 중 약 1/3을 살렸다"는 직관적 메시지.

**(4) v_AB ≈ v_AC, target/cue 비대칭이 성능에선 사라짐.** 표상 수준 prior(target 효과 d=1.13, cue d=0.30 → 약 4배)가 출력 성능에선 거의 동률(Latent: AB 0.619 / AC 0.624 / ToxiGen: AB 0.651 / AC 0.654). H1 예측 빗나감.

**(5) ToxiGen에서는 v_harm이 메인 벡터를 이긴다.** macro F1·hate_recall·FN_recovery 모두 v_harm이 우세 (0.660 / 0.742 / 0.553). minimal pair 데이터의 추가 가치가 ToxiGen에서는 성립 안 함. Latent에서는 v_AB/v_AC가 v_harm을 이김 (0.62 vs 0.60).

---

## 3. Ablation — 주입 위치 효과 (best L에서)

### Latent Hatred (L=14)

| vector | setting | macro_F1 | ΔF1 | FN_rec |
| --- | --- | --- | --- | --- |
| v_AB | main (last,+α) | 0.6185 | +0.030 | 0.374 |
| v_AB | α=0 (control) | 0.5892 | +0.001 | 0.002 |
| v_AB | sign flip (−α) | 0.5709 | −0.017 | 0.015 |
| v_AB | first-token | 0.5977 | +0.010 | 0.057 |
| **v_AB** | **all-token** | **0.5508** | **−0.037** | **0.697** |
| v_AC | main (last,+α) | 0.6239 | +0.036 | 0.369 |
| **v_AC** | **all-token** | **0.5830** | **−0.005** | **0.613** |

### ToxiGen (L=13) — 흥미로운 차이

| vector | setting | macro_F1 | ΔF1 | FN_rec |
| --- | --- | --- | --- | --- |
| v_AB | main (last,+α) | 0.6514 | +0.066 | 0.311 |
| v_AB | all-token | 0.6099 | +0.025 | **0.682** |
| v_AC | main (last,+α) | 0.6544 | +0.069 | 0.298 |
| v_AC | all-token | 0.6287 | +0.043 | **0.544** |

> **all-token 주입의 패턴**: macro F1은 떨어지거나 같지만, **FN recovery가 2배 가까이 올라감** (Latent v_AB: 0.37→0.70, v_AC: 0.37→0.61). 더 많은 hate를 잡는 대신 false positive도 같이 늘어나는 trade-off. "확실히 잡고 싶을 때" 쓸 수 있는 setting.
> 

---

## 4. H1 예측 빗나감 — 솔직한 해석

미팅노트(0528)에서 H1으로 박아둔 **"v_AB(target, d=1.13)가 v_AC(cue, d=0.30)보다 4배 우세할 것"**은 빗나갔다.

- 두 벡터 효과 크기가 거의 같음 (격차 ≤ 0.005 macro F1).
- 굳이 따지면 v_AC(cue)가 미세하게 앞.

가능한 해석 두 가지.

**(a) 표상 prior가 출력 성능 prior가 아니다.** d 값은 "표상에서 두 그룹이 얼마나 분리되는가"이고, steering은 "그 방향으로 hidden을 밀어 출력을 바꿀 수 있는가"다. 분리가 크다고 steering 효과가 큰 건 보장 안 됨. 분리는 표상 통계량, steering 효과는 분류 결정 경계와의 상호작용.

**(b) target과 cue가 같은 'harmful' 방향 성분으로 수렴.** 표상에서는 둘이 다른 축으로 분리되지만(d 비대칭), 최종 출력 헤드 직전에서는 두 신호가 모두 "이건 hate" 쪽으로 통합. 그래서 어느 쪽 vector로 밀어도 비슷한 boost.

(b)는 솔직히 사후적이지만 그럴듯하고, **각 vector를 axis 별로 분해해서 두 vector의 cosine 유사도를 보면** 검증 가능. 후속.

---

## 5. v_harm > 우리 vector (ToxiGen) — 더 정직한 해석

ToxiGen에서 v_harm이 우리 메인 vector들을 이긴다는 점은 미팅노트 청구를 약화시킨다. minimal pair에서 뽑은 vector의 "추가 가치"가 무조건 성립한다고 못 한다.

가능한 원인.

**(a) ToxiGen 분포가 v_harm 추출원(HateXplain)에 더 가깝다.** v_harm은 HateXplain hate vs normal로 만들었고, ToxiGen은 GPT-3 합성 implicit hate라 문법적·구조적으로 깔끔. HateXplain hate가 다양한 styles을 포함해 그 평균 방향이 ToxiGen에 잘 transfer됐을 수 있음.

**(b) v_AB·v_AC가 cue/target에 너무 특화.** 우리 minimal pair는 cue·target 조작에 정밀하지만, 그 외 implicit hate signature(서술 구조, 우회 표현 등)는 잡지 못함. ToxiGen은 그런 다양한 implicit hate를 포함해 더 broad한 v_harm이 유리.

Latent에서는 v_AB/v_AC > v_harm이라 결과가 split. 이 split을 어떻게 메시지화할지가 다음 결정 포인트.

---

## 6. 다른 흥미로운 관찰

**Random vector(B1)도 B0를 이긴다.** Latent에서 +0.019, ToxiGen에서 +0.051. random인데도 효과가 있다는 건, **steering 효과 일부는 "특정 방향"보다 "hidden을 평균에서 멀리 밀어내는 것" 자체에 기인**할 수 있음. probe가 학습한 결정 경계 근처에서 살짝만 밀어도 분류가 바뀌는 케이스. 이건 "Sober Look at Steering Vectors" (Pres et al. 2024)에서 경고한 패턴. **우리 vector의 "진짜 effect"는 B1 대비 추가 이득이지, B0 대비 전체가 아님**.

| vector | Latent ΔF1 vs B0 | Latent ΔF1 vs B1 | ToxiGen ΔF1 vs B0 | ToxiGen ΔF1 vs B1 |
| --- | --- | --- | --- | --- |
| v_AB | +0.030 | +0.012 | +0.066 | +0.015 |
| v_AC | +0.036 | +0.017 | +0.069 | +0.018 |
| v_harm | +0.017 | −0.002 | +0.075 | +0.024 |

**B1 대비 보면 모든 효과가 작아진다.** v_harm은 Latent에서 B1을 못 이김. ToxiGen에서도 가장 큰 ΔF1(v_harm B1 대비 +0.024)도 macro F1 0.024 향상이라 micro effect.

**Sign flip이 항상 음의 효과로 가지는 않는다.** Latent에서 v_AB −α는 −0.017이지만, ToxiGen v_AB −α는 −0.038로 더 큼. 방향 의미가 ToxiGen에서 더 뚜렷.

---

## 7. 메시지 재정렬 — 세 가지 선택지

원래 미팅노트 메시지("target 축이 메인, cue 축은 보조, 두 벡터 모두 v_harm 능가")는 데이터가 부분적으로만 지지. 재프레이밍 필요.

**선택지 A — 정직한 finding 보고**: "두 minimal pair vector가 B0를 의미 있게 이기지만, target/cue 비대칭이 출력 성능에선 사라진다. v_harm baseline과의 비교는 평가셋에 따라 split된다(Latent에서 우세, ToxiGen에서 열세)." H1 빗나감 자체를 흥미로운 finding으로 프레이밍. 약점: "그래서 우리 vector를 왜 써야 하는가"가 명확하지 않음.

**선택지 B — Mid-layer 메커니즘에 집중**: "implicit hate 탐지 개선은 mid-layer(L13-14)에서 일어난다. late layer의 표상 분화(axis attribution)와 steering 최적점이 다르다는 것이 mechanistic finding." 본인의 axis attribution(late 우세)과 steering(mid 우세)의 괴리를 핵심 메시지로. minimal pair vs v_harm 논쟁 회피.

**선택지 C — FN recovery 메시지**: "macro F1 +Δ는 작지만, B0가 놓치는 implicit hate의 약 37%를 회복한다. 모더레이션 실용 관점에서 의미 있다." all-token 주입에서 60-70%까지 가능. 약점: F1과 FN recovery의 trade-off가 명확해서 "실용적으로 어느 setting을 쓸 건가"가 모호.

> 권장: **B + C 조합**. mid-layer 메커니즘으로 mechanistic finding을 깔고, FN recovery 30-37%(또는 all-token 60%)로 실용적 가치를 더함. H1 빗나감과 v_harm split은 한계로 명시.
> 

---

## 8. 한계 — 선제 명시

1. **B1 random vector도 B0를 이긴다.** 우리 vector의 진짜 추가 효과는 B0 대비가 아니라 B1 대비로 봐야 함. 그 격차는 macro F1 +0.012~0.018로 작음.
2. **H1 예측 빗나감.** target/cue 표상 비대칭이 성능에 재현 안 됨.
3. **v_harm split.** Latent에서 우리 vector가 v_harm을 이기지만 ToxiGen에서는 진다. minimal pair 데이터의 우위가 일반적으로 성립한다고 못 함.
4. **Probe 자체의 한계.** B0가 Llama-3.2-3B last token hidden에 학습한 단순 logistic probe. steering이 "Llama 표상 개선"인지 "probe 한계 보정"인지 완전히 분리하기 어려움.
5. **Statistical significance.** 보고된 ΔF1에 신뢰구간/유의성 검정 없음. McNemar 등 후속 필요.
6. **두 평가셋만.** HateXplain held-out sanity 결과는 같이 보고하지 않음.

---

# 추가 실험 방향 정리

> **목적**: 현재 결과의 노벨티 부족 문제를 해결하기 위한 후속 실험 설계. 두 가지 축 — (A) L14가 왜 특별한가, (B) 우리 데이터셋 벡터가 왜 더 나은가 — 을 실험적으로 검증한다.
> 

---

## 1. Layer별 Linear Probe 정확도 곡선

### 선행연구 근거

**"Shifting Perspectives: Steering Vectors for Bias Mitigation" (2025, arxiv 2503.05371)**이 직접 사용한 방법이다. BBQ 데이터셋 8개 축(race, gender 등)별 300개 대조쌍으로 벡터를 추출하고, 각 레이어에서 PCA로 선형 분리 가능성을 시각화했다. L7, L13에서 logistic regression 분류 정확도가 명시적으로 높게 나타났다.

또한 **"Where to Steer" (2025, arxiv 2604.03867)**은 고정된 레이어에 steering하는 것이 최적이 아님을 이론·실증으로 보여줬다. 입력마다 최적 레이어가 다르다고 주장하며 W2S 프레임워크를 제안해 13개 데이터셋에서 fixed-layer보다 일관되게 좋은 성능을 보였다. **우리 실험에서 L14가 모든 조건에서 일관되게 최적으로 나오는 현상은 이 논문의 반례가 될 수 있다.** Implicit hate라는 특수한 도메인에서 optimal layer가 안정적으로 수렴한다는 것이 독립적인 finding이다.

### 가설

L14 근처에서 implicit hate probe 정확도가 peak를 찍고, last layer(-1)에서는 오히려 낮아진다. 이것이 "L14에 implicit hate 신호가 가장 강하게 선형 분리되어 있는데, 모델의 기본 처리 과정이 이 신호를 final layer까지 유지하지 못한다"는 mechanistic story의 직접 증거다.

### 구현

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# h_train: (N_train, N_LAYERS+1, hidden)  — extract_all_layers로 추출
# h_eval:  (N_eval,  N_LAYERS+1, hidden)

scores = {}
for L in range(N_LAYERS + 1):
    X_train_L = h_train[:, L, :]
    X_eval_L  = h_eval[:, L, :]
    sc = StandardScaler().fit(X_train_L)
    clf_L = LogisticRegression(C=1.0, max_iter=500)
    clf_L.fit(sc.transform(X_train_L), y_train)
    scores[L] = evaluate(
        clf_L.predict(sc.transform(X_eval_L)), ev_labels
    )
```

그래프: x축 레이어, y축 macro F1. v_harm axis attribution(harm_gap_z) 곡선을 오른쪽 y축에 overlay. 두 곡선의 peak 위치가 일치하는지 또는 diverge하는지를 본다.

### 기대 결과 해석

| 결과 패턴 | 해석 |
| --- | --- |
| L14에서 probe F1 peak, last layer 하락 | L14가 implicit hate 처리의 핵심 레이어 — mechanistic contribution 성립 |
| probe peak ≠ steering 최적 레이어 | "표상 분리 ≠ steering 효과" 추가 dissociation finding |
| last layer에서도 probe F1 유지 | L14 효과가 representation이 아니라 다른 요인 — 추가 분석 필요 |

---

## 2. Implicit-only Subset 재측정

### 근거

현재 평가셋(eval_latent_v2)은 implicit hate + explicit hate를 모두 포함한다. 우리 벡터(v_AB, v_AC)는 Cell A/B/C — implicit hate 구조에서 만든 대조쌍 — 로 추출했으므로, **explicit hate를 제외한 implicit-only subset에서 랜덤 벡터와의 격차가 더 크게 나와야 한다.**

만약 implicit-only에서 v_AB > v_random by more라면, "우리 벡터는 implicit hate에 특화된 방향을 포착한다"는 주장이 성립한다.

### 구현

```python
# eval_latent_v2.csv에 subtype 컬럼 있어야 함
impl_mask = (ev["subtype"] == "implicit_hate").to_numpy()

# week2 sweep에서 best 셋업별 pred 배열을 이미 갖고 있다면
all_preds = {
    "B0":       pred_b0,
    "B1_random": pred_b1,
    "B2_harm":   pred_b2,
    "E1_v_AB":   pred_ab,
    "E2_v_AC":   pred_ac,
}
for name, pred in all_preds.items():
    m_full = evaluate(pred, ev_labels)
    m_impl = evaluate(pred[impl_mask], ev_labels[impl_mask])
    print(f"{name}: full={m_full['macro_f1']:.4f}  impl-only={m_impl['macro_f1']:.4f}")
```

### 핵심 비교 포인트

- **v_AB vs v_random**: implicit-only에서 격차 > 전체 평가셋 격차인가?
- **v_AB vs v_harm**: implicit-only에서 우리 벡터가 우세한가? (Latent 전체에서는 이미 우세)

---

## 3. Robustness Check — 데이터셋 크기 실험 (256 → 128)

### 근거

256개로 만든 벡터와 128개로 만든 벡터 성능이 비슷하게 나오면, **"적은 minimal pair 데이터로도 충분히 효과적인 steering vector를 만들 수 있다"**는 practical contribution이 생긴다. 데이터 구축 비용이 높은 minimal pair 방법론의 실용성을 뒷받침하는 결과다.

### 구현

```python
subset_128 = tri.sample(128, random_state=42)
v_AB_128 = steering_vector(
    extract_all_layers(subset_128["A"].tolist()),
    extract_all_layers(subset_128["B"].tolist())
)
np.save("results/v_AB_128.npy", v_AB_128)

# 동일한 sweep grid(best L, best α)로 성능 측정
# 결과 비교: v_AB_256 vs v_AB_128
```

### 추가 변형

- **랜덤 시드 3개로 반복 샘플링** → 분산 확인. 벡터가 샘플링에 안정적이면 더 강한 주장 가능.
- **타겟 2개만 선별** (action item): Cohen's d가 가장 높은 쌍만 골라서 벡터 추출 시 성능 변화 확인.

---

## 4. Cosine Similarity by Layer

### 목적

- **v_AB ↔ v_AC**: H1(target 축 우세 예측)이 빗나간 이유 — mid-layer에서 두 벡터가 수렴하는가?
- **v_AB ↔ v_harm**: ToxiGen에서 v_harm이 우리 벡터를 이기는 이유 — minimal pair 벡터가 harm 방향의 부분집합인가?

### 구현

```python
import torch.nn.functional as F

def cos_by_layer(a, b):
    # a, b: (N_LAYERS+1, hidden)
    return [
        F.cosine_similarity(
            torch.tensor(a[L]).unsqueeze(0),
            torch.tensor(b[L]).unsqueeze(0)
        ).item()
        for L in range(len(a))
    ]

sim_AB_AC   = cos_by_layer(v_AB, v_AC)
sim_AB_harm = cos_by_layer(v_AB, v_harm)
sim_AC_harm = cos_by_layer(v_AC, v_harm)
```

그래프: 세 선을 같은 그림에. L13-14 vertical line 표시. mid-layer에서 cosine이 높아지면 "두 벡터가 같은 방향으로 수렴"의 직접 증거.

---

## 5. 파일별 수정 가이드 — 어디에 추가할까

| 실험 | 수정할 파일 | 이유 |
| --- | --- | --- |
| §4 Cosine similarity | **week3_analysis.py 맨 끝** | VECTORS dict가 이미 로드되어 있음. 모델 재로드 불필요. |
| §1 Layer별 linear probe | **week3_analysis.py** (axis attribution 섹션 바로 뒤) | extract_all_layers, scaler, clf, EV, FN이 모두 이미 정의되어 있음. |
| §2 Implicit-only subset | **week3_analysis.py** (final_table 함수 안 또는 바로 뒤) | prediction 배열을 run_ablation에서 이미 생성. ev 데이터프레임도 로드되어 있음. |
| §3 데이터셋 크기 실험 | **week2_sweep.py** (build_vectors 함수 뒤에 새 함수 추가) | Cell A/B 로딩, extract_all_layers, steering_vector, sweep 구조가 모두 있음. |

---

### §4 Cosine Similarity 추가 코드 (week3_analysis.py 맨 끝)

```python
# ── 추가: Cosine Similarity by Layer ──────────────────────
import torch.nn.functional as F

def cos_by_layer(a, b):
    return [F.cosine_similarity(torch.tensor(a[L]).unsqueeze(0),
                                torch.tensor(b[L]).unsqueeze(0)).item()
            for L in range(len(a))]

sim_AB_AC   = cos_by_layer(VECTORS["v_AB"], VECTORS["v_AC"])
sim_AB_harm = cos_by_layer(VECTORS["v_AB"], VECTORS["v_harm"])
sim_AC_harm = cos_by_layer(VECTORS["v_AC"], VECTORS["v_harm"])

np.save(os.path.join(args.out, "cosine_AB_AC.npy"),   sim_AB_AC)
np.save(os.path.join(args.out, "cosine_AB_harm.npy"), sim_AB_harm)
np.save(os.path.join(args.out, "cosine_AC_harm.npy"), sim_AC_harm)

layers = list(range(len(sim_AB_AC)))
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(layers, sim_AB_AC,   label="v_AB vs v_AC",   marker="o", ms=3)
ax.plot(layers, sim_AB_harm, label="v_AB vs v_harm",  marker="s", ms=3)
ax.plot(layers, sim_AC_harm, label="v_AC vs v_harm",  marker="^", ms=3)
ax.axvline(13, color="gray", ls="--", alpha=0.5)
ax.axvline(14, color="gray", ls=":",  alpha=0.5, label="L13/14")
ax.set_xlabel("Layer"); ax.set_ylabel("Cosine similarity")
ax.set_title("Steering vector cosine similarity by layer")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(args.out, "cosine_similarity.png"), dpi=150); plt.close()
print("[cosine] 저장 완료")
for L in [13, 14, 20, 27]:
    print(f"  L{L:2d}: AB^AC={sim_AB_AC[L]:+.3f}  AB^harm={sim_AB_harm[L]:+.3f}")
```

---

### §1 Layer별 Linear Probe 추가 코드 (week3_analysis.py — axis attribution 섹션 바로 뒤)

```python
# ── 추가: Layer별 Linear Probe 정확도 곡선 ────────────────
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler as SS

# HateXplain train hidden 추출 (probe 학습용)
# week1에서 probe를 hx_train으로 학습했으므로 같은 데이터 재사용
hx_tr = pd.read_csv(os.path.join(args.data_eval,
               "..", "hatexplain_train.csv"))  # 경로 확인 필요
print("[layer_probe] HateXplain train hidden 추출")
h_hx_all = extract_all_layers(hx_tr["text"].tolist())  # (N, L+1, H)
y_hx     = hx_tr["label"].astype(int).to_numpy()

# eval hidden 추출 (all layers)
for tag in TAGS:
    texts, labels, _ = EV[tag]
    print(f"[layer_probe] {tag} eval hidden 추출")
    h_ev_all = extract_all_layers(texts)   # (N_eval, L+1, H)

    layer_scores = {}
    for L in range(N_LAYERS + 1):
        sc_L  = SS().fit(h_hx_all[:, L, :])
        clf_L = LR(C=1.0, max_iter=500)
        clf_L.fit(sc_L.transform(h_hx_all[:, L, :]), y_hx)
        layer_scores[L] = evaluate(
            clf_L.predict(sc_L.transform(h_ev_all[:, L, :])), labels
        )["macro_f1"]

    # 저장
    pd.DataFrame({"layer": list(layer_scores.keys()),
                  "macro_f1": list(layer_scores.values())}).to_csv(
        os.path.join(args.out, f"layer_probe_{tag}.csv"), index=False)

    # 그래프 (harm_gap_z overlay)
    harm_z = np.load(os.path.join(args.out, "harm_gap_z_lasttoken.npy"))
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(list(layer_scores.keys()), list(layer_scores.values()),
             marker="o", ms=4, label="Layer probe macro F1")
    ax1.axvline(14, color="gray", ls="--", alpha=0.6, label="L14")
    ax1.set_xlabel("Layer"); ax1.set_ylabel("Probe macro F1")
    ax2 = ax1.twinx()
    ax2.plot(range(len(harm_z)), np.abs(harm_z),
             color="orange", lw=1.5, ls=":", alpha=0.8, label="harm |z|")
    ax2.set_ylabel("harm-axis |z|", color="orange")
    ax1.set_title(f"Layer probe accuracy + harm axis [{tag}]")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"layer_probe_{tag}.png"), dpi=150)
    plt.close()
    print(f"[layer_probe] {tag} 저장 완료")
    peak_L = max(layer_scores, key=layer_scores.get)
    print(f"  peak L={peak_L}  F1={layer_scores[peak_L]:.4f} | last L={N_LAYERS}  F1={layer_scores[N_LAYERS]:.4f}")
```

---

### §2 Implicit-only Subset 추가 코드 (week3_analysis.py — final_table 루프 뒤)

```python
# ── 추가: Implicit-only subset 재측정 ─────────────────────
ev_df = pd.read_csv(os.path.join(args.data_eval, "eval_latent_v2.csv"))
if "subtype" in ev_df.columns:
    impl_mask = (ev_df["subtype"] == "implicit_hate").to_numpy()
    texts, labels, _ = EV["eval_latent_v2"]
    batches, inv = prebatch(texts)

    rows = []
    setups = [("B0", None, None, 0.0), ("B1_random", "v_random", None, None),
              ("B2_harm",  "v_harm",  None, None),
              ("E1_v_AB",  "v_AB",    None, None),
              ("E2_v_AC",  "v_AC",    None, None)]
    for name, vec_name, _, _ in setups:
        if vec_name is None:
            X = extract(batches, inv, mode="none")
        else:
            L, a, _ = BEST["eval_latent_v2"][vec_name]
            X = extract(batches, inv, VECTORS[vec_name], L, a, "last")
        pred = predict(X)
        m_full = evaluate(pred, labels)
        m_impl = evaluate(pred[impl_mask], labels[impl_mask])
        rows.append({"setup": name,
                     "full_f1":  round(m_full["macro_f1"], 4),
                     "impl_f1":  round(m_impl["macro_f1"], 4),
                     "impl_hate_recall": round(m_impl["hate_recall"], 4)})
        print(f"{name}: full={m_full['macro_f1']:.4f}  impl-only={m_impl['macro_f1']:.4f}")

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(args.out, "implicit_only_table.csv"), index=False)
    print(f"\n===== IMPLICIT-ONLY TABLE =====\n{t.to_string(index=False)}")
else:
    print("[impl-only] subtype 컬럼 없음 — eval_latent_v2.csv 확인 필요")
```

---

### §3 데이터셋 크기 실험 추가 코드 (week2_sweep.py — build_vectors 함수 뒤)

```python
# ── 추가: size_robustness sweep ──────────────────────────
SIZES = [32, 64, 128, 256]

def run_size_robustness(tag, texts, labels, fn_idx, b0m):
    path = os.path.join(args.out, f"size_robustness_{tag}.csv")
    done = {int(r.n) for r in pd.read_csv(path).itertuples()} \
           if os.path.exists(path) else set()
    batches, inv = prebatch(texts)

    for n in SIZES:
        if n in done: continue
        sub = tri.sample(min(n, len(tri)), random_state=args.seed)
        hA  = extract_all_layers(sub["A"].tolist())
        hB  = extract_all_layers(sub["B"].tolist())
        vec = unit_per_layer(hA.mean(0) - hB.mean(0))  # v_AB_n

        L_use = int(pd.read_csv(os.path.join(args.out, f"sweep_coarse_{tag}.csv"))
                      .query("vector=='v_AB'").sort_values("macro_f1").iloc[-1]["layer"])
        a_use = float(pd.read_csv(os.path.join(args.out, f"sweep_coarse_{tag}.csv"))
                      .query("vector=='v_AB'").sort_values("macro_f1").iloc[-1]["alpha"])

        m, fnrec, _ = sweep_one(vec, L_use, a_use, batches, inv, labels, fn_idx)
        append_row(path, dict(n=n, layer=L_use, alpha=a_use,
                              macro_f1=round(m["macro_f1"],4),
                              hate_recall=round(m["hate_recall"],4),
                              fn_recovery=round(fnrec,4),
                              d_f1=round(m["macro_f1"]-b0m["macro_f1"],4)))
        print(f"[size] n={n:3d} L={L_use} a={a_use} F1={m['macro_f1']:.4f}")

# week2 맨 끝 실행부에 추가
for tag, txt, lab, fn, b0m in targets:
    run_size_robustness(tag, txt, lab, fn, b0m)
```

---

### 로컬 실행 순서 — 환경별 가이드

#### macOS (M4 Pro)

```bash
# 1. 코사인 유사도 — 모델 불필요, 바로 가능 (1분)
python week3_analysis.py --eval latent

# 2. 데이터셋 크기 실험 (~10분)
python week2_sweep.py --batch 16 --eval latent --no-fine

# 3. Layer probe + implicit-only (~15분)
python week3_analysis.py --eval latent
```

#### Windows (CUDA GPU)

**환경 확인 먼저:**

```powershell
# GPU 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**코드에서 바꿔야 할 부분 (단 한 줄):**

```python
# week2_sweep.py / week3_analysis.py 상단의 device 설정
# 기존:
device = "cuda" if torch.cuda.is_available() else "cpu"
# Windows + CUDA면 이미 자동으로 cuda로 잡힘 — 바꿀 거 없음
# mps 관련 코드가 있다면 제거:
# device = "mps" if torch.backends.mps.is_available() else ...  <-- 이 줄 삭제
```

**실행 명령어 (PowerShell 또는 CMD):**

```powershell
# 1. 코사인 유사도 (1분)
python week3_analysis.py --eval latent

# 2. 데이터셋 크기 실험
python week2_sweep.py --batch 32 --eval latent --no-fine

# 3. Layer probe + implicit-only
python week3_analysis.py --eval latent
```

**Windows GPU별 예상 소요 시간:**

| GPU | VRAM | batch 권장값 | 데이터셋 크기 실험 | Layer probe |
| --- | --- | --- | --- | --- |
| RTX 4090 / 3090 | 24GB | 64 | ~5분 | ~8분 |
| RTX 4080 / 3080 | 10~16GB | 32 | ~8분 | ~12분 |
| RTX 3070 / 4070 | 8~12GB | 16 | ~12분 | ~18분 |
| RTX 3060 8GB 이하 | 8GB↓ | 8, max_len=64 | ~20분 | ~30분 |

**VRAM 부족 오류 날 때:**

```powershell
# batch 줄이고 max_len 줄이기
python week2_sweep.py --batch 8 --max-len 64 --eval latent --no-fine
```

#### CPU only (GPU 없는 경우)

Layer probe, sweep 실험은 사실상 불가능(수 시간 소요).

코사인 유사도만 로컬에서 하고 나머지는 서버에서 돌릴 것.

```powershell
# 코사인 유사도만 — GPU 없이도 가능 (벡터 .npy 파일만 있으면 됨, 1분)
python -c "
import numpy as np, torch, torch.nn.functional as F, os
v_AB   = np.load('results/v_AB.npy')
v_AC   = np.load('results/v_AC.npy')
v_harm = np.load('results/v_harm.npy')
def cos(a,b): return [F.cosine_similarity(torch.tensor(a[L]).unsqueeze(0), torch.tensor(b[L]).unsqueeze(0)).item() for L in range(len(a))]
sim = cos(v_AB, v_AC)
for L in [13,14,20,27]: print(f'L{L}: {sim[L]:+.3f}')
"
```

#### 공통 주의사항

- `hatexplain_train.csv` 경로가 맞는지 확인. 스크립트에서 `args.data_eval + "/../hatexplain_train.csv"` 또는 `--hatexplain-csv` 인자로 직접 지정 가능.
- Windows에서 경로 구분자는 `\` 이지만 Python에서는 `/` 또는 `os.path.join()` 모두 동작함.
- HuggingFace 모델 캐시는 Windows 기준 `C:\Users\{유저명}\.cache\huggingface\` 에 저장됨. 용량 약 6GB 필요.
