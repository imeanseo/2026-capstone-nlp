## 핵심 메시지: FN recovery

이번 발표에서 가장 먼저 밀어야 할 문장은 macro F1 향상이 아니라 **B0가 놓친 implicit hate를 minimal-pair steering이 얼마나 되살렸는가**이다. Macro F1 +0.03~0.07은 숫자만 보면 작아 보일 수 있지만, FN recovery 30~46%는 “기존 모델이 놓친 사례의 약 1/3 이상을 다시 잡았다”는 실용적 의미가 바로 전달된다.

따라서 “성능을 조금 올렸다”가 아니라 **“implicit hate처럼 놓치기 쉬운 사례를 targeted steering으로 회복했다”**로 잡는다. 이 문장이 Intro, Result, Conclusion에 반복해서 들어가야 한다.

## 피드백 반영 요약

| 저번 피드백 | 이번에 반영한 수정 | 교수님께 말할 포인트 |
| --- | --- | --- |
| 배경 그래프를 평가셋별로 따로 계산해야 한다. | Latent와 ToxiGen 각각에서 hate vs non-hate를 `v_harm`에 투영해 Graph A를 다시 계산했다. | 이전 방법론 오류를 고쳤고, 평가셋별로 representation strength와 steering best가 다르게 정렬됨을 확인했다. |
| ToxiGen에서 v_harm이 높은 결과를 조심스럽게 해석해야 한다. | v_harm L=1, α=4.0을 SST-2 probe와 zero-shot 방식으로 sanity check했다. | v_harm L=1은 ToxiGen 점수는 높지만 일반 sentiment 능력을 크게 손상시켜, 안정적인 steering으로 보기 어렵다. |
| Train/test split과 평가 구조를 명확히 해야 한다. | Cell A/B/C는 vector extraction용, Latent/ToxiGen은 evaluation용으로 분리해 설명한다. | 같은 데이터로 vector를 만들고 평가한 것이 아니라 OOD evaluation 구조임을 분명히 말한다. |
| 정성 사례가 필요하다. | B0와 dehatebert가 놓친 사례 중 v_AB/v_AC가 회복한 문장을 선별했다. | FN recovery가 단순 숫자가 아니라 실제 문장 수준에서 어떤 의미인지 보여준다. |
| 발표는 논문 구조처럼 구성해야 한다. | Intro → Background → Proposed Approach → Experiments → Discussion → Conclusion 흐름으로 재구성한다. | 관련 연구는 짧게, Proposed Approach와 Experimental Results에 시간을 집중한다. |

## 논문 작성 피드백 반영

교수님이 말한 테크니컬 페이퍼 구조를 발표 흐름에도 반영한다. 발표자료는 단순 실험 나열이 아니라 **논문 밑그림**처럼 보여야 한다.

| 논문 구성 원칙 | 이번 발표에서 반영할 방식 |
| --- | --- |
| Title은 문장이 아니라 연구 분야와 핵심 approach가 드러나는 phrase여야 한다. | 제목 후보는 “Minimal-Pair Steering for Implicit Hate Speech Detection”처럼 연구 분야와 방법 키워드를 모두 포함한다. |
| Abstract는 Background → Method → Result → Conclusion 흐름이다. | 발표 첫 요약도 “implicit hate가 어렵다 → minimal-pair steering을 제안한다 → FN recovery가 오른다 → mid-layer 개입이 효과적이다” 순서로 말한다. |
| 2장 Background는 길게 쓰지 않는다. | 관련 연구는 hate speech detection, representation steering, minimal-pair data 정도만 짧게 언급하고 본문 시간은 방법과 실험에 쓴다. |
| 3장 Proposed Approach와 4장 Experiments가 핵심이다. | 발표 본문도 minimal-pair vector 생성 방식과 실험 결과표·그래프를 중심으로 구성한다. |
| 좋은 결과뿐 아니라 의도하지 않은 결과도 설명해야 한다. | ToxiGen에서 v_harm이 높았던 결과, AB/AC 결합 이득이 없었던 결과를 숨기지 않고 appendix 방어 논리로 정리한다. |

## 추가 실험 결과: 본문과 appendix 후보

### 수정 Graph A: 평가셋별 배경 재계산

기존에는 harm 배경을 두 평가셋에 동일하게 사용한 문제가 있었기 때문에, Latent와 ToxiGen 각각에서 hate vs non-hate를 따로 `v_harm`에 투영해 배경을 다시 계산했다.

**Latent**

!graphA_axis_eval_latent_v2.png

Latent에서는 E1/E2가 random과 v_harm을 모두 이기지만, harm |z|가 큰 후반 레이어와 steering 성능 봉우리가 어긋난다. 따라서 **혐오 표상이 강한 지점과 개입이 잘 먹히는 지점은 다르다**고 설명한다.

| Setup | Layer | α | macro F1 | hate recall | FN recovery | ΔF1 |
| --- | --- | --- | --- | --- | --- | --- |
| B0 No steering | — | — | 0.5882 | 0.545 | — | — |
| B1 Random | 14 | 4.0 | 0.6070 | 0.602 | 0.143 | +0.019 |
| B2 v_harm | 10 | 2.0 | 0.6050 | 0.668 | 0.270 | +0.017 |
| **E1 v_AB** | **8** | **4.25** | **0.6194** | **0.695** | **0.457** | **+0.031** |
| **E2 v_AC** | **14** | **4.0** | **0.6239** | **0.690** | **0.369** | **+0.036** |

Latent에서는 `v_AB`와 `v_AC`가 B0/B1/B2보다 모두 높은 macro F1을 보인다. 특히 FN recovery가 37~46%로, baseline이 놓친 hate를 실제로 많이 회복한다는 점이 핵심이다.

**ToxiGen**

!graphA_axis_eval_toxigen_v1.png

| Setup | Layer | α | macro F1 | hate recall | FN recovery | ΔF1 |
| --- | --- | --- | --- | --- | --- | --- |
| B0 No steering | — | — | 0.5854 | 0.423 | — | — |
| B1 Random | 21 | 4.0 | 0.6361 | 0.600 | 0.307 | +0.051 |
| **B2 v_harm** | **1** | **4.0** | **0.6730** | **0.753** | **0.604** | **+0.088** |
| E1 v_AB | 13 | 4.5 | 0.6514 | 0.594 | 0.311 | +0.066 |
| E2 v_AC | 13 | 4.5 | 0.6544 | 0.581 | 0.298 | +0.069 |

ToxiGen에서는 v_AB와 v_AC의 best가 모두 L13이고, 이 지점이 ToxiGen 기준 harm |z| 상위 레이어와 맞는다. 

ToxiGen에서는 E1/E2도 B0/B1보다 macro F1이 높지만, raw best는 `v_harm L=1`이다. 이 부분은 숨기지 말고, 뒤의 SST-2 sanity check로 해석을 보완한다.

| Eval | B0 | E1 | ΔF1 | McNemar p | Bootstrap 95% CI |
| --- | --- | --- | --- | --- | --- |
| Latent (L=8, α=4.25) | 0.5892 | 0.6194 | +0.030 | **0.004** | [+0.009, +0.051] |
| ToxiGen (L=13, α=4.5) | 0.5867 | 0.6514 | +0.065 | **2.8e-08** | [+0.045, +0.084] |

**McNemar test**는 같은 문장들에 대해 B0와 E1이 **맞고 틀린 패턴이 실제로 달라졌는지** 보는 paired test다. 단순히 F1 숫자만 비교하는 것이 아니라, 같은 evaluation sample에서 “B0는 틀렸는데 E1은 맞춘 경우”와 “B0는 맞췄는데 E1은 틀린 경우”의 불균형을 본다. 여기서 p-value가 0.05보다 작으면, 두 모델의 오류 패턴 차이가 우연이라고 보기 어렵다고 말할 수 있다.

**Bootstrap 95% CI**는 evaluation sample을 여러 번 다시 뽑아 macro F1 차이 ΔF1이 얼마나 안정적인지 보는 신뢰구간이다. 95% CI가 0을 포함하지 않고 전부 양수이면, 데이터 샘플이 조금 달라져도 E1의 성능 향상이 유지될 가능성이 높다는 뜻이다.

발표 문장:

> 두 평가셋 모두 McNemar test와 bootstrap CI에서 유의성을 확인했다. McNemar test는 같은 샘플에서 오류 패턴이 실제로 달라졌는지를 보고, bootstrap CI는 F1 향상이 샘플링 변동에도 안정적인지를 본다. 따라서 관찰된 향상은 단순한 샘플 변동으로 보기 어렵다.
> 

| Eval | Vector | Setting | macro F1 | ΔF1 | FN recovery |
| --- | --- | --- | --- | --- | --- |
| Latent | v_AB | last-token | 0.6194 | +0.031 | 0.457 |
| Latent | v_AB | all-token | 0.5238 | −0.064 | **0.820** |
| Latent | v_AC | last-token | 0.6239 | +0.036 | 0.369 |
| Latent | v_AC | all-token | 0.5830 | −0.005 | **0.613** |
| ToxiGen | v_AB | last-token | 0.6514 | +0.066 | 0.311 |
| ToxiGen | v_AB | all-token | 0.6099 | +0.025 | **0.682** |
| ToxiGen | v_AC | last-token | 0.6544 | +0.069 | 0.298 |
| ToxiGen | v_AC | all-token | 0.6287 | +0.043 | **0.544** |

발표 문장:

> Last-token 주입은 macro F1 기준 가장 균형 잡힌 설정이고, all-token 주입은 FN recovery를 크게 높이는 대신 FP가 늘어나는 공격적 설정이다. 따라서 둘은 우열이 아니라 운영 목적에 따른 trade-off다.
> 

### v_harm L=1 sanity check: 높은 점수의 부작용

ToxiGen에서 v_harm L=1, α=4.0이 가장 높은 macro F1을 보였지만, SST-2에서는 일반 sentiment 능력이 크게 무너졌다. 이 표는 “v_harm이 높았는데 왜 메인 메시지로 안 미느냐”는 질문에 대한 핵심 방어 자료다.

**Probe 기반 SST-2 결과**

| setup | acc | Δacc | neg_acc | pos_acc | pred_pos_rate |
| --- | --- | --- | --- | --- | --- |
| B0 no steering | 0.9025 | — | 0.886 | 0.919 | 0.524 |
| **v_harm L=1 α=4.0** | **0.5906** | **−0.3119** | **0.963** | **0.232** | **0.136** |
| v_harm L=13 α=4.0 | 0.8933 | −0.0092 | 0.944 | 0.845 | 0.458 |
| v_AB L=13 α=4.5 | 0.8544 | −0.0482 | 0.780 | 0.926 | 0.579 |
| v_AC L=13 α=4.5 | 0.8796 | −0.0229 | 0.953 | 0.809 | 0.435 |

**Zero-shot SST-2 결과**

| setup | acc | Δacc | neg_acc | pos_acc | pred_pos_rate |
| --- | --- | --- | --- | --- | --- |
| B0 no steering | 0.8784 | — | 0.792 | 0.962 | 0.592 |
| **v_harm L=1 α=4.0** | **0.5906** | **−0.2878** | **0.467** | **0.709** | **0.623** |
| v_harm L=13 α=4.0 | 0.8544 | −0.0241 | 0.946 | 0.766 | 0.416 |
| v_AB L=13 α=4.5 | 0.8865 | +0.0080 | 0.836 | 0.935 | 0.556 |
| v_AC L=13 α=4.5 | 0.8796 | +0.0011 | 0.946 | 0.815 | 0.442 |

!image.png

발표에서는 이 결과를 “v_harm L=1은 ToxiGen 점수는 높지만, 일반 능력을 크게 훼손한다. 그래서 단순 최고 점수보다 안정적인 개입인지 함께 봐야 한다”로 정리한다.

> ToxiGen에서 v_harm이 raw score로 가장 높았다는 점은 그대로 보고한다. 다만 해당 best setting은 SST-2에서 일반 sentiment 판단을 크게 손상시켰다. 따라서 v_harm L=1은 “좋은 steering”이라기보다 early-layer 과개입으로 recall을 과하게 끌어올린 설정일 수 있다.
> 

### 정성평가: 회복된 false negative 사례

FN recovery가 추상적인 숫자로만 보이지 않도록, B0가 놓쳤지만 v_AB/v_AC가 회복한 사례를 표로 보여준다.

**전체 recovery 통계**

| 지표 | Latent (n=2000) | ToxiGen (n=1560) |
| --- | --- | --- |
| B0가 놓친 hate (FN) | 451 | 449 |
| v_AB 회복 | **197 (43.7%)** | 140 (31.2%) |
| v_AC 회복 | 151 (33.5%) | 138 (30.7%) |
| 둘 중 하나라도 (any) | 227 (50.3%) | 172 (38.3%) |
| 둘 다 (both) | 121 (26.8%) | 106 (23.6%) |
| 회복 케이스 중 v_harm도 놓친 것 | **137 / 227** | 45 / 172 |

이 표는 본문 Result 1 또는 appendix 앞부분에 넣기 좋다. 특히 Latent에서 회복 케이스 227개 중 137개는 v_harm도 놓쳤으므로, minimal-pair vector가 generic harm 축과 다른 사례를 잡는다는 보조 근거가 된다.

**선별된 예시 + dehatebert**

| id | 대상 | B0 | v_AB/v_AC | v_harm | dehatebert P(hate) | 모더레이션 놓침 |
| --- | --- | --- | --- | --- | --- | --- |
| **LH_v1_00014** | 흑인+장애 | non-hate | 둘 다 hate | 놓침 | **0.029** | ✓ |
| **LH_v1_01084** | 무슬림 | non-hate | 둘 다 hate | 잡음 | **0.028** | ✓ |
| LH_v1_01422 | 이민(DACA) | non-hate | 둘 다 | 놓침 | 0.117 | ✓ |
| LH_v1_01963 | 무슬림형제단 | non-hate | 둘 다 | 놓침 | 0.068 | ✓ |

이 예시는 발표 첫 장 또는 appendix G에 넣는다. “dehatebert가 P(hate)≈0.03으로 거의 확신하며 놓친 문장을 minimal-pair steering이 회복했다”는 식으로 말하면 FN recovery의 의미가 직관적으로 전달된다.

**발표 시 사용 가능한 예시:**

- LH_v1_00014: time to start a serious discussion on disarming black people and reviving insane asylums.
    
    → 특정 인종을 대상으로 욕설 없이 배제 프레임으로 위해를 암시
    
- LH_v1_01084: Jihad is one of the Five Pillars of Islam. The whole thing is basically founded on going out and killing people for G-d.
    
    → 종교 집단을 대상으로 욕설 없이 사실 진술처럼 보이는 일반화 프레임, 집단을 폭력과 동일시
    

⇒ 둘 다 dehatebert에서 97% 확신하며 non-hate 판정했지만 v_AB/v_AC는 hate로 잡음

### Random 1,000 seed: 방향 특이성 검증

Random vector가 우연히 좋은 방향으로 뽑힌 것이 아니라는 점을 보여주기 위해, B1 random vector를 1,000개 seed로 반복한 결과를 appendix에 넣는다.

!13_random_seed_1000.png

발표에서는 “같은 크기의 random vector를 1,000번 반복해도 E1/E2를 넘지 못했다. 따라서 개선은 단순한 perturbation이 아니라 minimal-pair 방향 정보의 효과다”라고 말한다.

### Cosine similarity: AB와 AC는 완전 동일하지 않지만 많이 겹친다

AB와 AC를 함께 주입했을 때 결합 이득이 크지 않은 이유를 설명하기 위한 appendix 자료다.

| layer | AB_AC | AB_harm | AC_harm | AB_random |
| --- | --- | --- | --- | --- |
| 8 | +0.708 | +0.465 | +0.333 | +0.020 |
| 13 | +0.710 | +0.485 | +0.345 | −0.010 |
| 14 | +0.725 | +0.492 | +0.346 | +0.008 |
| 27 | +0.638 | +0.560 | +0.362 | +0.016 |

!06_cosine.png

해석:

- `v_AB`와 `v_AC`는 L13~14에서 cos≈0.71로 꽤 겹치지만 완전히 같은 방향은 아니다.
- `v_AB/v_AC`와 `v_harm`의 similarity는 더 낮아, minimal-pair vector가 generic harm 축의 단순 부분집합은 아니다.
- AB와 AC를 합쳤을 때 결합 이득이 제한적인 이유는 두 방향이 일부 중복되어 노름 증가 효과가 커지기 때문이다.

### Data efficiency: 128쌍부터 안정화

Minimal-pair 데이터가 많이 필요하지 않다는 실용적 기여를 보여주는 appendix 자료다.

| n (pairs) | macro F1 mean ± sd | ΔF1 vs B0 | vs v_harm |
| --- | --- | --- | --- |
| 32 | 0.6003 ± 0.004 | +0.010 | −0.004 |
| 64 | 0.6032 ± 0.004 | +0.013 | −0.001 |
| **128** | **0.6163 ± 0.002** | **+0.026** | **+0.012** |
| 256 | 0.6189 | +0.029 | +0.015 |

!05_data_size.png

발표에서는 “128쌍, 즉 256문장부터 600문장 기반 v_harm을 안정적으로 넘었다. minimal-pair 구축 비용 대비 효율성이 있다”로 짧게 정리한다.

### Joint injection: 결합 이득은 제한적이다

AB와 AC를 함께 넣었을 때 성능이 더 좋아지는지 확인한 결과다. 이 자료는 “AB와 AC를 합치면 더 좋아지지 않나요?”라는 질문 대응용이다.

!07_joint_heatmaps.png

!image.png

핵심 해석은 “FN recovery는 커질 수 있지만, macro F1 sweet spot은 단일 벡터 쪽에 더 가깝다. 두 방향은 보완재라기보다 일부 중복되는 대체재처럼 작동한다”이다.

## 발표 구조: 논문식 큰 흐름

### 0. 발표 제목 후보

**연구 분야 + 핵심 approach**가 드러나는 phrase

- **Minimal-Pair Steering for Implicit Hate Speech Detection**
- **Improving Implicit Hate Speech Detection via Minimal-Pair Steering**
- **Steering LLM Representations with Minimal Pairs for Implicit Hate Detection**

첫 번째가 가장 간결하다. 발표자료 제목으로는 **Minimal-Pair Steering for Implicit Hate Speech Detection**을 추천한다

### 1. Intro: 문제 사례 제시

첫 장은 숫자보다 사례로 시작한다. Explicit hate는 표면 단서가 강하지만, implicit hate는 slur나 노골적 cue 없이 특정 집단을 공격하기 때문에 baseline과 외부 moderation model도 놓칠 수 있다.

발표 문장:

> Explicit hate는 표면 단서가 비교적 뚜렷하지만, implicit hate는 공격 대상과 암시적 표현이 분리되어 있어 모델이 non-hate로 놓치기 쉽다. 본 연구는 minimal-pair에서 추출한 steering vector를 주입해 이런 false negative를 회복할 수 있는지 검증한다.
> 

### 2. Method: minimal-pair의 역할

방법론 설명은 복잡하게 들어가지 않는다. A/B/C cell의 모든 구축 과정을 말하기보다, 청중이 이해해야 하는 관계만 남긴다.

벡터 추출용 데이터는 **HateXplain → GPT-4o minimal-pair 생성 → 수동 검수**로 만든 Cell A/B/C다.

| Cell | 의미 | 벡터에서의 역할 |
| --- | --- | --- |
| Cell A | target과 harmful cue가 함께 있는 원문 | hate 방향의 기준점 |
| Cell B | cue는 유지하되 target을 제거하거나 바꾼 문장 | `v_AB = A − B`, target 축 |
| Cell C | target은 유지하되 harmful cue를 약화한 문장 | `v_AC = A − C`, cue 축 |

발표 문장:

> A−B는 “누구를 향한 말인가”에 가까운 target 방향이고, A−C는 “어떤 harmful cue가 있는가”에 가까운 cue 방향이다. 우리는 두 방향을 각각 steering vector로 만들고, LLM의 특정 layer hidden state에 주입했다.
> 
- A는 target과 harmful cue가 함께 있는 문장이다.
- B는 cue는 유지하되 target을 제거하거나 바꾼 문장이다.
- C는 target은 유지하되 harmful cue를 약화한 문장이다.
- A−B에서 `v_AB`를 만들고, A−C에서 `v_AC`를 만든다.

### 3. Experiment: 비교군 고정

실험 설계는 표 하나로 끝낸다. 청중이 알아야 하는 비교군은 다섯 개뿐이다.

| 구분 | 의미 | 발표에서의 역할 |
| --- | --- | --- |
| B0 | No steering | 기본 모델이 놓치는 기준점 |
| B1 | Random vector | 그냥 hidden을 미는 효과와 방향 효과를 분리하는 control |
| B2 | v_harm | 일반 혐오 방향과 minimal-pair 방향을 비교하는 baseline |
| E1 | v_AB, target vector | minimal-pair에서 target 축으로 만든 steering |
| E2 | v_AC, cue vector | minimal-pair에서 cue 축으로 만든 steering |

평가셋은 Latent Hatred와 ToxiGen을 메인으로 둔다. 데이터셋 차이는 깊게 해석하지 말고, “서로 다른 분포에서 같은 방향의 효과가 보이는지 확인하기 위해 두 평가셋을 사용했다” 정도로만 말한다.

## Results: 본문 핵심 네 장

교수님 피드백대로 본문은 큰 줄기만 간다. 세부 분석은 appendix로 보낸다.

### Result 1. FN recovery 30~46%

첫 결과 표는 Latent와 ToxiGen의 best setup 표다. 표에서 강조할 숫자는 macro F1보다 FN recovery다.

- Latent Hatred에서 E1/E2는 macro F1을 약 +0.03~0.04 올리고, FN recovery 37~46%를 달성했다.
- ToxiGen에서 E1/E2는 macro F1을 약 +0.07 올리고, FN recovery 약 30% 수준을 보였다.
- 통계적으로도 McNemar p<0.005, bootstrap 95% CI가 0을 포함하지 않았다.

발표 문장은 이렇게 잡는다.

> 효과 크기를 macro F1로만 보면 작아 보일 수 있지만, 실제로는 baseline이 놓친 implicit hate의 30~46%를 회복했다. 이 연구의 핵심 성과는 바로 이 FN recovery다.
> 

### Result 2. Mid-layer L13~14

Graph A는 레이어별 성능 곡선으로 보여준다. 여기서 말할 핵심은 “후반 레이어에 hate representation이 강하다고 해서, 그곳이 steering에 좋은 위치는 아니다”이다.

- Latent와 ToxiGen 모두 E1/E2는 L13~14 근처에서 가장 안정적으로 작동한다.
- 후반 레이어 L20~27에서는 효과가 약해진다.
- 수정된 배경 그래프에서는 각 평가셋별 harm strength를 따로 계산했으므로, 이전 방법론 오류를 보완했다.

발표 문장은 이렇게 잡는다.

> Hate-related representation 자체는 후반 레이어에서 강해질 수 있지만, 실제로 개입이 잘 먹히는 위치는 중간 레이어였다. 즉 표상이 가장 강한 지점과 steering이 가장 효과적인 지점은 다르다.
> 

### Result 3. Random 대비 방향 특이성

Random vector 1,000번 반복 실험은 본문에 짧게 넣어도 좋다. 복잡한 통계 설명보다 시각적으로 “random 분포 위에 E1/E2 선이 따로 올라가 있다”는 그림이 설득력이 있다.

말할 문장은 하나다.

> 같은 크기의 random vector를 1,000번 뽑아도 E1/E2를 넘지 못했다. 따라서 성능 향상은 단순히 hidden state를 흔든 효과가 아니라 minimal-pair에서 얻은 방향 정보의 효과로 볼 수 있다.
> 

### Result 4. Last-token과 all-token trade-off

주입 위치 결과는 “우리가 왜 last-token을 본문 기준으로 삼는가”를 설명하는 역할이다.

- last-token 주입은 macro F1과 precision/recall 균형이 가장 좋다.
- all-token 주입은 FN recovery를 최대 82%까지 올릴 수 있지만 FP가 함께 증가한다.

발표 문장은 이렇게 잡는다.

> Last-token steering은 균형 잡힌 분류 성능을 위한 설정이고, all-token steering은 놓치는 hate를 최대한 줄이는 공격적인 설정이다. 두 설정은 우열이 아니라 precision-recall trade-off로 해석해야 한다.
> 

## Discussion: 예상 질문 방어

### ToxiGen에서 v_harm이 가장 높았던 결과는 이렇게 말한다

예전에는 이 부분이 가장 위험한 지점이었지만, SST-2 sanity check를 했기 때문에 방어가 가능하다.

말할 흐름은 다음과 같다.

1. ToxiGen에서는 v_harm L=1이 가장 높은 macro F1을 보였다.
2. 그런데 이 세팅은 SST-2에서 일반 sentiment 능력을 크게 손상시켰다.
3. 특히 probe 기준 positive class accuracy가 크게 무너지고 predicted positive rate도 크게 흔들렸다.
4. 따라서 v_harm L=1의 높은 점수는 targeted hate detection 개선이라기보다 early-layer 과개입에 따른 불안정한 효과일 수 있다.

발표 문장은 이렇게 잡는다.

> ToxiGen에서 v_harm이 더 높았다는 사실은 그대로 보고한다. 다만 추가 sanity check 결과, 그 best setting은 일반 sentiment 판단을 크게 훼손했다. 그래서 우리는 단순 최고 점수보다, 성능 향상과 일반 능력 보존 사이의 균형을 함께 봐야 한다고 해석한다.
> 

### AB와 AC를 합쳤을 때 결합 이득이 없는 이유는 이렇게 말한다

이 질문은 교수님이 이미 예상한 질문이므로 appendix에 cosine과 joint heatmap을 준비해둔다.

- v_AB와 v_AC는 L8~20에서 cosine similarity 약 0.71이다.
- 완전히 같은 방향은 아니지만, 독립적인 두 축이라고 보기에는 많이 겹친다.
- 그래서 단순히 더하면 새로운 정보가 더해지기보다 노름이 커져 precision이 깎인다.

발표 문장은 이렇게 잡는다.

> 두 벡터는 45도 정도 떨어져 있어 완전히 같지는 않지만, 상당히 겹친다. 그래서 결합은 새로운 정보를 더한다기보다 같은 방향 성분을 과하게 미는 효과가 커졌고, macro F1 기준으로는 단일 벡터가 더 안정적이었다.
> 

### L13~14 근거는 intermediate layer 레퍼런스로 설명한다

이 부분은 “가설”이라고만 말하기보다, 최신 LLM 레이어 연구와 우리 실험 결과가 같은 방향을 가리킨다고 설명한다. 다만 특정 회로를 규명했다고 단정하지는 않는다.

말할 흐름은 다음과 같다.

- Skean et al. (2024), *Does Representation Matter? Exploring Intermediate Layers in Large Language Models*는 final layer가 항상 downstream task에 최선이 아니며, **intermediate layers가 더 informative할 수 있다**고 보고한다.
- Skean et al. (2025), *Layer by Layer: Uncovering Hidden Representations in Language Models*도 intermediate layers가 final layer보다 더 rich한 정보를 보존할 수 있고, layer별로 정보 보존과 압축의 trade-off가 있다고 본다.
- Jawahar et al. (2019), Tenney et al. (2019), Niu et al. (2022) 계열의 BERTology 연구는 언어 모델 내부 정보가 layer-wise로 다르게 분포한다는 배경을 제공한다. 단, “의미 정보는 무조건 어느 층”처럼 단순화하지 않고, 정보 유형이 레이어별로 달라진다는 정도로만 안전하게 사용한다.
- 우리 결과에서 cross-domain probe peak와 steering best가 L12~14 근처로 맞았기 때문에, L13~14는 임의로 고른 레이어가 아니라 **정보가 읽히면서도 개입이 가능한 intermediate layer**로 제시할 수 있다.

발표 문장은 이렇게 잡는다.

> 선행 연구에서도 LLM의 final layer가 항상 downstream task에 최선은 아니며, intermediate layers가 더 informative할 수 있다고 보고된다. 우리 실험에서도 cross-domain probe peak와 steering best가 L12~14 근처로 맞았기 때문에, L13~14는 임의의 sweet spot이 아니라 선행 연구의 intermediate-layer 관찰과 정합적인 개입 지점으로 해석한다. 다만 특정 회로를 규명한 것은 아니므로, causal mechanism 자체는 후속 분석으로 남긴다.
> 

## Appendix는 질문 대응용으로 구성한다

본문에서 다 말하지 말고, 교수님이 질문하면 바로 꺼낼 수 있게 appendix 순서를 정리한다.

### Appendix A. 데이터와 split 검증

- minimal-pair 데이터는 vector extraction에만 사용했다.
- Latent/ToxiGen은 evaluation 전용으로 분리했다.
- HateXplain 출처와 평가셋 간 leakage guard를 확인했다.
- 질문 대응 문장: “같은 데이터로 vector를 만들고 평가한 것이 아니라, 구축 데이터와 평가 데이터를 분리했습니다.”

### Appendix B. 수정된 Graph A

- Latent용 harm background와 ToxiGen용 harm background를 따로 계산한 그래프를 둔다.
- 질문 대응 문장: “이전 그래프는 같은 background를 공유한 문제가 있어서, 피드백 이후 평가셋별로 다시 계산했습니다.”

### Appendix C. v_harm L=1 sanity check

- SST-2 probe 결과와 zero-shot 결과를 둔다.
- 질문 대응 문장: “v_harm L=1은 ToxiGen 점수는 높지만 일반 sentiment 판단을 크게 손상시켜, 좋은 steering이라고 해석하기 어렵습니다.”

### Appendix D. Random 1,000 seed 검증

- random distribution과 E1/E2 선을 보여준다.
- 질문 대응 문장: “같은 크기의 random vector를 반복해도 E1/E2를 넘지 못했기 때문에 방향 특이성이 있습니다.”

### Appendix E. Cosine similarity와 joint injection

- AB/AC cosine 약 0.71 그래프를 둔다.
- joint heatmap을 함께 둔다.
- 질문 대응 문장: “AB와 AC는 독립 축이라기보다 상당히 겹치는 두 방향이라, 단순 결합 이득이 제한적입니다.”

### Appendix F. Layer probe와 probe swap

- cross-domain probe peak와 steering best가 L12~14로 맞는 결과를 둔다.
- 질문 대응 문장: “steering이 빈 레이어를 건드린 것이 아니라, cross-domain으로 hate signal이 읽히는 레이어에서 작동했습니다.”

### Appendix G. 정성평가 사례

- B0와 dehatebert가 놓쳤지만 v_AB/v_AC가 회복한 사례를 둔다.
- 발표 앞부분에 한두 개만 보여주고, 나머지는 질문 대응용으로 둔다.

## 실제 발표 흐름 초안

### 1분: 문제 제기

Implicit hate는 명시적 욕설 없이 특정 집단을 공격하기 때문에 모델이 놓치기 쉽다. 실제로 baseline과 외부 moderation model도 놓치는 사례가 있었다. 그래서 우리는 minimal-pair에서 만든 steering vector로 모델 내부 representation을 조정하면 이런 false negative를 회복할 수 있는지 물었다.

### 2분: 방법

HateXplain을 바탕으로 GPT-4o minimal-pair를 만들고 수동 검수했다. A/B/C 구조에서 target 축 `v_AB`와 cue 축 `v_AC`를 만들었다. 비교군은 no steering, random, generic harm vector, 그리고 두 minimal-pair vector다. 평가는 Latent Hatred와 ToxiGen에서 했다.

### 3분: 메인 결과

두 minimal-pair vector는 두 평가셋에서 baseline을 일관되게 개선했다. 특히 Latent에서는 B0가 놓친 hate의 37~46%를 회복했다. ToxiGen에서도 약 +0.07 macro F1 향상이 있었고, 통계적으로 유의했다. 이 결과는 random 1,000번 반복에서도 재현되지 않아 방향 특이적이다.

### 4분: 레이어 결과

효과는 후반 레이어가 아니라 L13~14 부근의 mid-layer에서 가장 강했다. 평가셋별 harm background를 따로 계산해도, steering best와 representation strength peak가 항상 같지는 않았다. Cross-domain probe에서는 probe peak와 steering best가 L12~14로 맞아, 이 레이어가 정보가 읽히면서도 개입 가능한 구간이라는 해석을 보조한다.

### 5분: 피드백 이후 보강 실험

ToxiGen에서 v_harm이 가장 높았던 결과는 그대로 보고하되, SST-2 sanity check로 해당 세팅이 일반 sentiment 능력을 크게 손상시킴을 확인했다. 그래서 단순 최고 점수보다 안정적인 개입인지 함께 봐야 한다. 또한 train/test split, 추가 평가셋, 정성 사례, cosine/joint 분석을 정리해 예상 질문에 답할 수 있게 했다.

### 6분: 결론

Minimal-pair steering은 implicit hate 탐지에서 baseline이 놓친 false negative를 실질적으로 회복한다. 효과는 mid-layer에서 가장 안정적이고, random vector로 설명되지 않는다. 다만 v_harm과의 비교, AB/AC 결합 이득 부재, 단일 모델 한계는 남아 있어 후속 연구에서 확장해야 한다.

## 발표에서 절대 길게 말하지 않을 것

- ToxiGen이 GPT 생성이라서 어떻고 Latent가 인간 라벨이라서 어떻다는 식의 데이터셋 기원 해석은 깊게 가지 않는다.
- L13~14의 메커니즘을 확정적으로 말하지 않는다.
- AB/AC 중 어느 것이 “진짜”인지 과하게 주장하지 않는다. 둘 다 효과가 있고, 결합 이득은 제한적이라고 말한다.
- 모든 보조 실험을 본문에서 설명하지 않는다. 본문은 FN recovery, mid-layer, random 검증, v_harm sanity check까지만 둔다.

## 최종 요약

<aside>
✅

발표는 **“implicit hate의 false negative를 minimal-pair steering으로 회복한다”**는 메시지 하나로 정리합니다. 본문에서는 FN recovery 30~46%, mid-layer L13~14, random 1,000 seed 검증, v_harm sanity check만 말하고, 나머지 cosine·probe·data efficiency·joint injection·정성 사례는 appendix에서 질문 대응용으로 사용합니다. ToxiGen에서 v_harm이 더 높았던 약점은 숨기지 않고, SST-2 sanity check를 통해 “높은 점수가 항상 좋은 steering은 아니다”라는 식으로 안전하게 해석합니다.

</aside>