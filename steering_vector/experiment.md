<aside>
📌

**이 실험은 왜 하는가 — 절대 까먹지 말 것**

우리는 *원인 규명*을 하는 게 아니다. cue와 sentiment가 다르다, harmfulness 축이 따로 있다는 것은 이미 보여줬다. **이제부터의 목표는 단 하나, 그 발견을 써먹어서 implicit hate 탐지 성능을 실제로 올리는 것이다.** 데이터 구축 방법론(Cell A/B/C)이 단순 분석 도구가 아니라 **성능 향상의 재료**임을 숫자로 증명한다.

</aside>

## 0. 한 문장 요약

> 0528 미팅에서 확인된 비대칭이 출발점이다. 외부 hate 분류기(dehatebert)는 Cell A를 P(hate)=0.795로 잡지만 Cell C는 0.038로 떨어뜨린다(미팅노트 §4). v_harm 축은 late layer에서 명확히 존재(|z|=1.462, 미팅노트 §2-2)하는데도 implicit hate에서는 이 축이 활용되지 않는다. 이 갭을 메우기 위해, A-B 쌍(cue 고정·target 변화)과 A-C 쌍(target 고정·cue 변화)에서 뽑은 스티어링 벡터를 harm-dominant 레이어에 주입한다. 어느 축·레이어·강도가 implicit hate macro F1과 B0 FN recovery를 가장 크게 올리는지를 실험적으로 결정한다.
> 

## 1. 이번 실험의 가설은 무엇인가

교수님 피드백의 핵심은 "가설과 방법론이 따로 놀고 논지가 흐려진다"는 것이었다. 그래서 이번에는 가설을 한 줄로 박아두고 그 한 줄만 본다.

### 메인 가설

모델 내부에는 **유해성(harmfulness)을 담당하는 방향**이 존재한다(0528 미팅 §2-2·§3·§4에서 표상·외부 분류기 양쪽으로 확인). 이 방향은 explicit hate에서는 활성화되지만 implicit hate에서는 잘 활성화되지 않는다(dehatebert P(hate) 0.80 → 0.04). 추론 시점에 minimal-pair에서 뽑은 스티어링 벡터를 이 방향으로 강화 주입하면 implicit hate 탐지 성능, 특히 baseline이 놓치는 케이스에서의 회복률이 올라간다.

### 하위 가설 두 개를 갈라서 본다, 그리고 prior에 따라 우선순위가 다르다

- **H1 (target 축 가설, 메인 베팅)**: A−B 차이(A vs B에서 cue 고정·target 변화 → target 효과)를 강화하면 implicit hate 탐지가 좋아진다. 즉 "누구를 향한 말인가"가 핵심 신호. 0528 미팅 prior: target 효과 Cohen's d ≈ 1.13.
- **H2 (cue 축 가설, 보조 베팅)**: A−C 차이(A vs C에서 target 고정·cue 변화 → cue 효과)를 강화하면 implicit hate 탐지가 좋아진다. 즉 "표현 강도 신호"가 핵심. 0528 미팅 prior: cue 효과 Cohen's d = 0.296.

두 벡터의 효과 차이가 그 자체로 논문의 메인 finding이 된다. target 효과가 cue 효과의 약 4배라는 표상 수준 prior가 성능 수준에서도 재현되는지를 본다.

## 2. 교수님 피드백에서 가져갈 원칙

실험을 끌고 가는 동안 흔들릴 때마다 돌아올 기준점이다.

<aside>
⚠️

- **수렴부터, 발산은 금지.** 새로운 분석을 추가하고 싶어질 때마다 "이게 성능 숫자 한 줄에 기여하나?"를 먼저 묻는다. 답이 "아니"면 안 한다.
- **원인 분석은 끝났다.** 왜 그런지가 아니라 *그래서 얼마나 좋아지는지*만 본다. 해석은 결과 표가 나온 다음에.
- **평가 기준은 실험 시작 전에 박아둔다.** 평가셋과 메트릭은 §3에서 확정하고 그 뒤로는 그것만 본다.
- **모델은 Llama-3.2-3B 하나로 집중.** 다른 모델로 확장은 성능 결과가 나온 뒤 여유가 있을 때만.
- **3주 안에 "베이스라인 대비 +Δ F1" 표 한 장 + 레이어 sweep 곡선 한 장을 만들어 온다.** 이게 최우선 산출물.
</aside>

## 3. 평가 세팅은 실험 시작 전에 잠가둔다

실험이 발산하는 가장 큰 원인은 "중간에 평가 기준을 바꾸는 것"이다. 그래서 평가셋·지표를 가장 먼저 결정하고, 그 뒤로는 절대 바꾸지 않는다.

<aside>
🪜

**스테이징 — 1단계는 Latent Hatred 단독으로 잠근다**

라벨 신뢰도가 가장 높은(인간 작성·전문가 3-class 라벨) Latent Hatred로 메인 표·곡선·다음 미팅 보고까지 먼저 완성한다. ToxiGen-HumanVal은 GPT-3 생성에 `toxicity_ai` 임계값 매핑이 한 단계 더 들어가고, 행동 수준 LD가 Cell C/HateXplain과 반대 방향이라는 특이성(미팅노트 §5-4: ToxiGen LD +0.372)도 있다. 그래서 ToxiGen은 **1단계 결과가 안정된 뒤 2단계 확장(외부 일반화 보강)** 으로 미룬다. 본 §3·§5·§6·§8은 1단계 기준으로 읽고, ToxiGen 관련 항목은 "2단계"로 라벨링되어 있다.

</aside>

### 평가셋 구성

- **메인 평가셋 1 — Latent Hatred / Implicit Hate Corpus (ElSherief et al., EMNLP 2021).** 실제 Twitter 기반 implicit hate에 명시적 라벨이 붙어 있어 가설과 정확히 정렬된다. hate / non-hate 균형 맞춰 약 2,000건 샘플.
- **(2단계 확장) 외부 일반화 보강 — ToxiGen-HumanVal (Hartvigsen et al., ACL 2022).** GPT-3로 생성된 implicit hate에 human annotation이 붙어 있어, slur·욕설 없는 implicit hate를 명시적으로 다룬다. `annotated` split에서 `toxicity_ai ≥ 4` → hate, `≤ 2` → non-hate, `== 3`은 제외. 13개 minority group stratified 약 1,500건. Latent Hatred(소셜미디어 구어체)와 도메인·문체가 달라 두 평가셋 사이 일관성이 확인되면 모델 내부 처리의 구조적 특성으로 해석 가능(미팅노트 §5-3에서 Cell C와 Spearman ρ=0.93으로 궤적 일치 확인). **1단계 sweep이 안정된 뒤 추가한다.**
- **In-domain sanity check — HateXplain.** v_harm 벡터를 여기서 뽑았으므로 **leakage guard 통과한 held-out split만** 소량 사용. 메인 평가 자리가 아니라 "in-domain에서 무너지지 않는지" 확인용.
- **Cell A/B/C는 평가셋이 아니라 *스티어링 벡터 추출용 학습 자원*으로만 쓴다.** 같은 데이터로 벡터 만들고 평가하면 점수가 부풀려진다.
- **공통 leakage guard**: HateXplain 출처 문장은 모든 평가셋에서 제거. ToxiGen은 별개 소스(GPT-3 생성)라 자연 분리되지만, Llama-3.2 사전학습 노출 가능성은 한계 섹션에 한 줄로 명시한다.

### 지표는 두 개로 고정한다

- **메인 지표 — Macro F1 on hate vs non-hate.** 클래스 불균형에 robust하고 표준적이라 reviewer 설득에 유리.
- **보조 지표 — Hate class recall.** "놓치는 implicit hate를 얼마나 줄였는지"가 우리 메시지의 핵심이라 별도로 본다.

다른 지표(AUC, precision-recall 곡선 등)는 본문에 넣지 않는다. 부록으로만.

### 분석 단위는 두 층으로 분리한다

0528 미팅 §4에서 외부 dehatebert가 Cell C P(hate)=0.038로 implicit hate를 거의 non-hate로 분류한다는 점이 확인됐다. 우리 B0 baseline에서도 implicit hate 상당 비율이 놓일 가능성이 높으므로, 평가를 두 층으로 본다.

- **전체 평가셋**: 메인 표·그래프는 전체 macro F1 / hate recall로 본다.
- **B0 false-negative subset**: B0가 implicit hate로 분류 못 한 케이스만 따로 떼서, 스티어링이 그중 몇 %를 회복시키는지(FN recovery rate)를 같은 표의 별도 컬럼으로. 이 subset 인덱스는 1주차에 평가셋과 함께 저장해 sweep 중에 절대 재생성하지 않는다.

메시지가 "macro F1을 +Δ 올렸다"에서 "harm 표상은 분명히 있는데 implicit hate에선 못 쓰는 그 비대칭을, minimal-pair 벡터가 깨운다"로 바뀐다.

### Baseline은 세 개

- **(B0) No steering**: 개입 없이 Llama-3.2-3B 마지막 토큰 hidden → linear probe로 hate 분류.
- **(B1) Random direction steering**: 같은 노름을 가진 무작위 벡터로 같은 위치·강도에 주입. 스티어링 자체의 효과가 아니라 *방향의 의미*가 중요함을 보여주는 control.
- **(B2) v_harm (HateXplain에서 뽑은 기존 방향) steering**: Cell A/B/C 데이터를 안 쓴, 우리가 이미 갖고 있는 harmfulness 축. "minimal pair 데이터가 추가 가치가 있는가"를 보여주는 비교점.

## 4. 스티어링 벡터를 어떻게 만들 것인가

### 두 가지 벡터 정의

각 레이어 ℓ에서, PASS 처리된 쌍들의 mean-pooling hidden을 평균낸 뒤 차이를 취한다.

```python
v_AB(ℓ) = normalize( mean(h_A(ℓ)) - mean(h_B(ℓ)) )   # target 축 (A vs B: cue 고정, target 변화)
v_AC(ℓ) = normalize( mean(h_A(ℓ)) - mean(h_C(ℓ)) )   # cue 축    (A vs C: target 고정, cue 변화)
```

A−B는 "타겟이 있을 때 추가로 켜지는 방향"(target 축), A−C는 "explicit cue가 있을 때 추가로 켜지는 방향"(cue 축)이다. 두 벡터를 분리해서 본다는 것 자체가 minimal-pair 데이터의 기여를 가시화하는 장치다. 0528 미팅 prior상 target 효과(d≈1.13)가 cue 효과(d=0.296)보다 약 4배 크므로, **v_AB가 implicit hate 탐지에 더 크게 기여할 것**이라 미리 예측한다.

### 주입 방식

forward hook으로 해당 레이어 출력의 마지막 토큰 hidden에 `+α·v`를 더한다. 모델 파라미터는 건드리지 않는다.

```python
def steer_hook(module, inp, out):
    out[0][:, -1, :] = out[0][:, -1, :] + alpha * v
    return out

handle = model.model.layers[L].register_forward_hook(steer_hook)
# 평가셋 forward
handle.remove()
```

### 학습-평가 데이터 분리는 엄격하게

벡터를 뽑는 Cell A/B/C와 평가셋(Latent Hatred 등)은 **출처가 다르므로 자연스러운 OOD 평가**가 된다. 추가로 HateXplain 출신 문장은 leakage guard로 평가셋에서 모두 제거한다.

## 5. 실험 매트릭스

### Sweep할 차원은 세 개

- **벡터 종류**: {v_AB, v_AC} (메인 비교) + {random, v_harm 기존} (control)
- **레이어 ℓ**: 0528 미팅 §6 axis attribution에서 harm-dominant로 찍힌 [4, 5, 9, 10, 11, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]. 우선순위는 **late(20–27) → mid(13–14, 18–19) → early(4–5, 9–11)** 순. 근거: late layer에서 harm |z|=1.462로 sent |z|=1.134를 넘고 v_harm gap이 가장 명확히 벌어진다(미팅노트 §2-2). 또 그 구간이 logit lens가 가장 흔들리는 구간과 겹친다(미팅노트 §6). "유해성 벡터가 가장 크게 튀는 지점"이 곧 "implicit hate에선 그게 안 작동해서 LD가 흔들리는 지점"이라는 점이 곧 우리 개입 지점이다. coarse sweep 후 best 주변을 fine sweep.
- **강도 α**: {0.5, 1, 2, 4} 거친 sweep → best 주변에서 {0.25 간격} fine sweep. 부호도 ±로 검증("미는 방향"이 맞는지 확인).

### 메인 비교 표는 이런 모양이 된다

**1단계 메인 표는 Latent Hatred 단독으로 완성한다.** 아래 빈칸 양식을 그대로 채우는 게 다음 미팅 산출물의 본체다. ToxiGen-HumanVal에 대한 동일 표는 2단계에서 추가하고, 그때 그룹별 hate recall 미니 표를 부록으로 함께 붙인다.

| Setup | Best layer | Best α | Macro F1 (전체) | Hate Recall | FN recovery | vs B0 Δ |
| --- | --- | --- | --- | --- | --- | --- |
| B0 No steering | — | — | ? | ? | — | 0 |
| B1 Random | ? | ? | ? | ? | ? | ? |
| B2 v_harm (HateXplain) | ? | ? | ? | ? | ? | ? |
| **E1 v_AB (target 축, 메인)** | ? | ? | ? | ? | ? | ? |
| **E2 v_AC (cue 축, 보조)** | ? | ? | ? | ? | ? | ? |

E1 또는 E2가 B0/B1/B2를 의미 있게 이기면 "minimal pair 데이터로 만든 스티어링 벡터가 implicit hate 탐지에 기여한다"는 청구가 성립한다.

### 메인 그래프는 두 장으로 끝낸다

- **그래프 A — 레이어별 성능 곡선**: x축은 주입 레이어, y축은 macro F1. v_AB·v_AC·random 세 선을 같은 그림에. **1단계는 Latent Hatred 한 장만**, 2단계에서 ToxiGen 동일 양식 한 장을 추가해 두 평가셋 사이 봉우리 위치 일치 여부를 본다.
- **그래프 B — α 강도 곡선**: best layer에서 α를 sweep했을 때의 F1. 효과의 dose-response를 보여주는 그림.

## 6. 3주 일정은 이렇게 끊는다

### 1주차는 파이프라인을 잠근다

실험이 망가지는 두 번째 원인은 "평가 파이프라인이 마지막 주에 버그가 나는 것"이다. 첫 주는 결과 욕심을 버리고 파이프라인부터 단단히 만든다.

- **eval_latent_v1 구축 — Latent Hatred** 다운로드, 라벨 정리, hate / non-hate 균형 약 2,000건 샘플 확정. 한 번 픽스한 split을 끝까지 쓴다.
- eval_latent_v1에 HateXplain 출처 leakage guard 적용. 길이·라벨 비율 히스토그램 한 장 뽑아두기.
- 메트릭 코드 픽스: `macro_f1`, `hate_recall` 두 개만 반환하는 단일 함수로 묶기.
- B0 (no steering) baseline을 **eval_latent_v1에서** 측정. 이 숫자가 1단계 모든 비교의 0점.
- 한 레이어에서 **v_AB**(메인 베팅인 target 축) 스티어링이 forward hook으로 제대로 들어가는지 sanity check. F1이 베이스 대비 어떻게든 움직이기만 하면 OK.
- **eval_latent_v1에서 B0 false-negative subset 인덱스 파일을 저장한다.** 이후 sweep에서 매번 재생성하지 않고 같은 인덱스를 끝까지 쓴다.
- 산출물: "파이프라인 동작 확인용 mini result" 표 한 줄 + eval_latent_v1 B0 숫자 + eval_latent_v1 B0 FN subset 크기.
- **(2단계 준비)** ToxiGen-HumanVal 다운로드·매핑·샘플링은 1단계 sweep이 끝난 뒤 시작. 1주차에는 손대지 않는다.

### 2주차는 본 sweep을 돌린다

- v_AB, v_AC 두 벡터 × 17개 레이어 × α 4단계 = 약 136 셋업. **모두 eval_latent_v1(Latent Hatred) 위에서만** 돌린다. 하나당 평가셋 forward는 가벼우므로 하루~이틀로 끝남.
- random·v_harm baseline도 같은 sweep grid로 측정해 비교 가능하게.
- best 셋업 주변 fine sweep.
- 산출물: 1단계 메인 표 한 장(Latent Hatred 단독) + 두 그래프(레이어별·α별).
- **(2단계 확장, 시간 여유 시 2주차 후반~3주차 초)** eval_toxigen_v1 구축 → 동일 sweep grid로 ToxiGen-HumanVal 위에서 재측정. 그룹별 hate recall도 같이 뽑아 H1(target 축 v_AB) 검증 보강. 단, ToxiGen은 행동 수준 LD가 Cell C/HateXplain과 반대 방향이라(미팅노트 §5-4: ToxiGen LD +0.372, Cell C −0.342) **표상 수준 sweep(probe macro F1)을 메인 지표로** 보고 LD 흔들림은 부수 관찰로만 기록한다.

### 3주차는 글쓰기와 ablation에 쓴다

- 결과 해석을 본문에 녹인다. "왜 그 레이어인지"는 axis attribution 그림과 자연스럽게 연결만 시키고, 새 분석은 더 추가하지 않는다.
- 최소한의 ablation: α=0 control, 부호 반전(−α), 무작위 토큰 위치에 주입한 control 한두 개. 모두 메인 표의 부록 라인으로.
- 한계와 future work 한 단락. "인과 주장은 단일 모델 단일 평가셋 수준"이라는 점을 미리 명시해 reviewer 공격 포인트를 줄인다.

## 7. 무엇을 하지 않을 것인가

이 줄을 적어두는 이유는, 발산을 막기 위해 *유혹 리스트*를 미리 박제하는 것이다.

- 새로운 cause-analysis 분석은 추가하지 않는다 (attention 가설, double dissociation 등은 future work로만).
- Llama-3.2-3B 외 다른 모델로 확장하지 않는다 (성능 표가 다 나온 뒤 여유 있으면 그때).
- 새로운 평가셋을 중간에 추가하지 않는다 (Latent Hatred + ToxiGen-HumanVal + HateXplain sanity check로 끝).
- 새로운 데이터(Cell D 등) 구축하지 않는다.
- v_sent와 v_harm의 직교성, attention 분포 분석 등 "분리" 주장 보강용 분석은 본문에 포함하지 않는다(0528 미팅 §8 미해결 1·5번 그대로 future work).
- dehatebert 외 추가 외부 hate 분류기로 비교를 확장하지 않는다(0528 §4에서 motivation 한 줄로 충분).

## 8. 다음 미팅에 들고 가야 하는 것

3주 뒤가 아니라 *다음 미팅* 기준. 교수님이 "성능 향상 결과"를 명시적으로 요구하셨다.

- **eval_latent_v1 (Latent Hatred) 평가셋 확정과 B0 baseline macro F1 한 개** (1단계 필수, 무조건).
- **v_AB**(target 축, 메인 베팅) 한 레이어 한 강도 셋업의 mini result 한 줄 + 같은 셋업의 FN recovery rate 한 줄 (eval_latent_v1 기준).
- ToxiGen-HumanVal은 "2단계 확장 예정"으로 한 줄만 언급한다. 1단계 결과 안정화 후 추가.
- 미팅 슬라이드 앞장에 dehatebert P(hate) **0.80 → 0.04** (0528 미팅 §4) 비대칭을 problem statement로 한 줄. "표상에는 있는데 모더레이션 모델조차 못 잡는다"가 이 실험의 출발점임을 명시.
- 위 §5 메인 표의 빈칸 양식과 §6의 일정표를 같이 들고 가서 "이대로 가겠다"는 동의를 받는다.

---

### 평가셋 구축 체크리스트 — eval_latent_v1 / eval_toxigen_v1

<aside>
🎯

**이 페이지의 목적**

스티어링 벡터 실험의 §3(평가 세팅 잠그기)을 실행하기 위한 **작업 로그 + 체크리스트**. 평가셋이 잠긴 뒤로는 split·라벨 매핑·메트릭을 **절대 바꾸지 않는다**.

**스테이징**: 1단계는 §1(eval_latent_v1 — Latent Hatred) + §3(sanity_hatexplain) + §4 leakage guard 중 eval_latent_v1·sanity 항목 + §5 분포 점검 중 eval_latent_v1 항목 + §6 메트릭 코드 + §7-1·7-2의 eval_latent_v1·sanity B0 측정만 마치고 v_AB·v_AC 추출과 sweep으로 넘어간다. §2(eval_toxigen_v1) 및 ToxiGen 관련 leakage·분포·B0 항목은 1단계 sweep이 끝난 뒤 2단계로 미룬다.

</aside>

## 0. 최종 산출물 정의

이 체크리스트가 끝나면 아래 5개가 존재해야 한다.

| 산출물 | 경로 (예시) | 역할 |
| --- | --- | --- |
| `eval_latent_v1.csv` | `data/eval/eval_latent_v1.csv` | Latent Hatred 메인 평가셋 (약 2,000건) |
| `eval_toxigen_v1.csv` (2단계) | `data/eval/eval_toxigen_v1.csv` | ToxiGen-HumanVal 2단계 확장 평가셋 (약 1,500건, 그룹 stratified). 1단계 sweep 이후 구축. |
| `sanity_hatexplain.csv` | `data/eval/sanity_hatexplain.csv` | HateXplain leakage-guard-통과 held-out (소량, sanity check 전용) |
| `metrics.py` | `src/eval/metrics.py` | `evaluate(preds, labels) -> {macro_f1, hate_recall}` 단일 함수 |
| `b0_baseline.json` | `results/b0_baseline.json` | 세 평가셋에서의 No-Steering baseline 숫자 |

## 1. eval_latent_v1 (Latent Hatred) 구축

<aside>
🔒

**1단계 결정 박제 — 라벨 + 최소 정규화**

1단계 steering 실험에 쓸 Latent Hatred 처리 규칙은 아래 두 줄로 잠그고, sweep이 끝날 때까지 절대 바꾸지 않는다.

1. **라벨 매핑 (이진)**: `implicit_hate` + `explicit_hate` → **hate**, `not_hate` → **non-hate**. `implicit_class` 컬럼은 `subtype`으로만 보존하고 메인 라벨 결정에는 쓰지 않는다.
2. **텍스트 정규화는 최소 4개만**: `strip()`, 연속 공백 → 단일 공백, CSV 이스케이프 잔재 `""` → `"`, `post`/`class` 빈 행 drop. **그 외는 ElSherief 원본 그대로 둔다** — URL·`@mention`·`#hashtag`·punctuation 통일·추가 lowercasing 모두 하지 않는다. 논문 비교 가능성과 원본 보존이 토크나이저 친화성보다 우선.
</aside>

### 1-1. 다운로드 & 원본 정리

- [ ]  [Latent Hatred 공식 repo](https://github.com/SALT-NLP/implicit-hate)에서 raw 다운로드
- [ ]  라이선스 확인 후 사용 가능 컬럼만 추출: `post`, `class`, (선택) `implicit_class`
- [ ]  원본 행 수, 라벨 분포(`explicit_hate` / `implicit_hate` / `not_hate`)를 로그에 기록
- [ ]  **텍스트 최소 정규화만 적용한다** (ElSherief 원본 보존이 우선, 논문과 직접 비교 가능성 유지).
    - `post.strip()` — leading/trailing whitespace 제거
    - `re.sub(r"\s+", " ", post)` — 연속 공백 → 단일 공백
    - CSV 이스케이프 잔재 `""` → `"` (트위터 RT 인용 마크업이 이중 이스케이프되어 잘못 들어온 행이 많다)
    - `post` 또는 `class`가 빈 행 drop
- [ ]  **하지 않을 것**: lowercasing(이미 적용됨), URL 제거, `@mention`·`#hashtag` 제거, punctuation 통일—모두 원본 그대로 둔다. Llama-3.2 BPE 토크나이저가 흡수한다.

### 1-2. 라벨 매핑 규칙 고정

| 원본 class | 우리 label |
| --- | --- |
| `implicit_hate` | **hate** |
| `explicit_hate` | **hate** (단, 분석 시 implicit-only 서브셋 별도 표기) |
| `not_hate` | **non-hate** |
- [ ]  매핑 규칙을 `data/eval/mapping_rule.md`로 박제
- [ ]  implicit-only / explicit-only 서브셋 표시 컬럼 `subtype` 추가

### 1-3. 샘플링 & split 고정

- [ ]  `RANDOM_SEED = 20260528` (또는 실행 일자 고정) 박아두기
- [ ]  hate / non-hate 1:1 균형, 총 약 2,000건
- [ ]  `eval_latent_v1.csv` 컬럼: `id, text, label, subtype, source="latent_hatred"`
- [ ]  **한 번 저장한 뒤로는 절대 재샘플링 금지**

## 2. (2단계) eval_toxigen_v1 (ToxiGen-HumanVal) 구축

<aside>
⏸️

**이 섹션은 1단계 sweep이 끝난 뒤에 시작한다.** 1단계 메인 표·곡선이 다음 미팅에 통과되면 그때 HuggingFace `toxigen/toxigen-data` `annotated` split(gated) 다운로드부터.

</aside>

### 2-1. 다운로드 & 원본 정리

- [ ]  HF 로그인 + `toxigen/toxigen-data` access 요청 통과
- [ ]  `load_dataset("toxigen/toxigen-data", name="annotated")` 로 로드 후 사용 컬럼 추출: `text`, `toxicity_ai`, `toxicity_human`, `target_group`, `intent`
- [ ]  원본 행 수, `toxicity_ai` 분포(1~5), `target_group` 분포 로그에 기록

### 2-2. 라벨 매핑 규칙 고정

ToxiGen의 `toxicity_ai`는 1~5 척도다. 경계값을 제외하고 매핑한다.

| 조건 | 우리 label |
| --- | --- |
| `toxicity_ai >= 4` | **hate** |
| `toxicity_ai <= 2` | **non-hate** |
| `toxicity_ai == 3` (경계) | **제외** |
- [ ]  매핑 규칙은 위 `mapping_rule.md`에 같이 박제
- [ ]  `intent`가 명백히 benign generation인 행은 toxicity 점수와 충돌 시 점수를 우선

### 2-3. 그룹 stratified 샘플링

- [ ]  `target_group`별 카운트 확인 (13개: black, asian, latino, lgbtq, women, jewish, muslim, native_american, mental_dis, physical_dis, mexican, middle_east, chinese 등 데이터셋 정의 그대로)
- [ ]  **그룹 × label** 셀당 동일 수 (예: 그룹당 hate 60, non-hate 60) → 총 약 1,500건
- [ ]  한 그룹이 부족하면 부족분만큼 다른 그룹에서 보충하지 말고 그대로 둠 (왜곡 방지)
- [ ]  `eval_toxigen_v1.csv` 컬럼: `id, text, label, target_group, toxicity_ai, source="toxigen_humanval"`

## 3. sanity_hatexplain 구축

- [ ]  HateXplain raw에서 v_harm 추출에 사용한 split을 **제외**
- [ ]  남은 held-out에서 hate / non-hate 균형 약 500건 샘플
- [ ]  `sanity_hatexplain.csv` 컬럼: `id, text, label, source="hatexplain_heldout"`

## 4. 공통 leakage guard 로그

레이커지가 없다는 것을 **숫자로** 남겨둬야 reviewer 질문에 답할 수 있다.

- [ ]  eval_latent_v1 ∩ Cell A/B/C 텍스트 일치 검사 → 0건 확인
- [ ]  eval_latent_v1 ∩ HateXplain post_id 일치 검사 → 0건 확인 (Latent Hatred는 별개 소스라 자연 분리되지만 확인은 한 번)
- [ ]  eval_toxigen_v1 ∩ HateXplain → 0건 (출처가 다르므로 거의 확실하지만 확인)
- [ ]  sanity_hatexplain ∩ v_harm 추출 split → 0건
- [ ]  결과를 `data/eval/leakage_check.md`로 남기기

## 5. 분포 점검 (구축 직후 1회)

평가셋이 완성됐는지 신뢰할 수 있는지 가벼운 sanity plot으로 확인.

- [ ]  토큰 길이 히스토그램: `eval_latent_v1` vs `eval_toxigen_v1` 같은 그림에. ToxiGen이 평균적으로 더 길게 나올 것 — 차이가 너무 크면 length-controlled subset도 같이 준비 (옵션)
- [ ]  라벨 비율 막대그래프 — 두 셋 모두 1:1 근처여야 함
- [ ]  ToxiGen은 `target_group`별 카운트 막대그래프 추가
- [ ]  그림 3장을 `figs/eval_dist/`에 저장

## 6. 메트릭 코드 픽스 — 한 번만 작성하고 모두 이걸 부른다

```python
# src/eval/metrics.py
from sklearn.metrics import f1_score, recall_score

HATE = 1
NON_HATE = 0

def evaluate(preds, labels):
    """All experiments call ONLY this function."""
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "hate_recall": recall_score(labels, preds, pos_label=HATE),
    }

def evaluate_by_group(preds, labels, groups):
    """For ToxiGen group-stratified analysis."""
    out = {}
    for g in set(groups):
        mask = [gi == g for gi in groups]
        p = [pi for pi, m in zip(preds, mask) if m]
        l = [li for li, m in zip(labels, mask) if m]
        out[g] = {
            "hate_recall": recall_score(l, p, pos_label=HATE) if HATE in l else None,
            "n": len(p),
        }
    return out
```

- [ ]  `metrics.py` 작성 & 단위 테스트 (toy preds/labels로 0/1 모두 통과)
- [ ]  다른 곳에서 절대로 `sklearn` 메트릭을 직접 부르지 않기 — 모든 호출은 `evaluate()` 경유

## 7. B0 (No Steering) baseline 측정

### 7-1. 분류기 정의

Llama-3.2-3B 마지막 레이어 마지막 토큰 hidden → linear probe (logistic regression) → hate/non-hate

- [ ]  학습용 split: HateXplain train (v_harm 추출에 쓴 것과 동일, 분류기 학습용으로만)
- [ ]  검증: `eval_latent_v1`, `eval_toxigen_v1`, `sanity_hatexplain` 세 곳 모두

### 7-2. 측정 & 결과 저장

```python
# results/b0_baseline.json (목표 양식)
{
  "model": "meta-llama/Llama-3.2-3B",
  "probe": "logistic_regression",
  "layer": -1,
  "results": {
    "eval_latent_v1":            {"macro_f1": 0.??, "hate_recall": 0.??, "n": ????},
    "eval_toxigen_v1":    {"macro_f1": 0.??, "hate_recall": 0.??, "n": ????},
    "sanity_hatexplain": {"macro_f1": 0.??, "hate_recall": 0.??, "n": ???}
  },
  "timestamp": "2026-??-??"
}
```

- [ ]  **1단계**: `eval_latent_v1` + `sanity_hatexplain` 두 숫자만 채워서 저장. `eval_toxigen_v1` 자리는 비워둔다.
- [ ]  **이 메인 숫자(`eval_latent_v1` macro_f1)가 1단계 모든 후속 비교의 0점.** ToxiGen은 2단계 확장 시 채운다.

## 8. 다음 미팅 보고 양식

<aside>
📎

다음 미팅에 들고 갈 한 페이지 요약은 아래 세 줄이다.

1. `eval_latent_v1` (Latent Hatred, n=????) 확정, B0 macro F1 = ?.???
2. v_AB(target 축, 메인 베팅) steering hook sanity check: layer L=??, α=??에서 macro F1이 B0 대비 ±?.??? 만큼 움직임 → 파이프라인 동작 확인
3. ToxiGen-HumanVal 확장은 "2단계 예정"으로 명시. 1단계 결과 안정화 후 진행.
</aside>

## 9. 작업 로그

작업하면서 결정·이슈를 한 줄씩 적어두기. 나중에 한계 섹션·reviewer 응답 작성할 때 그대로 인용.

- [ ]  (날짜) Latent Hatred 다운로드 완료, 원본 라벨 분포: implicit=????, explicit=????, not_hate=????
- [ ]  (2단계·날짜) ToxiGen-HumanVal 다운로드 완료, `toxicity_ai==3` 제외 후 사용 가능 행 ????건
- [ ]  (날짜) leakage check 통과 (위 §4 네 항목 모두 0건 확인)
- [ ]  (날짜) 1단계 B0 baseline 측정 완료, `eval_latent_v1` + `sanity_hatexplain` 두 숫자 박제. `eval_toxigen_v1`은 2단계에서 추가.