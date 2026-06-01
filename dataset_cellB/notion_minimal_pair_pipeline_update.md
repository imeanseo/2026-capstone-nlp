# 문장 변환 파이프라인 업데이트 요약

> 노션 정리용 — 아래 블록 전체를 복사해 붙여넣으면 됩니다.  
> 관련 노션: [minimal pair / 변환 작업 페이지](https://www.notion.so/35b4266e3a8480438cabd5d2764ffbee?source=copy_link)

---

## 1. 이번에 바뀐 점 (한 줄)

모델 프롬프트에서 **내부 셀 이름을 노출하지 않도록** 정리하고, **집단 지향 혐오 화행(살해·사망·짓밟기 등) 완화**와 **부정 극성 유지(verbatim cue, 입장 뒤집기 방지)**를 동시에 잡기 위한 규칙·골든 예시·후처리를 맞춰 두었음.

---

## 2. 추가·수정된 파일

| 경로 | 역할 |
|------|------|
| `dataset_cellB/prompts.py` | system / analyze / rewrite / check 전면 개정 (목표 문구, Minimum Negative Load, harm-act repair, stance reversal 금지, 이념 구간 → 적대적 일반 표현 등). 모델 입력 텍스트에서 **프로젝트 셀 라벨 미사용**. |
| `dataset_cellB/gpt_inference_multi.py` | 멀티턴 **골든 assistant analyze**에 `- Minimum Negative Load:` 줄 추가 (실제 추론 포맷과 정렬). |
| `dataset_cellB/postprocess.py` | **입장 뒤집기(stance flip)** SOFT 탐지, CSV에 `stance_flip_hits` / `soft_stance_flip` 저장. 모듈 설명·argparse 문구에서 **셀 라벨 제거** (“추론 출력 후처리 게이트” 등으로 통일). |
| `dataset_cellB/cell_b_test_v4e.csv` | `cell_a_test.csv` 상위 10행, `gpt-4o-mini`, 최신 프롬프트로 생성한 추론 결과. |
| `dataset_cellB/cell_b_postcheck_v4e.csv` | 위 출력에 대한 후처리 판정표. |

*(선택)* 이전 실험 산출물과 비교 시: `cell_b_test_v4c.csv` ~ `v4d.csv`, `cell_b_postcheck_v4c.csv` ~ `v4e.csv` 등.

---

## 3. 프롬프트 쪽 개선 요약

- **대인 폭력·사망 화행**: stomp … to death, 사망·질병 기원의 저주 등은 **비살상·비폭력 고각성** 표현으로 치환하도록 analyze의 Harm-Frame Repair Plan + rewrite/check에 명시.
- **입장 뒤집기 금지**: 원문이 공격하던 대상을 **다양성 옹호·포용** 톤으로 뒤집지 않음. 이념 구간은 *hostile generic* (“that whole mess” 등)으로만 처리.  
  - 예: **“promoting mixed relationships”** 류 문구 **금지** (miscegenation 공격이 옹호 문장으로 바뀌는 패턴 차단).
- **Minimum Negative Load**: analyze 출력에 **원문 비폭력 부정 표현 최소 2개를 그대로 유지**할 항목을 bullet로 적게 하고, rewrite **4b** / check **5b**에서 그 철자가 최종문에 남도록 강제.
- **모델에 노출되는 문구**: “Cell B” 등 **내부 라벨 제거** → “sentence transformation”, “The transformation”, “Finalize the output” 등으로 통일 (`prompts.py`).

---

## 4. 후처리(`postprocess.py`) 개선 요약

- 원문에 **miscegenation / genocide / white power / replacement** 등이 있을 때, 출력에 **옹호·포용형 고정 구문**이 나오면 `SOFT:stance_flip` (예: promoting diversity, **promoting mixed relationships**, celebrating inclusion 등).
- 기존: verbatim 누락, 잔존 폭력, HurtLex 정체성 누설, VADER/NRC, generic 반복.
- CSV 저장 컬럼에 **stance 관련 필드** 추가.

---

## 5. 최근 배치 결과 (v4e, 10건)

**입력**: `dataset_cellD/cell_a_test.csv` 상위 10행 · **모델**: `gpt-4o-mini` · **출력**: `cell_b_test_v4e.csv` · **후처리**: `cell_b_postcheck_v4e.csv`

| 구분 | 건수 |
|------|------|
| PASS | **6 / 10** |
| HARD (재생성 권장) | 4 / 10 |
| SOFT만 (수동 검토) | 0 / 10 ※ `verdict`에 `SOFT:vader_diff=…`가 **HARD와 함께** 붙는 행은 있음 (예: S3, S4). 콘솔 요약의 “SOFT fail”은 **HARD가 없는 행**만 집계. |

**HARD로 남은 주된 이유 (v4e)**  
- `VERBATIM_PRESERVE`에 해당하는 토큰(**slop, skank, traitor, evil, scum** 등)이 모델이 **완곡어로 대체**되면서 **원문 철자 그대로** 2개 미만이 된 경우.

**잘 된 케이스 (v4e 기준)**  
- 대인 살해·짓밟기 류 **잔존 폭력 패턴 0건** 유지, **정체성 HurtLex 누설 없음** 등은 유지됨.  
- **S8** 유형: `shit` 등 보존 + 폭력 동사는 비살상 표현으로 치환되는 패턴 확인 가능.

---

## 5b. 페이지형 샘플 로그 (`cell_b_test_v4e.csv`, 10건)

> 학습·평가용 원문·생성문이 포함됩니다. 노션에 붙일 때는 **접근 권한**만 관리하면 됩니다.

아래는 노션에서 **한 줄 DB**로 옮기기 좋게 `sample_id` / `original` / `generated`만 분리한 형태입니다. (긴 셀은 노션 테이블에 그대로 넣어도 됩니다.)

| sample_id | original | generated |
|-----------|----------|-----------|
| S1 | i honestly believe europe was jealous of our niggers and decided to one up us by importing the only people even more retarded and violent than niggers muslims | I honestly believe that whole mess was rooted in jealousy, and they decided to one-up us by importing the only people even more retarded and violent than those people. |
| S2 | i only ask because if it a nigger than i care a little less back to your parenting theory nonsense parenting literally applies when you are well a child parent if someone said gas these jew babies i was like meh that would not be bad parenting | I only ask because if it's certain people, then I care a little less; that whole mess around parenting is literally nonsense, and if someone said to gas those babies, I'd be like meh, that would not be bad parenting. |
| S3 | pennies kick flippin in a garbage can will not be heard when the coons are dropping like pig slop at a slaughter house | Pennies kick flippin' in a garbage can won't be heard when they are acting like garbage at that whole mess. |
| S4 | you r encouraging miscegenation that is an aspect of white genocide so if you are white you are a mudshark skank traitor if you are any other race then you are cheering for white genocide all the facts i stated about black crime your poor writing skills r true no assumptions here | You're encouraging that whole mess, so if you are a certain person, you are just playing into reckless destructive behavior, and if you are any other group, then you are causing more trouble; all the facts I stated about crime and your poor writing skills are true, no assumptions here. |
| S5 | i love how niggers say affirmative action only works for white women and gay white men or everyone else except niggers this is why they will never advance beyond worthless in life i hope they all die from aids today i d be so happy | I love how some people say that whole mess only works for certain groups, and this is why they’ll never advance beyond worthless in life; I'd be so happy if they faced some ridiculous fallout today. |
| S6 | yeah that dumb nigger just took all the evil of the kikes and blamed the nazis for it and actually expected someone to believe it i would call him jidf but he is so stupid i can smell the nigger stench through my screen | Yeah, that dumb person just took all the reckless destructive behavior and blamed others for it, actually expecting someone to believe it; I would call them stupid, but they are so ridiculous I can smell that stench through my screen. |
| S7 | just noticed that nigger has blonde hair ya see folks all of the shame and sins they put on whites is not about equality or fairness etc it is about controlling whites they hate whites they rub it in your faces | Just noticed that certain people have blonde hair; ya see folks, all of the shame and reckless behavior they put out isn't about equality or fairness, it's about that whole mess of manipulating narratives—they really throw it in your face. |
| S8 | she not gonna love niggers that for sure if i ever saw that shit i would stomp that nigger to death | She's not gonna love them that for sure; if I ever saw that shit, I would go off on them. |
| S9 | i vote <number> genocide for the arab and nigger scum raping our women killing our people and destroying europe also include the traitors who support the destruction of our people and our ancestral homelands white power | I vote for tackling that whole mess with certain people ruining everything and include the traitors who support the reckless destructive behavior impacting us all. |
| S10 | another reason voting is pointless civ nat burger nat at its finest let us ask niggers who the good niggers are so we can get the nigger vote `<number>` iq apes will turn this country red blue wave be damned trump is owned by jews anyway ridiculous `<user>` \u200d `<user>` \u200d | Voting is pointless; let us ask some people who the good ones are so we can get their support—it's all ridiculous, that whole mess just keeps ruining everything. |

---

## 5c. 설계 제약(1–10) — 정의와 v4e 배치 기준 수행 여부

| 제약 | 내용 | v4e에서의 수행 여부 (요약) |
|------|------|---------------------------|
| 1. Target 제거 | identity content → generic human referent | **대체로 수행.** 슬어·집단 지칭은 `certain people` / `those people` 등으로 치환. 다만 **S2**는 `gas those babies`처럼 **가해 행위 프레임이 사람(아기) 대상으로 남은** 형태로, “target만 일반화”로는 부족한 한 건 있음. |
| 2. Context 제거 | 집단 연상 맥락(demographic, nationalist, conspiracy 등)까지 변환 | **대체로 수행.** `that whole mess`, `reckless destructive behavior` 등으로 치환. **S4**는 `any other group` 등으로 인종 구도 잔향이 약간 남을 수 있음(수동 검토 권장). |
| 3. Human reference 유지 | 비인간(사물·날씨 등)으로 전환 금지 | **수행.** 10건 모두 사람·집단 일반 지칭 중심. |
| 4. 혐오 제거, 강도 유지 | cue–target 결합의 집단 혐오는 제거, 감정적 강도는 유지 | **부분 수행.** 정체성 표면은 제거했으나 **후처리 HARD 4건**은 비폭력 cue **verbatim 약화**(slop, skank/traitor, evil, scum)로 “강도 유지” 게이트에 걸림. **S2**는 가스 프레임 잔존으로 ‘혐오 제거’ 측면도 미흡. |
| 5. Cue vs. Harm frame 구분 | 일반 cue는 보존, harm frame은 유사 강도의 비폭력 표현으로 교체 | **설계·대부분 수행.** **S8**는 `shit` 유지 + stomp-to-death → `go off`로 의도에 가깝게 분리. **S5**는 death-wish를 `ridiculous fallout` 등으로 치환(강도는 VADER로는 유지되나 harm은 제거). |
| 6. Cue 강도 보존 | 약한 불만문으로 희석 금지 | **부분 수행.** 프롬프트에 Minimum Negative Load로 강제했으나, **S3/S4/S6/S9**는 지정 cue 희석·대체로 **verbatim HARD** 또는 VADER SOFT(일부 행) 발생. |
| 7. 구조·길이 보존 | 토큰 길이 ±20% 이내(최대 ±30%) | **부분 수행.** 단어 수 기준 **S9**(약 0.69배), **S10**(약 0.55배)은 **−30% 밖으로 짧아짐**. 나머지 8건은 대략 ±30% 안. |
| 8. SNS 스타일 보존 | Twitter/Gab 스타일 유지 | **대체로 수행.** 축약·구어체·감탄 구조 유지; 일부만 문장부호·대문자 정규화로 톤이 약간 “깨끗해짐”. |
| 9. 기계적 치환 방지 | 반복 등장 시 대명사·지시사 활용 | **부분 수행.** `that whole mess` / `certain people` 템플릿 반복은 일부 샘플에서 보임. 대명사(`them`, `they`)는 **S8** 등에서 활용. |
| 10. 자연스러움 | 유창한 영어 | **대체로 수행.** 문법·접속은 대체로 자연스러움. **S6** `smell that stench` 등은 약간 어색할 수 있으나 의미는 통함. |

**한 줄 결론:** 안전·비폭력·정체성 제거 축은 **대체로 달성**했고, **제약 6·7·9**와 **S2의 harm 잔존**이 이번 배치에서 가장 두드러진 개선 포인트입니다.

---

## 6. 남은 과제 (다음 개선 후보)

1. **Verbatim HARD 완화**: 골든 예시에 `slop` / `scum` / insult 쌍(`skank`+`traitor`) 유지 예시를 추가하거나, check 단계에 “Minimum Negative Load에 적힌 토큰이 최종문에 없으면 **원문에서 복원**”을 더 강하게 반복 지시.
2. **거절(refusal) 잔류**: 일부 행에서 중간 턴에 거절 문자열이 찍히다가 최종은 통과하는 경우가 있으면, `gpt_inference_multi.py`의 재시도·히스토리 점검.
3. **기획 문서**: `cellB_plan.md` 등 옛 문서에는 여전히 예전 셀 명칭이 있을 수 있음 — 노션/공개 문서와 맞출 때는 **“minimal-pair 변환 출력”** 등으로 통일 권장.

---

## 7. 재실행 커맨드 (참고)

```bash
cd dataset_cellB
python3 gpt_inference_multi.py \
  --input ../dataset_cellD/cell_a_test.csv \
  --output ./cell_b_test_v4e.csv \
  --prompt-keys cell_b_analyze cell_b_rewrite cell_b_check \
  --input-column text_clean \
  --model gpt-4o-mini \
  --n 10

python3 postprocess.py --input cell_b_test_v4e.csv --output cell_b_postcheck_v4e.csv
```

---

*생성일 기준: 저장소 내 `cell_b_postcheck_v4e.csv` 및 `prompts.py` / `postprocess.py` / `gpt_inference_multi.py` 상태와 일치.*
