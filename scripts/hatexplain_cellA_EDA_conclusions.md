# HateXplain Cell A EDA 노트북 분석 결론

분석 대상: `scripts/hatexplain_cellA_EDA.ipynb`에 **저장된 실행 출력(stdout)** 과 코드 흐름을 기준으로 정리했습니다. (데이터 소스: `hatexplain_prediction.csv`)

---

## 1. 데이터 스코프

| 항목 | 값 |
|------|-----|
| 전체 샘플 | 19,229건 |
| `hatespeech` + `offensive` (이하 H/O) | 11,415건 (약 **59.4%** of 전체) |
| 라벨 비율 (전체) | normal **40.6%**, hatespeech **30.9%**, offensive **28.5%** |

H/O는 이후 감정·슬러·프레이밍·회귀·클러스터링의 **주 분석 코퍼스**로 사용됩니다.

---

## 2. 어노테이터 합의(agreement)

- 전체 데이터에서 `unanimous` **51.2%**, `majority` **48.8%** (3-way `split`은 이 파이프라인에서 **0건**으로 집계됨).
- **라벨 × 합의**: `normal`은 `unanimous`가 많고(5,124 vs 2,690), `offensive`는 `majority`가 훨씬 많음(3,719 vs 1,761). `hatespeech`는 두 합의 유형이 거의 비슷(2,975 vs 2,960).
- H/O 서브셋에서 `unanimous` 비율은 약 **41.4%** (체크리스트 출력 기준).

**해석**: “공격적이지만 혐오로 보기 애매한” 구간이 `offensive`에 많이 몰려 있어 **다수결 라벨 비중이 높고**, `hatespeech`는 상대적으로 **만장일치에 가까운 분포**를 보입니다.

---

## 3. 길이(토큰 수)와 Cell A 필터 퍼널

- 전체 `n_tokens` 중앙값 **20**, H/O에서도 중앙값 **21** (최종 요약).
- H/O 중 토큰 **10~60** 구간: **82.5%** 커버.
- 단계별 탈락(퍼널 출력 기준): `split` 제외 탈락 0건 → 토큰 10~60에서 **17.5%** 탈락 → “유효 타겟 + 비-Other 타겟 수 ≤ 3” 조건에서 추가 **약 20.1%** 탈락.

**해석**: 길이 필터만으로도 일부 짧·긴 게시물이 걸러지며, 타겟 관련 필터가 **두 번째로 큰 축소** 요인입니다.

---

## 4. 타겟 파싱(Cell 3)과 VALID_GROUPS·공동 출현

### 4.1 `targets` 파싱 (`parse_targets_col`)

- 노트북 Cell 3에서 `parse_targets_col`로 `targets` 컬럼을 **항상 `list`로 정규화**: 이미 `list`면 그대로, `str`이면 `ast.literal_eval`, 그 외 타입은 `[]` (파싱 실패 시에도 `[]`).
- 이전에 발생하던 현상(리스트가 아닌 문자열에 `for t in ts`를 돌려 **글자 단위로 빈도가 잡히던 문제**)을 제거합니다.
- 실행 시 검증 출력 예: 첫 행 `['Hindu', 'Islam', 'Other']`, 타입 `list`, **빈 리스트 0건**.

### 4.2 전체 타겟 토큰 빈도 (샘플당 타겟을 펼친 누적; 노트북 출력 기준)

상위 일부: **`None` 10,183**, African 4,253, Women 3,726, Other 3,487, Islam 2,992, Homosexual 2,472, Jewish 2,396, Arab 1,661, Caucasian 1,594, Men 1,539, Refugee 1,534, Asian 675, Hispanic 663, … (이하 Christian, Minority, Disability 등).

- **`None`이 최빈**인 것은 어노테이션에서 “대상 없음/미기재”류가 리스트에 `None`으로 들어온 경우가 많다는 뜻으로 해석할 수 있습니다. 이후 필터(예: Cell A 유효 타겟만)에서는 **`None`·`Other`를 어떻게 취급할지**를 명시하는 것이 좋습니다.

### 4.3 VALID_GROUPS와 공동 출현

- 분석에 쓰인 유효 타겟 집합: African, Islam, Jewish, Women, Refugee, Homosexual, Arab, Latino_Americans, Asian.
- 공동 출현 행렬에서 **Islam–Arab**, **African–Women**, **Jewish–African** 등 교차가 큼 → 멀티 타겟·교차 신원 표현이 많은 서브코퍼스.
- H/O에서 유효 타겟 개수 분포: **0개 1,324건**, **1개 6,088건**이 주류.

**해석**: “단일 타겟” 가정을 쓰는 실험 설계와 맞추려면 **0타겟·다중 타겟·`None`/`Other`** 처리 규칙이 결과에 직접 영향을 줍니다.

---

## 5. VADER 및 hatespeech vs offensive

- H/O에서 Mann–Whitney 검정 결과, **n_tokens, n_targets, vader_neg, vader_compound, unanimous** 모두 두 라벨 간 **p < 0.001**.
- 평균: `hatespeech`가 **더 길고**, 타겟 수·`vader_neg`가 더 크고, **VADER compound가 더 음수(덜 긍정)**이며, **unanimous 비율이 더 높음** (hate 0.499 vs offensive 0.321).
- 타겟별 VADER compound **중앙값**은 Homosexual, African, Women 등이 더 낮은(더 부정) 쪽에 몰리는 패턴(바이올린/박스플롯과 일치).

**해석**: “혐오” 라벨은 단순 욕설이 아니라 **합의 구조·문장 길이·부정 강도**와 어느 정도 연동됩니다. 다만 효과 크기는 후술 OLS R²와 함께 보면 **선형으로 다 설명되지는 않음**이 분명합니다.

---

## 6. NRC 감정 사전(긍정 연관 단어만 사용)

- `hatespeech` vs `offensive`의 감정별 평균 카운트는 **숫자상으로는 매우 근접** (anger/disgust/fear 등 소수 차이).
- `neg_cue_sum`(anger+disgust+fear) 평균: hatespeech **2.125**, offensive **2.182** → offensive가 오히려 약간 높음.
- `neg_cue_sum` × `unanimous` Spearman: **ρ ≈ -0.016, p ≈ 0.094** (유의하지 않음).

**해석**: 이 노트북 설정(NRC의 association==1만, 토큰 매칭)으로는 **hatespeech/offensive 구분 신호가 약하거나 방향이 일관되지 않음**. Cell A의 “감정 cue”는 **보조 지표** 수준으로 보는 것이 타당합니다.

---

## 7. HurtLex 기반 슬러

- H/O에서 슬러 포함 샘플 **78.7%**, 평균 밀도 **0.083**.
- 타겟별 `has_slur_rate`는 Jewish, African, Homosexual 순으로 높게 나옴.
- 슬러 밀도 vs 라벨(이진) Spearman: **ρ ≈ 0.18, p 극소** → 슬러는 **hatespeech 쪽과 통계적으로 연관**되나, 단독으로 라벨을 완전히 설명하진 못함(회귀 R²와 일치).

---

## 8. 룰·키워드 기반 프레이밍 탐지

- **NONE_DETECTED: 59.6%** → 현재 사전/패턴으로는 절반 이상이 “미검출”.
- 검출된 프레이밍 중 빈도 상위: INTELLECTUAL_INFERIORITY, CRIMINAL_DANGER, THREAT_VIOLENCE, SEXUAL_GENDERED 등.
- 최종 요약: **복합 프레이밍(2개 이상) 약 8.8%**.

**해석**: 룰 기반 태깅은 **해석 가능성**은 있으나 **커버리지와 정밀도 한계**가 크므로, 이후 모델링에서는 **보조 특성** 또는 **정성 사례 탐색** 용도가 적합합니다.

---

## 9. OLS 선형 회귀(탐색적)

| 종속변수 | R² |
|----------|-----|
| hatespeech(이진) | **0.059** |
| unanimous(이진) | **0.027** |

- **hatespeech**: `slur_density`, `exclusion`, `dehumanization`, `threat_violence` 등이 유의한 양(+)의 계수, `vader_neg`/`vader_compound`는 음의 방향 등 — 방향은 “혐오·위협·배제 프레이밍 + 슬러”와 정합.
- **unanimous**: `slur_density`, `dehumanization`, `threat_violence` 등 일부 유의, R²는 매우 낮음.

**해석**: 선택한 수치·룰 특성만으로는 **분산의 약 3~6%** 수준만 설명합니다. 라벨과 합의는 **다요인·문맥 의존**이며, 선형 OLS는 **가설 점검용**으로만 쓰는 것이 좋습니다.

---

## 10. K-Means (k=8, 표준화 특성)

- 군집별로 **거의 전부 hatespeech**(예: hate_rate ≈ 0.99, n≈2,536), **거의 전부 offensive**(hate_rate 0, n≈2,686), **혼합·중간 비율**(hate_rate 0.44~0.76) 등이 분리됨.
- 일부 군집은 **VADER compound 평균이 양수**(예: cluster 3, avg_vader ≈ 0.32)인 반면 라벨은 혼합 → **비꼬기·역설적 표현** 등 VADER가 놓치는 유형이 존재할 수 있음.

**해석**: 특성 공간에서 **“강한 혐오 슬러 군집” vs “offensive 위주” vs “혼합”** 같은 **스타일 군집**은 잡히나, 군집–골드 라벨은 1:1 대응이 아닙니다.

---

## 11. unanimous vs majority (H/O 전체, 11,415건이 아닌 df_raw의 H/O agreement 집계)

- `unanimous` 4,721 vs `majority` 6,694 (cell 18 기준; `analysis_df`와 집계 범위가 다를 수 있음).
- Mann–Whitney: **unanimous**가 **더 짧은 문장**, **더 음의 compound**, **슬러 개수는 더 많음**; `n_targets`는 **유의 차이 없음**(p≈0.50).

**해석**: “만장일치”인 경우가 반드시 더 단순한 표현만은 아니고, **슬러가 더 많이 실리는 경향**이 있습니다. 반면 **타겟 개수**만으로는 합의 유형이 갈리지 않습니다.

---

## 12. 구현·데이터 주의사항

1. **`agreement` 정의**: 노트북은 `annotators` 문자열에서 `label` 배열만 추출해 3명 전원 일치/2:1만 구분; 원 데이터의 `split` 정의와 완전히 같다고 가정하면 안 됩니다.
2. **회귀·군집**: 인과 해석 금지, **탐색적(EDA)** 결론으로 한정하는 것이 맞습니다.

*(Cell 3 타겟 빈도는 `parse_targets_col` 반영 후 정상적인 타겟 이름 단위로 집계됨 — §4 참고.)*

---

## 13. Cell A / 후속 연구에 대한 총괄 결론

1. **H/O는 전체의 과반에 가깝고**, `offensive`는 **majority 합의 비중이 높아** “경계 사례” 풀이 큼 → 최소쌍·대조 학습 시 **합의 유형 stratify**를 고려할 가치가 있음.
2. **길이 10~60**은 대부분을 살리면서 꼬리를 줄이는 **합리적 1차 필터** (H/O 기준 약 82.5%).
3. **슬러(HurtLex)**는 hatespeech와 **통계적으로 연관**되나, **단독 설명력은 낮음**(R² 낮음, 라벨 중첩).
4. **NRC(이 설정)**와 **neg_cue_sum**은 hatespeech/offensive **분리에 거의 기여하지 않음**.
5. **룰 기반 프레이밍**은 해석용으로는 유용하나 **미검출 60%** 근방 → 자동 라벨 대체재로는 부족.
6. **VADER + 길이 + 합의**는 hatespeech vs offensive에 **유의한 평균 차**를 보이나, 선형 모형으로는 **분산의 일부만** 포착 → **문맥 모델(Transformer 등)** 또는 **rationales** 쪽이 본 연구의 핵심 레버로 보임.

