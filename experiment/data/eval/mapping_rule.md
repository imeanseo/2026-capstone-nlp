# eval 라벨 매핑 규칙 (고정)

> **잠금일**: 2026-05-28  
> 1단계 steering sweep이 끝날 때까지 아래 규칙을 변경하지 않는다.

## Latent Hatred (`eval_latent_v1`)

### 이진 라벨 (`label`)

| 원본 `class` | `label` |
| --- | --- |
| `implicit_hate` | `hate` |
| `explicit_hate` | `hate` |
| `not_hate` | `non-hate` |

- `implicit_class`(fine-grained)는 **메인 라벨 결정에 사용하지 않는다**.
- 분석용 서브셋 표시는 `subtype` 컬럼에 원본 `class`를 그대로 둔다.

### 텍스트 최소 정규화 (`text`)

ElSherief 원본 보존 우선. 아래 4개만 적용:

1. `strip()` — 앞뒤 공백 제거
2. `re.sub(r"\s+", " ", text)` — 연속 공백 → 단일 공백
3. CSV 이스케이프 잔재 `""` → `"`
4. `text` 또는 원본 `class`가 빈 행 drop

**하지 않음**: lowercasing, URL 제거, `@mention`/`#hashtag` 제거, punctuation 통일

### 샘플링 (`eval_latent_v1.csv`)

- `RANDOM_SEED = 20260528`
- hate / non-hate **1:1**, 총 **2,000건** (각 1,000건)
- 컬럼: `id`, `text`, `label`, `subtype`, `source` (= `latent_hatred`)
- **한 번 저장 후 재샘플링 금지**

## ToxiGen-HumanVal (`eval_toxigen_v1`) — 2단계

1단계 sweep 이후 구축. 규칙은 `experiment.md` §2 참고.
