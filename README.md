# Minimal-Pair Steering for Implicit Hate Speech Detection

HateXplain Normal 데이터 EDA부터 minimal-pair 데이터셋(Cell A–D) 구축, Llama-3.2-3B 스티어링 벡터 실험까지 이어지는 2026 캡스톤 NLP 프로젝트입니다.

**핵심 질문:** 암묵적 혐오(implicit hate)는 모델 내부에서 어떻게 표현되며, minimal-pair에서 추출한 steering vector로 baseline이 놓친 false negative를 회복할 수 있는가?

**핵심 결론:** macro F1 +0.03~0.07보다 **FN recovery 30~46%**가 본 연구의 핵심 성과입니다. minimal-pair steering(`v_AB`, `v_AC`)은 Latent Hatred·ToxiGen에서 baseline이 놓친 implicit hate를 실질적으로 되살리며, 효과는 mid-layer(L13~14)에서 가장 안정적이고 random vector 1,000회 반복으로는 설명되지 않습니다.

---

## 발표 자료

- **최종 발표 PPT:** [Google Slides](https://docs.google.com/presentation/d/15wUPyx_ThzNg2Nj_bCKnYf03zcgE7uZuj-E1Alo4HgQ/edit?usp=sharing)
- **최종 결론·발표 스크립트:** [`summary.md`](summary.md)

---

## 최종 실험 결과 요약

### 방법

| Cell | 의미 | 벡터 |
|------|------|------|
| **A** | target + harmful cue가 함께 있는 원문 | hate 방향 기준점 |
| **B** | cue 유지, target 제거·변경 | `v_AB = A − B` (target 축) |
| **C** | target 유지, harmful cue 약화 | `v_AC = A − C` (cue 축) |

- **벡터 추출:** HateXplain → GPT-4o minimal-pair 생성 → 수동 검수 (Cell A/B/C)
- **평가:** Latent Hatred·ToxiGen (OOD evaluation, vector 추출 데이터와 분리)
- **모델:** Llama-3.2-3B + linear probe (last-token steering)

### 비교군

| 구분 | 의미 |
|------|------|
| **B0** | No steering (baseline) |
| **B1** | Random vector (방향 효과 control) |
| **B2** | `v_harm` (generic harm 방향) |
| **E1** | `v_AB` (target vector) |
| **E2** | `v_AC` (cue vector) |

### 메인 결과 (Latent Hatred)

| Setup | Layer | α | macro F1 | hate recall | FN recovery | ΔF1 |
|-------|-------|---|----------|-------------|-------------|------|
| B0 No steering | — | — | 0.588 | 0.545 | — | — |
| B1 Random | 14 | 4.0 | 0.607 | 0.602 | 0.143 | +0.019 |
| B2 v_harm | 10 | 2.0 | 0.605 | 0.668 | 0.270 | +0.017 |
| **E1 v_AB** | **8** | **4.25** | **0.619** | **0.695** | **0.457** | **+0.031** |
| **E2 v_AC** | **14** | **4.0** | **0.624** | **0.690** | **0.369** | **+0.036** |

### 메인 결과 (ToxiGen)

| Setup | Layer | α | macro F1 | hate recall | FN recovery | ΔF1 |
|-------|-------|---|----------|-------------|-------------|------|
| B0 No steering | — | — | 0.585 | 0.423 | — | — |
| B1 Random | 21 | 4.0 | 0.636 | 0.600 | 0.307 | +0.051 |
| **B2 v_harm** | **1** | **4.0** | **0.673** | **0.753** | **0.604** | **+0.088** |
| E1 v_AB | 13 | 4.5 | 0.651 | 0.594 | 0.311 | +0.066 |
| E2 v_AC | 13 | 4.5 | 0.654 | 0.581 | 0.298 | +0.069 |

- **통계적 유의성:** McNemar p < 0.005, bootstrap 95% CI가 0을 포함하지 않음 (두 평가셋 모두)
- **방향 특이성:** random vector 1,000 seed 반복에서도 E1/E2를 넘지 못함
- **레이어:** steering best는 L13~14 부근 mid-layer; representation strength peak와는 항상 일치하지 않음
- **주입 위치:** last-token은 macro F1 균형, all-token은 FN recovery 최대 82% (FP 증가 trade-off)

### 주요 해석·한계

- ToxiGen에서 `v_harm L=1`이 raw best이나, SST-2 sanity check에서 일반 sentiment 능력이 크게 손상 → 높은 점수 ≠ 안정적인 steering
- `v_AB`와 `v_AC`는 cosine ≈ 0.71로 일부 겹침 → joint injection 결합 이득 제한적
- 128 minimal-pair(256문장)부터 `v_harm` 대비 안정적 이득 확인 (data efficiency)
- 정성 사례: dehatebert가 P(hate)≈0.03으로 non-hate 판정한 implicit hate를 v_AB/v_AC가 회복

상세 표·그래프·발표 방어 논리·appendix 구성은 [`summary.md`](summary.md)를 참고하세요.

---

## 브랜치

| 브랜치 | 내용 |
|--------|------|
| **`main`** | Cell B–D 파이프라인, steering vector 실험, Cell A 고품질 필터, 최종 분석 |
| **`eda-cell-a`** | HateXplain EDA, Borderline 분석, Cell A anchor 추출 등 초기 분석 스냅샷 |

초기 EDA·회귀 분석 기록은 `eda-cell-a` 브랜치에서 그대로 확인할 수 있습니다.

---

## 프로젝트 구조

```
capstone_nlp/
├── summary.md                       # 최종 결론·발표 스크립트
├── scripts/                         # EDA·Cell A·회귀 분석
│   ├── hatexplain_cellA_EDA.ipynb
│   ├── hatexplain_cellA_EDA_conclusions.md
│   ├── select_cellA.ipynb
│   ├── minimal_pair_pilot_v5.ipynb
│   ├── build_cell_a_high_quality.py
│   ├── prepare_latent_hatred.py
│   └── build_eval_latent_v1.py
│
├── results/
│   ├── cell_a_high_quality.csv
│   ├── cell_a_anchors_v2_framed.csv
│   ├── p0_base/ … p4_regression/
│
├── dataset_cellB/                     # Cell B (cue 보존·target 일반화)
├── dataset_cellC/                     # Cell C (cue 제거·target 유지)
├── dataset_cellD/                     # Cell D (implicit hate 변환)
│
├── experiment/                        # steering vector 실험 (Week 1–3)
│   ├── week1_pipeline.py
│   ├── week2_sweep.py
│   ├── week3_analysis.py
│   ├── extract_qualitative_examples.py
│   ├── data/eval/                     # eval_latent_v2, eval_toxigen_v1 등
│   ├── data/train/
│   └── results/
│
├── lexicons/hurtlex_EN.tsv
├── NRC-Emotion-Lexicon/
├── hatexplain_prediction.csv
├── .env.example
└── README.md
```

---

## 데이터 파이프라인 (Cell A → D)

| Cell | 역할 | 디렉터리 |
|------|------|----------|
| **A** | HateXplain에서 explicit hate anchor 추출 | `scripts/`, `results/cell_a_high_quality.csv` |
| **B** | cue(혐오 표현) 보존, target만 일반화 | `dataset_cellB/` |
| **C** | target 유지, cue 제거 (implicit화) | `dataset_cellC/` |
| **D** | C를 자연스러운 implicit hate로 재작성 | `dataset_cellD/` |

각 Cell 디렉터리에 `gpt_inference*.py`, `prompts*.py`, `postprocess*.py` 및 CSV 산출물이 포함되어 있습니다.

---

## Steering Vector 실험

minimal-pair(Cell A/B/C)에서 추출한 벡터를 Llama-3.2-3B 특정 레이어에 주입해, Latent Hatred·ToxiGen 평가셋에서 macro F1·FN recovery를 측정합니다.

- **H1** `v_AB` = A−B → target 축
- **H2** `v_AC` = A−C → cue 축

상세 실행 방법·산출물 목록은 [`experiment/README.md`](experiment/README.md)를 참고하세요.

```bash
cd experiment
python week1_pipeline.py    # probe · B0
python week2_sweep.py       # sweep (GPU 권장)
python week3_analysis.py    # ablation · 최종 분석
```

Mac 로컬 실행: `experiment/setup_mac.sh`, `experiment/run_mac.sh`

```bash
# B1 random seed 반복 실험
bash run_mac.sh random-seed

# 정성평가 예시 선정
bash run_mac.sh qual-examples
```

---

## 환경 설정

```bash
cp .env.example .env
# .env에 OPENAI_API_KEY, HF_TOKEN 입력
```

| 변수 | 용도 |
|------|------|
| `OPENAI_API_KEY` | Cell B/C/D GPT 추론 |
| `HF_TOKEN` | Llama-3.2-3B 등 gated 모델 다운로드 |

```bash
# Cell B/C/D 추론
pip install -r dataset_cellD/requirements.txt

# steering 실험
cd experiment && pip install -r requirements.txt
```

---

## EDA 핵심 인사이트 (HateXplain Normal)

**데이터:** Normal 7,814건 — Borderline(2–4명 동의) 34.4% / 만장일치(5명) 65.6%

**회귀 분석 (P4) 핵심 발견:**
- `has_target`(타겟 집단 언급)이 Borderline 예측에 가장 강력 (계수 ≈ +3.1)
- 기존 11개 프레이밍 vs Hurtlex 17개 → F1 72.7%로 **성능 동일**
- 프레이밍 방식보다 "누구를 겨냥했는가"가 논쟁성의 1차 변수

상세 분석·시각화: `results/p0_base/` ~ `p4_regression/`, `scripts/hatexplain_cellA_EDA_conclusions.md`

---

## 참고 문헌

- **HateXplain:** Mathew et al. (2021)
- **Latent Hatred / Implicit Hate Corpus:** ElSherief et al. (2021)
- **HurtLex:** Bassignana et al. (2018)
- **Framing:** ElSherief et al. (2021), Ocampo et al. (2023)
- **Intermediate layers in LLMs:** Skean et al. (2024, 2025)

---

## 작성자

**Minseo** · 2026 캡스톤 디자인 프로젝트
