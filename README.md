# Implicit Hate Detection — Capstone NLP

HateXplain Normal 데이터 EDA부터 minimal-pair 데이터셋(Cell A–D) 구축, Llama-3.2-3B 스티어링 벡터 실험까지 이어지는 캡스톤 프로젝트입니다.

**핵심 질문:** 암묵적 혐오(implicit hate)는 모델 내부에서 어떻게 표현되며, minimal-pair에서 추출한 steering vector로 탐지 성능을 개선할 수 있는가?

---

## 브랜치

| 브랜치 | 내용 |
|--------|------|
| **`main`** | Cell B–D 파이프라인, steering vector 실험, Cell A 고품질 필터 등 최신 작업 |
| **`eda-cell-a`** | HateXplain EDA, Borderline 분석, Cell A anchor 추출 등 초기 분석 스냅샷 |

초기 EDA·회귀 분석 기록은 `eda-cell-a` 브랜치에서 그대로 확인할 수 있습니다.

---

## 프로젝트 구조

```
capstone_nlp/
├── scripts/                         # EDA·Cell A·회귀 분석
│   ├── hatexplain_cellA_EDA.ipynb
│   ├── hatexplain_cellA_EDA_conclusions.md
│   ├── select_cellA.ipynb
│   ├── minimal_pair_pilot_v5.ipynb
│   ├── build_cell_a_high_quality.py   # Cell A 고품질 필터
│   └── prepare_latent_hatred.py       # Latent Hatred eval 빌드
│
├── results/
│   ├── cell_a_high_quality.csv        # 고품질 Cell A anchor
│   ├── cell_a_anchors_v2_framed.csv   # (eda-cell-a) framing 메타 포함 anchor
│   ├── p0_base/ … p4_regression/     # Borderline·프레이밍·회귀 분석 산출물
│
├── dataset_cellB/                     # Cell B 생성 (A → cue 보존·target 일반화)
├── dataset_cellC/                     # Cell C 생성 (B → cue 제거·target 유지)
├── dataset_cellD/                     # Cell D 생성 (C → implicit hate 변환)
│
├── experiment/                        # steering vector 실험 (Week 1–3)
│   ├── week1_pipeline.py              # probe 학습 · B0 baseline
│   ├── week2_sweep.py                 # 벡터 sweep
│   ├── week3_analysis.py              # ablation · 최종 표/그래프
│   ├── data/eval/                     # eval_latent_v2, eval_toxigen_v1 등
│   ├── data/train/                    # probe 학습용 train split
│   └── results/                       # probe.pkl, sweep csv, png
│
├── steering_vector/                   # eval 데이터 준비 · 실험 설계 문서
│   ├── experiment.md
│   ├── data/latent_hatred/            # Implicit Hate Corpus 전처리
│   └── scripts/build_eval_latent_v1.py
│
├── lexicons/hurtlex_EN.tsv
├── NRC-Emotion-Lexicon/
├── hatexplain_prediction.csv          # HateXplain 원본 (Normal 7,814건)
├── .env.example                       # API 키 템플릿
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

---

## 작성자

**Minseo** · 2026 캡스톤 디자인 프로젝트
