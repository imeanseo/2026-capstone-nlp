# Probe 학습용 train split

| 파일 | 출처 | 설명 |
|------|------|------|
| `hatexplain_train.csv` | HF HateXplain train | week1에서 생성·저장. probe는 이 중 90% 사용, 10%는 `eval/sanity_hatexplain.csv` |
| `latent_train.csv` | `eval/eval_latent_v2.csv` 80% | `build_probe_train_csvs.py`로 생성 |
| `toxigen_train.csv` | `eval/eval_toxigen_v1.csv` 80% | `build_probe_train_csvs.py`로 생성 |
| `*_train.meta.json` | — | split 메타 (seed, train_n, train_ids) |

재생성:

```bash
python build_probe_train_csvs.py
```
