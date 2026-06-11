## 실험 목적

B1(random vector)은 매 실행마다 다른 방향으로 생성되기 때문에 단일 seed 결과만으로는 E1/E2의 추가이득(E1−B1, E2−B1)이 유의미한지 판단하기 어렵다. 이 실험에서는 B1을 여러 seed로 반복 생성한 뒤, 같은 반복 내에서 E1·E2와의 차이를 계산하여 추가이득의 평균과 표준편차를 구한다.

**검증 질문:** E1/E2의 B1 대비 추가이득(ΔF1_incremental)이 0보다 통계적으로 유의미하게 큰가?

---

## 기존 코드와의 통합 위치

`week3_analysis.py`의 **§6 최종 표 (`final_table` 루프) 바로 뒤**에 아래 코드를 통째로 붙여넣으면 된다. `VECTORS`, `EV`, `BEST`, `extract`, `predict`, `evaluate`, `prebatch`, `args` 등 모든 의존성이 이미 그 위에서 정의되어 있다.

```bash
# 실행 방법 (week3_analysis.py에 추가한 뒤)
python week3_analysis.py --eval both
# 또는 한쪽만
python week3_analysis.py --eval latent
```

---

## 추가 코드 (week3_analysis.py 맨 끝에 붙여넣기)

```python
# ── B1 Random Seed 반복 실험: 추가이득 평균 ± SD ──────────────────
N_RAND_SEEDS = 20  # 반복 횟수 (20~30 권장)

def run_random_seed_experiment(tag):
    texts, labels, _ = EV[tag]
    fn = FN[tag]
    batches, inv = prebatch(texts)

    # E1·E2: 고정 벡터 — seed와 무관하므로 루프 밖에서 1회만 추출
    L_ab, a_ab, _ = BEST[tag]["v_AB"]
    L_ac, a_ac, _ = BEST[tag]["v_AC"]
    X_e1 = extract(batches, inv, VECTORS["v_AB"], L_ab, a_ab, "last")
    X_e2 = extract(batches, inv, VECTORS["v_AC"], L_ac, a_ac, "last")
    f1_e1 = evaluate(predict(X_e1), labels)["macro_f1"]
    f1_e2 = evaluate(predict(X_e2), labels)["macro_f1"]

    # B1: seed별 random vector 반복
    # 기존 week2_sweep.py의 v_random은 단일 seed — 여기서는 seed마다 새로 생성
    records = []
    for seed in tqdm(range(N_RAND_SEEDS), desc=f"rand_seed [{tag}]", unit="seed"):
        rng = np.random.default_rng(seed)
        v_rand = rng.standard_normal((N_LAYERS + 1, model.config.hidden_size)).astype(np.float32)
        norms = np.linalg.norm(v_rand, axis=-1, keepdims=True)
        v_rand = v_rand / np.where(norms > 0, norms, 1.0)  # unit norm per layer

        # best_L은 v_AB 기준으로 고정 (기존 sweep 결과 재사용)
        X_b1 = extract(batches, inv, v_rand, L_ab, a_ab, "last")
        f1_b1 = evaluate(predict(X_b1), labels)["macro_f1"]

        records.append({
            "seed":        seed,
            "b1_f1":       round(f1_b1, 4),
            "e1_f1":       round(f1_e1, 4),
            "e2_f1":       round(f1_e2, 4),
            "delta_e1_b1": round(f1_e1 - f1_b1, 4),
            "delta_e2_b1": round(f1_e2 - f1_b1, 4),
        })
        print(f"[rand_seed {tag}] seed={seed:2d}  B1={f1_b1:.4f}  "
              f"ΔE1={f1_e1-f1_b1:+.4f}  ΔE2={f1_e2-f1_b1:+.4f}")

    df = pd.DataFrame(records)
    out_path = os.path.join(args.out, f"random_seed_experiment_{tag}.csv")
    df.to_csv(out_path, index=False)

    # 요약 통계
    b1_arr  = df["b1_f1"].values
    de1_arr = df["delta_e1_b1"].values
    de2_arr = df["delta_e2_b1"].values
    pct_e1 = (de1_arr > 0).mean() * 100
    pct_e2 = (de2_arr > 0).mean() * 100

    print(f"\n===== [{tag}] RANDOM SEED SUMMARY ({N_RAND_SEEDS} seeds) =====")
    print(f"  B1  macro_F1 : {b1_arr.mean():.4f} ± {b1_arr.std():.4f}")
    print(f"  E1−B1 (v_AB) : {de1_arr.mean():+.4f} ± {de1_arr.std():.4f}  "
          f"({pct_e1:.0f}% seeds에서 E1>B1)")
    print(f"  E2−B1 (v_AC) : {de2_arr.mean():+.4f} ± {de2_arr.std():.4f}  "
          f"({pct_e2:.0f}% seeds에서 E2>B1)")

    return df

def plot_random_seed_results(df, tag):
    b1    = df["b1_f1"].values
    seeds = df["seed"].values
    e1    = df["e1_f1"].values[0]
    e2    = df["e2_f1"].values[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: seed별 B1 산점도 + E1/E2 수평선
    ax = axes[0]
    ax.scatter(seeds, b1, color="steelblue", s=25, alpha=0.7, label="B1 (per seed)")
    ax.axhline(b1.mean(), color="steelblue", ls="--", lw=1.5,
               label=f"B1 mean={b1.mean():.4f}")
    ax.fill_between(seeds, b1.mean()-b1.std(), b1.mean()+b1.std(),
                    alpha=0.15, color="steelblue", label="B1 ±1 SD")
    ax.axhline(e1, color="tomato",   lw=2, label=f"E1 v_AB={e1:.4f}")
    ax.axhline(e2, color="seagreen", lw=2, label=f"E2 v_AC={e2:.4f}")
    ax.set_xlabel("Random seed")
    ax.set_ylabel("macro F1")
    ax.set_title(f"[{tag}] B1 분포 vs E1/E2 고정값")
    ax.legend(fontsize=8)

    # 오른쪽: 추가이득 히스토그램
    ax2 = axes[1]
    de1 = df["delta_e1_b1"].values
    de2 = df["delta_e2_b1"].values
    ax2.hist(de1, bins=10, alpha=0.6, color="tomato",
             label=f"E1−B1  mean={de1.mean():+.4f}")
    ax2.hist(de2, bins=10, alpha=0.6, color="seagreen",
             label=f"E2−B1  mean={de2.mean():+.4f}")
    ax2.axvline(0, color="black", ls="--", lw=1.5, label="0 (no gain)")
    ax2.set_xlabel("ΔF1 (E − B1)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"[{tag}] 추가이득 분포")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(args.out, f"random_seed_experiment_{tag}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[rand_seed plot] {out_path}")

# ── 실행 ──
for tag in TAGS:
    df_rand = run_random_seed_experiment(tag)
    plot_random_seed_results(df_rand, tag)

print("\n[all done]")
```

---

## 주의사항

`extract` 함수 시그니처가 `week3_analysis.py`에서 `(batches, inv, vec, L, alpha, mode)` 순서임을 확인했다. 위 코드는 이 시그니처를 그대로 따른다.

`BEST[tag]["v_AB"]`에서 꺼낸 `L_ab`를 B1에도 그대로 사용했는데, B1은 원래 random이라 어떤 layer든 상관없지만 **같은 조건으로 비교**해야 추가이득이 의미 있으므로 E1과 동일한 `best_L` 사용이 맞다.

---

## 기대 결과 해석

| 결과 패턴 | 해석 |
| --- | --- |
| E1−B1 평균 > 0, SD 작음, 대부분 seed에서 E1>B1 | v_AB 추가이득이 random 노이즈를 일관되게 초과 → 벡터 효과 입증 |
| E1−B1 평균 > 0이지만 SD가 커서 일부 seed에서 B1>E1 | 효과 있지만 불안정 → seed 수 늘리거나 alpha 재조정 권장 |
| E1−B1 ≈ 0 또는 음수 | v_AB의 추가이득이 random baseline과 다르지 않음 → 메시지 재검토 필요 |

**pct_seeds_e1_beats_b1 ≥ 90%** 이상이면 논문에서 "v_AB consistently outperforms random baseline across seeds"로 서술 가능.

---

## 소요 시간 예측

| 환경 | N_RAND_SEEDS=20 기준 | 비고 |
| --- | --- | --- |
| macOS M4 Pro (MPS) | 약 25~35분 | E1/E2 hidden 추출은 1회 — B1만 20번 반복 |
| Windows RTX 3080 이상 | 약 10~15분 | batch=32 권장 |