#!/usr/bin/env python
"""eval CSV에서 probe 학습용 train split 생성 (80%, stratified, seed=42)."""
import os, json, argparse
import pandas as pd
from sklearn.model_selection import train_test_split

P = argparse.ArgumentParser()
P.add_argument("--data-eval", default="data/eval")
P.add_argument("--out-dir", default="data/train")
P.add_argument("--train-frac", type=float, default=0.8)
P.add_argument("--seed", type=int, default=42)
args = P.parse_args()

os.makedirs(args.out_dir, exist_ok=True)


def norm_labels(df):
    df = df.copy()
    if not pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(str).str.strip().str.lower().map(
            {"hate": 1, "non-hate": 0})
    df["label"] = df["label"].astype(int)
    return df


def split_save(name, src_path, id_col="id"):
    df = norm_labels(pd.read_csv(src_path))
    train, hold = train_test_split(
        df, train_size=args.train_frac, stratify=df["label"],
        random_state=args.seed)
    out = os.path.join(args.out_dir, f"{name}_train.csv")
    train[["text", "label"]].to_csv(out, index=False)
    meta = {
        "source": src_path,
        "train_n": len(train),
        "holdout_n": len(hold),
        "train_frac": args.train_frac,
        "seed": args.seed,
        "train_ids": train[id_col].tolist() if id_col in train.columns else None,
    }
    meta_path = os.path.join(args.out_dir, f"{name}_train.meta.json")
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"[{name}] train={len(train)} holdout={len(hold)} → {out}")
    return out

split_save("latent", os.path.join(args.data_eval, "eval_latent_v2.csv"))
split_save("toxigen", os.path.join(args.data_eval, "eval_toxigen_v1.csv"))
print("[done] latent_train.csv, toxigen_train.csv 준비 완료")
