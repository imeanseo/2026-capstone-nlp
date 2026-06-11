#!/usr/bin/env python
"""
정성평가용 예시 선정 — B0/E1/E2/B1 예측 비교 후 발표 슬라이드용 후보 추출.

사용 예:
  python -u extract_qualitative_examples.py --eval latent
  python -u extract_qualitative_examples.py --eval both --top-k 5
"""
import os, json, sys, argparse
import numpy as np, pandas as pd, torch, joblib

from huggingface_hub import login
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

P = argparse.ArgumentParser()
P.add_argument("--model", default="meta-llama/Llama-3.2-3B")
P.add_argument("--out", default="results")
P.add_argument("--data-eval", default="data/eval")
P.add_argument("--src", default="src/eval")
P.add_argument("--batch", type=int, default=16)
P.add_argument("--max-len", type=int, default=128)
P.add_argument("--eval", choices=["both", "latent", "tg"], default="latent")
P.add_argument("--top-k", type=int, default=5, help="카테고리별 상위 예시 수")
P.add_argument("--max-chars", type=int, default=180, help="슬라이드 가독성용 최대 글자 수")
args = P.parse_args()

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
_model_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
print(f"[cfg] device={device} eval={args.eval} top_k={args.top_k}")

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=_model_dtype).to(device).eval()
model.config.output_hidden_states = False
N_LAYERS = model.config.num_hidden_layers

sys.path.insert(0, args.src)
from metrics import HATE, NON_HATE, evaluate

pb = joblib.load(os.path.join(args.out, "probe.pkl"))
scaler, clf = pb["scaler"], pb["clf"]
VECTORS = {n: np.load(os.path.join(args.out, f"{n}.npy"))
           for n in ["v_AB", "v_AC", "v_random", "v_harm"]}

def load_eval_df(path, group=False):
    df = pd.read_csv(path)
    if not pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(str).str.strip().str.lower().map(
            {"hate": 1, "non-hate": 0})
    df["label"] = df["label"].astype(int)
    if group and "target_group" not in df.columns:
        df["target_group"] = None
    return df

EV_FILES = {
    "eval_latent_v2": os.path.join(args.data_eval, "eval_latent_v2.csv"),
    "eval_toxigen_v1": os.path.join(args.data_eval, "eval_toxigen_v1.csv"),
}
TAGS = (["eval_latent_v2", "eval_toxigen_v1"] if args.eval == "both"
        else ["eval_latent_v2"] if args.eval == "latent" else ["eval_toxigen_v1"])

def load_sweep(tag):
    c = pd.read_csv(os.path.join(args.out, f"sweep_coarse_{tag}.csv"))
    fp = os.path.join(args.out, f"sweep_fine_{tag}.csv")
    return pd.concat([c, pd.read_csv(fp)], ignore_index=True) if os.path.exists(fp) else c

def best_of(df, name):
    sub = df[df.vector == name]
    b = sub.loc[sub.macro_f1.idxmax()]
    return int(b.layer), float(b.alpha)

BEST = {tag: {n: best_of(load_sweep(tag), n)
              for n in ["v_AB", "v_AC", "v_random"]}
        for tag in TAGS}
FN = {tag: np.load(os.path.join(args.out, f"fn_subset_{tag}.npy")) for tag in TAGS}

class AblHook:
    def __init__(self, vec, alpha, mode="last"):
        self.v = torch.tensor(vec, dtype=torch.float16, device=device)
        self.a = float(alpha)
        self.mode = mode
        self.h = None

    def _fn(self, m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if self.mode == "last":
            h[:, -1, :] = h[:, -1, :] + self.a * self.v
        return o

    def attach(self, layer):
        self.h = layer.register_forward_hook(self._fn)
        return self

    def detach(self):
        if self.h is not None:
            self.h.remove()
            self.h = None

def prebatch(texts):
    texts = [str(t) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    inv = np.argsort(order)
    B = [tok([texts[i] for i in order[s:s + args.batch]], return_tensors="pt",
             padding=True, truncation=True, max_length=args.max_len)
         for s in range(0, len(order), args.batch)]
    return B, inv

@torch.inference_mode()
def extract(batches, inv, vec=None, L=None, alpha=0.0, mode="none"):
    hk = (AblHook(vec[L], alpha, mode).attach(model.model.layers[L])
          if mode != "none" else None)
    out = []
    try:
        for enc in batches:
            enc = {k: v.to(device) for k, v in enc.items()}
            out.append(model.model(**enc, use_cache=False,
                                   output_hidden_states=False
                                   ).last_hidden_state[:, -1, :].float().cpu().numpy())
    finally:
        if hk:
            hk.detach()
    return np.concatenate(out, 0)[inv]

def predict(X):
    return clf.predict(scaler.transform(X))

def pred_label(v):
    return "hate" if v == HATE else "non-hate"

def display_text(row):
    if "original_text" in row and pd.notna(row["original_text"]):
        return str(row["original_text"]).strip()
    return str(row["text"]).strip()

def collect_predictions(tag):
    df = load_eval_df(EV_FILES[tag], group=(tag == "eval_toxigen_v1"))
    texts = df["text"].tolist()
    labels = df["label"].to_numpy()
    batches, inv = prebatch(texts)

    print(f"[{tag}] B0 extract...")
    p_b0 = predict(extract(batches, inv, mode="none"))
    L_ab, a_ab = BEST[tag]["v_AB"][:2]
    L_ac, a_ac = BEST[tag]["v_AC"][:2]
    L_b1, a_b1 = BEST[tag]["v_random"][:2]

    print(f"[{tag}] E1/E2/B1 extract (L_ab={L_ab}, L_ac={L_ac}, L_b1={L_b1})...")
    p_e1 = predict(extract(batches, inv, VECTORS["v_AB"], L_ab, a_ab, "last"))
    p_e2 = predict(extract(batches, inv, VECTORS["v_AC"], L_ac, a_ac, "last"))
    p_b1 = predict(extract(batches, inv, VECTORS["v_random"], L_b1, a_b1, "last"))

    out = df.copy()
    out["label_name"] = out["label"].map({1: "hate", 0: "non-hate"})
    out["b0_pred"] = p_b0
    out["e1_pred"] = p_e1
    out["e2_pred"] = p_e2
    out["b1_pred"] = p_b1
    out["display_text"] = out.apply(display_text, axis=1)
    out["text_len"] = out["display_text"].str.len()
    out["in_fn_pool"] = out.index.isin(FN[tag])
    return out, labels

def assign_category(row):
    y = row["label"]
    b0, e1, e2, b1 = row["b0_pred"], row["e1_pred"], row["e2_pred"], row["b1_pred"]

    if y == HATE and b0 == NON_HATE and e1 == HATE:
        return "success_fn_recovery_e1"
    if y == HATE and b0 == NON_HATE and e2 == HATE and e1 == NON_HATE:
        return "success_fn_recovery_e2_only"
    if y == HATE and b0 == HATE and e1 == HATE:
        return "success_stable_tp"
    if y == NON_HATE and b0 == NON_HATE and e1 == NON_HATE:
        return "success_stable_tn"
    if y == HATE and b0 == NON_HATE and e1 == NON_HATE:
        return "failure_fn_persistent"
    if y == NON_HATE and b0 == NON_HATE and e1 == HATE:
        return "failure_fp_e1"
    if y == NON_HATE and b0 == HATE:
        return "failure_fp_b0"
    if y == HATE and b0 == HATE and e1 == NON_HATE:
        return "failure_steering_hurt"
    return "other"

def score_row(row, cat):
    s = 0
    tl = row["text_len"]
    if tl <= args.max_chars:
        s += 3
    if tl <= 100:
        s += 2
    if row.get("subtype") == "implicit_hate":
        s += 4
    if row.get("subtype") == "explicit_hate":
        s += 1
    if cat == "success_fn_recovery_e1":
        if row["b1_pred"] == NON_HATE:
            s += 3
        if row["e2_pred"] == HATE:
            s += 1
        if row["in_fn_pool"]:
            s += 2
    if cat == "failure_fn_persistent" and row.get("subtype") == "implicit_hate":
        s += 2
    if cat == "failure_fp_e1" and row["b0_pred"] == NON_HATE and row["b1_pred"] == NON_HATE:
        s += 2
    return s

CATEGORY_META = {
    "success_fn_recovery_e1": {
        "title": "Success — FN recovery (B0 miss → E1 hit)",
        "slide_hint": "발표 앞부분 핵심: implicit hate를 B0가 놓쳤지만 steering으로 회복",
    },
    "success_fn_recovery_e2_only": {
        "title": "Success — E2-only recovery",
        "slide_hint": "cue 축(v_AC)만 효과가 있는 케이스",
    },
    "success_stable_tp": {
        "title": "Success — stable true positive",
        "slide_hint": "B0/E1 모두 맞춘 hate (baseline도 가능한 케이스)",
    },
    "success_stable_tn": {
        "title": "Success — stable true negative",
        "slide_hint": "non-hate를 안정적으로 유지",
    },
    "failure_fn_persistent": {
        "title": "Failure — persistent FN",
        "slide_hint": "steering 후에도 놓치는 어려운 케이스",
    },
    "failure_fp_e1": {
        "title": "Failure — E1 false positive",
        "slide_hint": "steering trade-off: non-hate를 hate로 오분류",
    },
    "failure_fp_b0": {
        "title": "Failure — B0 false positive",
        "slide_hint": "steering 없이도 틀린 케이스",
    },
    "failure_steering_hurt": {
        "title": "Failure — steering hurt",
        "slide_hint": "B0는 맞았는데 E1이 오히려 틀림",
    },
}

def select_top(df, cat, k):
    sub = df[df["category"] == cat].copy()
    if sub.empty:
        return sub
    sub["score"] = sub.apply(lambda r: score_row(r, cat), axis=1)
    return sub.sort_values(["score", "text_len"], ascending=[False, True]).head(k)

def write_presentation_md(tag, picks, out_md):
    lines = [f"# Qualitative examples — {tag}\n"]
    front = ["success_fn_recovery_e1", "success_stable_tp", "success_stable_tn"]
    back = ["failure_fn_persistent", "failure_fp_e1", "failure_steering_hurt"]

    for section, cats in [("Recommended for slide front", front),
                          ("Misclassification / limitations", back)]:
        lines.append(f"## {section}\n")
        for cat in cats:
            meta = CATEGORY_META.get(cat, {"title": cat, "slide_hint": ""})
            lines.append(f"### {meta['title']}")
            lines.append(f"> {meta['slide_hint']}\n")
            sub = picks.get(cat, pd.DataFrame())
            if sub.empty:
                lines.append("_No examples found._\n")
                continue
            for i, (_, r) in enumerate(sub.iterrows(), 1):
                extra = []
                if pd.notna(r.get("subtype")):
                    extra.append(f"subtype={r['subtype']}")
                if pd.notna(r.get("target_group")):
                    extra.append(f"group={r['target_group']}")
                meta_s = f" ({', '.join(extra)})" if extra else ""
                lines.append(f"**{i}. [{r['id']}]{meta_s}**")
                lines.append(f"- Gold: **{r['label_name']}**")
                lines.append(f"- B0: {pred_label(r['b0_pred'])} | "
                             f"B1: {pred_label(r['b1_pred'])} | "
                             f"E1: {pred_label(r['e1_pred'])} | "
                             f"E2: {pred_label(r['e2_pred'])}")
                lines.append(f"- Text: \"{r['display_text']}\"")
                lines.append("")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {out_md}")

def main():
    for tag in TAGS:
        df, labels = collect_predictions(tag)
        df["category"] = df.apply(assign_category, axis=1)

        m = {
            "b0": evaluate(df["b0_pred"].to_numpy(), labels),
            "e1": evaluate(df["e1_pred"].to_numpy(), labels),
        }
        print(f"[{tag}] macro_F1  B0={m['b0']['macro_f1']:.4f}  E1={m['e1']['macro_f1']:.4f}")
        print(df["category"].value_counts().to_string())

        picks = {cat: select_top(df, cat, args.top_k)
                 for cat in CATEGORY_META}
        selected = pd.concat(
            [s.assign(category=cat) for cat, s in picks.items() if len(s)],
            ignore_index=True,
        )
        out_csv = os.path.join(args.out, f"qualitative_examples_{tag}.csv")
        out_md = os.path.join(args.out, f"qualitative_examples_{tag}_presentation.md")
        cols = ["id", "display_text", "label_name", "category", "score",
                "subtype", "target_group", "text_len",
                "b0_pred", "b1_pred", "e1_pred", "e2_pred", "in_fn_pool"]
        cols = [c for c in cols if c in selected.columns]
        selected[cols].to_csv(out_csv, index=False)
        write_presentation_md(tag, picks, out_md)
        print(f"[saved] {out_csv} ({len(selected)} examples)")

    print("\n[done]")

if __name__ == "__main__":
    main()
