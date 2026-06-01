import os, re, json, time, argparse, importlib, sys
import numpy as np, pandas as pd, torch, joblib
from tqdm.auto import tqdm

from huggingface_hub import login
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

# ─────────────────────────── CLI ───────────────────────────
P = argparse.ArgumentParser()
P.add_argument("--model", default="meta-llama/Llama-3.2-3B")
P.add_argument("--eval-latent",  default="data/eval/eval_latent_v2.csv")
P.add_argument("--eval-latent-text-col",  default="text")
P.add_argument("--eval-latent-label-col", default="label")
P.add_argument("--cell-c", default="cell_c_test_final.csv")
P.add_argument("--cell-b", default="cell_bbb_domain_v10_256_revised.csv")
P.add_argument("--hatexplain-csv", default="data/train/hatexplain_train.csv",
               help="없으면 HF에서 만들어 저장")
P.add_argument("--out",  default="results")
P.add_argument("--data-eval", default="data/eval")
P.add_argument("--src",  default="src/eval")
P.add_argument("--batch", type=int, default=16)
P.add_argument("--max-len", type=int, default=128)
P.add_argument("--seed", type=int, default=20260528)
P.add_argument("--sanity-layer", type=int, default=20)
P.add_argument("--sanity-alpha", type=float, default=4.0)
P.add_argument("--per-group-per-label", type=int, default=60,
               help="ToxiGen stratified 샘플링: 그룹×label당 N건")
P.add_argument("--probe-test-size", type=float, default=0.1,
               help="HateXplain held-out 비율 (sanity용)")
P.add_argument("--skip-toxigen", action="store_true",
               help="eval_toxigen_v1.csv 이미 있으면 건너뜀")
args = P.parse_args()

for d in [args.out, args.data_eval, args.src]:
    os.makedirs(d, exist_ok=True)

# Windows + NVIDIA GPU:
# device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
_model_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
np.random.seed(args.seed)
print(f"[cfg] device={device} dtype={_model_dtype} batch={args.batch} sanity=L{args.sanity_layer}/α{args.sanity_alpha}")

# ─────────────────────────── 모델 ───────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"   # 마지막 토큰이 항상 -1
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=_model_dtype, output_hidden_states=True
).to(device).eval()
N_LAYERS = model.config.num_hidden_layers
HIDDEN   = model.config.hidden_size
print(f"[model] layers={N_LAYERS} hidden={HIDDEN}")

# ─────────────────────────── 공용 함수 ───────────────────────────
@torch.no_grad()
def extract_hidden(texts, keep_layers=None, batch_size=None):
    """left-pad + 마지막 토큰 hidden.
    keep_layers=None -> (N, HIDDEN)
    keep_layers=[..] -> (N, len(keep_layers), HIDDEN)"""
    bs = batch_size or args.batch
    feats = []
    for i in range(0, len(texts), bs):
        batch = [str(t) for t in texts[i:i+bs]]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_len).to(device)
        hs = model(**enc).hidden_states
        if keep_layers is None:
            feats.append(hs[-1][:, -1, :].float().cpu().numpy())
        else:
            stk = np.stack([hs[L][:, -1, :].float().cpu().numpy()
                            for L in keep_layers], axis=1)
            feats.append(stk)
    return np.concatenate(feats, axis=0)

class SteeringHook:
    def __init__(self, vec, alpha):
        self.v = torch.tensor(vec, dtype=torch.float16, device=device)
        self.a = float(alpha); self.h = None
    def _fn(self, m, i, o):
        if isinstance(o, tuple):
            o[0][:, -1, :] = o[0][:, -1, :] + self.a * self.v
            return o
        o[:, -1, :] = o[:, -1, :] + self.a * self.v
        return o
    def attach(self, layer): self.h = layer.register_forward_hook(self._fn); return self
    def detach(self):
        if self.h is not None: self.h.remove(); self.h = None

def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", str(text).lower())).strip()

# metrics.py 자리 잡기
METRICS_SRC = '''from sklearn.metrics import f1_score, recall_score
HATE, NON_HATE = 1, 0

def evaluate(preds, labels):
    return {"macro_f1":   f1_score(labels, preds, average="macro"),
            "hate_recall": recall_score(labels, preds, pos_label=HATE)}

def evaluate_by_group(preds, labels, groups):
    out = {}
    for g in set(groups):
        idx = [k for k, gi in enumerate(groups) if gi == g]
        l = [labels[k] for k in idx]; p = [preds[k] for k in idx]
        out[g] = {"hate_recall": recall_score(l, p, pos_label=HATE) if HATE in l else None,
                  "n": len(p)}
    return out
'''
mpath = os.path.join(args.src, "metrics.py")
if not os.path.exists(mpath):
    with open(mpath, "w") as f: f.write(METRICS_SRC)
sys.path.insert(0, args.src)
import metrics; importlib.reload(metrics)
from metrics import evaluate, HATE
assert abs(evaluate([1,0,1,0],[1,0,1,0])["macro_f1"] - 1.0) < 1e-9
print("[metrics] OK")

# ─────────────────────────── 1. eval_latent 로드 ───────────────────────────
ev = pd.read_csv(args.eval_latent)
ev = ev.rename(columns={args.eval_latent_text_col: "text",
                        args.eval_latent_label_col: "label"})
ev["label"] = ev["label"].astype(str).str.strip().str.lower().map(
    {"hate": 1, "non-hate": 0}).astype(int)
eval_latent_texts  = ev["text"].tolist()
eval_latent_labels = ev["label"].to_numpy()
print(f"[eval_latent] {len(ev)}건 label={np.bincount(eval_latent_labels).tolist()}")

# Cell A/B/C 정렬 (v_AB 추출용)
dfc = pd.read_csv(args.cell_c); dfb = pd.read_csv(args.cell_b)
tri = (dfc[["idx","cell_a","cell_c_modified"]]
       .rename(columns={"cell_a":"A","cell_c_modified":"C"})
       .merge(dfb[["text_clean","generated_text"]]
              .rename(columns={"text_clean":"A","generated_text":"B"}),
              on="A", how="inner")
       .dropna(subset=["A","B","C"]))
tri = tri[(tri["B"].astype(str).str.len()>0) & (tri["C"].astype(str).str.len()>0)]
A_texts, B_texts, C_texts = tri["A"].tolist(), tri["B"].tolist(), tri["C"].tolist()
print(f"[triple] PASS={len(tri)}")

# ─────────────────────────── 2. HateXplain 로드 ───────────────────────────
def load_hatexplain():
    if os.path.exists(args.hatexplain_csv):
        df = pd.read_csv(args.hatexplain_csv)
        print(f"[hx] 기존 split 사용 -> {args.hatexplain_csv}")
        return df
    from datasets import load_dataset
    ds = load_dataset("hatexplain", split="train")
    rows = []
    for r in ds:
        text = " ".join(r["post_tokens"])
        labs = r["annotators"]["label"]
        maj = max(set(labs), key=labs.count)
        rows.append({"text": text, "label": 0 if maj == 1 else 1})
    df = pd.DataFrame(rows)
    df.to_csv(args.hatexplain_csv, index=False)
    print(f"[hx] HF에서 생성·저장 -> {args.hatexplain_csv}")
    return df

hx = load_hatexplain()
print(f"[hx] {len(hx)}건 label={np.bincount(hx['label'].astype(int)).tolist()}")

# ─────────────────────────── 3. eval_latent leakage check ───────────────────────────
ev_norm   = set(ev["text"].map(norm))
cell_norm = set(map(norm, A_texts + B_texts + C_texts))
hx_norm   = set(hx["text"].map(norm))
ov_cell = len(ev_norm & cell_norm)
ov_hx   = len(ev_norm & hx_norm)
print(f"[leak] eval_latent ∩ Cell A/B/C = {ov_cell}, ∩ HateXplain = {ov_hx}")
with open(os.path.join(args.data_eval, "leakage_check.md"), "w") as f:
    f.write(f"# leakage check\n\n- eval_latent ∩ Cell A/B/C: **{ov_cell}** 건\n"
            f"- eval_latent ∩ HateXplain: **{ov_hx}** 건\n")
if ov_cell or ov_hx:
    print("⚠️  겹침 발견 — eval에서 제거 후 재실행 필요")

# ─────────────────────────── 4. eval_toxigen 구축 ───────────────────────────
tg_path = os.path.join(args.data_eval, "eval_toxigen_v1.csv")
if args.skip_toxigen and os.path.exists(tg_path):
    tox = pd.read_csv(tg_path)
    print(f"[toxigen] 기존 파일 사용 ({len(tox)}건)")
else:
    from datasets import load_dataset
    tg = load_dataset("toxigen/toxigen-data", name="annotated")
    tg = pd.concat([tg[s].to_pandas() for s in tg.keys()], ignore_index=True)
    tg.columns = [c.lower() for c in tg.columns]
    tg = tg[tg["toxicity_ai"] != 3].copy()
    tg["label"] = (tg["toxicity_ai"] >= 4).astype(int)
    RULES = [
        ("chinese",         ["chinese"]),
        ("asian",           ["asian"]),
        ("mexican",         ["mexican"]),
        ("latino",          ["latino","latina","hispanic"]),
        ("muslim",          ["muslim","islam"]),
        ("jewish",          ["jewish","jew"]),
        ("black",           ["black","african"]),
        ("lgbtq",           ["lgbtq","lgbt","gay","lesbian","queer","bisexual","trans"]),
        ("women",           ["women","woman","female"]),
        ("middle_east",     ["middle eastern","middle-eastern","middle east","middle_east"]),
        ("native_american", ["native american","native-american","indigenous","native_american"]),
        ("mental_dis",      ["mental disab","mental ill","mental health","cognitive","mental_dis"]),
        ("physical_dis",    ["physical disab","disab","physical_dis"]),
    ]
    def to13(g):
        s = str(g).lower()
        for canon, keys in RULES:
            if any(k in s for k in keys): return canon
        return "OTHER"
    tg["group13"] = tg["target_group"].map(to13)
    unmapped = sorted(tg.loc[tg.group13=="OTHER","target_group"].unique())
    if unmapped: print(f"[toxigen] UNMAPPED(drop): {unmapped}")
    tg = tg[tg["group13"] != "OTHER"].copy()
    parts = []
    for g, gdf in tg.groupby("group13"):
        for lab in [0,1]:
            sub = gdf[gdf["label"] == lab]
            take = min(args.per_group_per_label, len(sub))
            if take: parts.append(sub.sample(take, random_state=args.seed))
    tox = pd.concat(parts, ignore_index=True)
    tox = tox.rename(columns={"group13":"target_group","target_group":"target_group_raw"})
    tox = tox[["text","label","target_group","target_group_raw","toxicity_ai"]]
    tox["source"] = "toxigen_humanval"
    tox.insert(0, "id", [f"TG{i:05d}" for i in range(len(tox))])
    tox = tox[~tox["text"].map(norm).isin(hx_norm)].reset_index(drop=True)
    tox.to_csv(tg_path, index=False)
    print(f"[toxigen] 저장 {len(tox)}건, 그룹 {tox['target_group'].nunique()}개, "
          f"label={np.bincount(tox['label']).tolist()}")

toxigen_texts  = tox["text"].tolist()
toxigen_labels = tox["label"].astype(int).to_numpy()
toxigen_groups = tox["target_group"].tolist() if "target_group" in tox else None

# ─────────────────────────── 5. probe 학습 + sanity_hatexplain ───────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

@torch.inference_mode()
def extract_last_fast(texts, batch_size=None, desc="extract"):
    bs = batch_size or args.batch
    texts = [str(t) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    inv = np.argsort(order)
    out = []
    for s in tqdm(range(0, len(order), bs), desc=desc):
        idx = order[s:s+bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt",
                  padding=True, truncation=True, max_length=args.max_len).to(device)
        h = model.model(**enc, use_cache=False,
                        output_hidden_states=False).last_hidden_state[:, -1, :]
        out.append(h.float().cpu().numpy())
        del enc, h
    return np.concatenate(out, 0)[inv]

hx_train, hx_hold = train_test_split(
    hx, test_size=args.probe_test_size,
    stratify=hx["label"].astype(int), random_state=args.seed)
print(f"[probe] train={len(hx_train)} hold={len(hx_hold)}")

t0 = time.time()
X_tr = extract_last_fast(hx_train["text"].tolist(), desc="probe-train")
y_tr = hx_train["label"].astype(int).to_numpy()
print(f"[probe] hidden {X_tr.shape}  ({time.time()-t0:.1f}s)")

scaler = StandardScaler().fit(X_tr)
clf = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(X_tr), y_tr)
joblib.dump({"scaler": scaler, "clf": clf}, os.path.join(args.out, "probe.pkl"))
print(f"[probe] train_acc={clf.score(scaler.transform(X_tr), y_tr):.4f}")

nph = 250
s1 = hx_hold[hx_hold.label==1].sample(min(nph, int((hx_hold.label==1).sum())),
                                       random_state=args.seed)
s0 = hx_hold[hx_hold.label==0].sample(min(nph, int((hx_hold.label==0).sum())),
                                       random_state=args.seed)
sanity = pd.concat([s1, s0]).reset_index(drop=True)
sanity["source"] = "hatexplain_heldout"
sanity[["text","label","source"]].to_csv(
    os.path.join(args.data_eval, "sanity_hatexplain.csv"), index=False)
print(f"[sanity] {len(sanity)}건 저장 label={np.bincount(sanity['label'].astype(int)).tolist()}")

# ─────────────────────────── 6. B0 (No steering) ───────────────────────────
def predict_no_steer(texts):
    X = extract_last_fast(texts, desc="B0")
    return clf.predict(scaler.transform(X))

pred_latent = predict_no_steer(eval_latent_texts)
b0_latent  = evaluate(pred_latent, eval_latent_labels)
pred_tg = predict_no_steer(toxigen_texts)
b0_tg = evaluate(pred_tg, toxigen_labels)
sa_texts  = sanity["text"].tolist()
sa_labels = sanity["label"].astype(int).to_numpy()
pred_sa   = predict_no_steer(sa_texts)
b0_sa     = evaluate(pred_sa, sa_labels)

b0 = {
    "model": args.model,
    "probe": "logistic_regression",
    "layer": -1,
    "results": {
        "eval_latent_v2":   {**b0_latent, "n": int(len(eval_latent_labels))},
        "eval_toxigen_v1":  {**b0_tg,     "n": int(len(toxigen_labels))},
        "sanity_hatexplain":{**b0_sa,     "n": int(len(sa_labels))},
    }
}
json.dump(b0, open(os.path.join(args.out, "b0_baseline.json"), "w"), indent=2)
print(f"[B0] eval_latent  {b0_latent}")
print(f"[B0] eval_toxigen {b0_tg}")
print(f"[B0] sanity_hx    {b0_sa}")

# ─────────────────────────── 7. FN subset 저장 ───────────────────────────
def save_fn(pred, labels, name):
    fn = np.where((labels == HATE) & (pred == 0))[0]
    np.save(os.path.join(args.out, f"fn_subset_{name}.npy"), fn)
    print(f"[FN] {name}: {len(fn)}/{int((labels==HATE).sum())}")
    return fn

fn_latent = save_fn(pred_latent, eval_latent_labels, "eval_latent_v2")
fn_tg     = save_fn(pred_tg,     toxigen_labels,     "eval_toxigen_v1")

# ─────────────────────────── 8. v_AB 추출 + sanity hook ───────────────────────────
print("[v_AB] Cell A/B hidden 추출")
h_A = extract_hidden(A_texts, keep_layers=list(range(N_LAYERS+1)))
h_B = extract_hidden(B_texts, keep_layers=list(range(N_LAYERS+1)))
def steering_vector(h_src, h_tgt):
    d = h_src.mean(0) - h_tgt.mean(0)
    return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
v_AB = steering_vector(h_A, h_B)
np.save(os.path.join(args.out, "v_AB.npy"), v_AB)
print(f"[v_AB] saved {v_AB.shape}")

L, a = args.sanity_layer, args.sanity_alpha
hook = SteeringHook(v_AB[L], a).attach(model.model.layers[L])
try:
    X_st = extract_hidden(eval_latent_texts)
finally:
    hook.detach()
pred_st = clf.predict(scaler.transform(X_st))
m_st = evaluate(pred_st, eval_latent_labels)
delta = m_st["macro_f1"] - b0_latent["macro_f1"]
verdict = "PASS" if abs(delta) > 1e-6 else "FAIL (hook 미적용)"
print(f"[sanity] v_AB L={L} α={a} macro_f1={m_st['macro_f1']:.4f} "
      f"(Δ{delta:+.4f}) -> {verdict}")

# ─────────────────────────── 9. 보고 3줄 ───────────────────────────
print("="*60)
print(f"1. eval_latent_v2  n={len(eval_latent_labels)}  B0 macro_F1={b0_latent['macro_f1']:.4f}")
print(f"2. eval_toxigen_v1 n={len(toxigen_labels)}  "
      f"groups={tox['target_group'].nunique() if 'target_group' in tox else '?'}  "
      f"B0 macro_F1={b0_tg['macro_f1']:.4f}")
print(f"3. v_AB sanity: L={L} α={a}  Δ={delta:+.4f}  ({verdict})")
print(f"   FN subset: latent={len(fn_latent)} toxigen={len(fn_tg)}")
print("="*60)
