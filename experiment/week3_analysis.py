#!/usr/bin/env python
"""
Week 3 — 해석 고정.
sweep 결과(coarse/fine) 로드 → best 셋업 확정 → ablation → 유의성 검정
→ axis attribution 연결 → 최종 표/그래프 저장.

사용 예:
  python week3_analysis.py
  python week3_analysis.py --best-layer-AB 22 --best-alpha-AB 2.0   # 강제 지정
  python week3_analysis.py --eval latent                            # 한쪽만
"""
import os, re, json, sys, argparse
import numpy as np, pandas as pd, torch, joblib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from scipy.stats import binomtest
from tqdm.auto import tqdm

from huggingface_hub import login
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

# ─────────────────────────── CLI ───────────────────────────
P = argparse.ArgumentParser()
P.add_argument("--model", default="meta-llama/Llama-3.2-3B")
P.add_argument("--out",  default="results")
P.add_argument("--data-eval", default="data/eval")
P.add_argument("--src",  default="src/eval")
P.add_argument("--cell-c", default="cell_c_test_final.csv")
P.add_argument("--batch", type=int, default=32)
P.add_argument("--max-len", type=int, default=128)
P.add_argument("--seed", type=int, default=42)
P.add_argument("--eval", choices=["both","latent","tg"], default="both")
P.add_argument("--bootstrap-n", type=int, default=2000)
# best 셋업 강제 지정 (안 주면 sweep CSV에서 자동 선택)
P.add_argument("--best-layer-AB", type=int, default=None)
P.add_argument("--best-alpha-AB", type=float, default=None)
P.add_argument("--best-layer-AC", type=int, default=None)
P.add_argument("--best-alpha-AC", type=float, default=None)
P.add_argument("--harm-layers", default="4,5,9,10,11,13,14,18,19,20,21,22,23,24,25,26,27",
               help="0528 axis attribution (참고용, 코드 동작엔 영향 없음)")
P.add_argument("--hatexplain-csv", default="data/train/hatexplain_train.csv",
               help="Layer probe 학습용 HateXplain train CSV")
P.add_argument("--layer-probe-only", action="store_true",
               help="§1 Layer probe만 실행 (Mac 로컬, ~15분)")
P.add_argument("--probe-swap-only", action="store_true",
               help="probe 학습 데이터 교체 실험 (hatexplain/latent/toxigen × both eval)")
P.add_argument("--random-seed-only", action="store_true",
               help="B1 random seed 반복 실험만 실행 (~25분/eval, Mac MPS)")
P.add_argument("--rand-seeds", type=int, default=20,
               help="B1 random seed 반복 횟수 (기본 20)")
args = P.parse_args()
_FAST_ONLY = args.layer_probe_only or args.probe_swap_only or args.random_seed_only

# ── device: Mac(MPS) 우선, Windows CUDA는 주석 참고 ──
# Windows + NVIDIA GPU:
# device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
_model_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
_mode = (" layer_probe_only" if args.layer_probe_only else
         " probe_swap_only" if args.probe_swap_only else
         " random_seed_only" if args.random_seed_only else "")
print(f"[cfg] device={device} dtype={_model_dtype} batch={args.batch} eval={args.eval}{_mode}")

# ─────────────────────────── 모델 ───────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=_model_dtype).to(device).eval()
model.config.output_hidden_states = False
N_LAYERS = model.config.num_hidden_layers
 
# ─────────────────────────── 자산 로드 ───────────────────────────
sys.path.insert(0, args.src)
from metrics import evaluate, evaluate_by_group, HATE
 
pb = joblib.load(os.path.join(args.out, "probe.pkl"))
scaler, clf = pb["scaler"], pb["clf"]
b0 = json.load(open(os.path.join(args.out, "b0_baseline.json")))
b0m = {
    "eval_latent_v2":  b0["results"]["eval_latent_v2"],
    "eval_toxigen_v1": b0["results"]["eval_toxigen_v1"],
}
 
def load_eval(path, group=False):
    df = pd.read_csv(path)
    # 라벨이 숫자 dtype이 아니면 문자열 매핑 (object/string/ArrowString 모두 안전)
    if not pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(str).str.strip().str.lower().map(
            {"hate":1,"non-hate":0})
    df["label"] = df["label"].astype(int)
    g = df["target_group"].tolist() if (group and "target_group" in df) else None
    return df["text"].tolist(), df["label"].to_numpy(), g
 
EV = {
    "eval_latent_v2":  load_eval(os.path.join(args.data_eval, "eval_latent_v2.csv")),
    "eval_toxigen_v1": load_eval(os.path.join(args.data_eval, "eval_toxigen_v1.csv"),
                                  group=True),
}
FN = {tag: np.load(os.path.join(args.out, f"fn_subset_{tag}.npy")) for tag in EV}
VECTORS = {n: np.load(os.path.join(args.out, f"{n}.npy"))
           for n in ["v_AB","v_AC","v_random","v_harm"]}
 
def load_sweep(tag):
    c = pd.read_csv(os.path.join(args.out, f"sweep_coarse_{tag}.csv"))
    fp = os.path.join(args.out, f"sweep_fine_{tag}.csv")
    return pd.concat([c, pd.read_csv(fp)], ignore_index=True) if os.path.exists(fp) else c
 
TAGS = ["eval_latent_v2","eval_toxigen_v1"] if args.eval=="both" \
       else (["eval_latent_v2"] if args.eval=="latent" else ["eval_toxigen_v1"])
SW = {tag: load_sweep(tag) for tag in TAGS}
_b0summary = ", ".join(f"{k}={v['macro_f1']:.4f}" for k,v in b0m.items())
print(f"[loaded] B0 {{{_b0summary}}}")
 
# ─────────────────────────── 공용 함수 ───────────────────────────
_ABL = {}
 
class AblHook:
    """mode: last / first / all / none"""
    def __init__(self, vec, alpha, mode="last"):
        self.v = torch.tensor(vec, dtype=torch.float16, device=device)
        self.a = float(alpha); self.mode = mode; self.h = None
    def _fn(self, m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if self.mode == "last":
            h[:, -1, :] = h[:, -1, :] + self.a * self.v
        elif self.mode == "first":
            fi = _ABL["first"]; B = h.shape[0]
            ar = torch.arange(B, device=h.device)
            h[ar, fi, :] = h[ar, fi, :] + self.a * self.v
        elif self.mode == "all":
            h[:, :, :] = h + self.a * self.v
        return o
    def attach(self, layer): self.h = layer.register_forward_hook(self._fn); return self
    def detach(self):
        if self.h is not None: self.h.remove(); self.h = None
 
def prebatch(texts):
    texts = [str(t) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    inv = np.argsort(order)
    B = [tok([texts[i] for i in order[s:s+args.batch]], return_tensors="pt",
             padding=True, truncation=True, max_length=args.max_len)
         for s in range(0, len(order), args.batch)]
    return B, inv
 
@torch.inference_mode()
def extract(batches, inv, vec=None, L=None, alpha=0.0, mode="none"):
    hk = AblHook(vec[L], alpha, mode).attach(model.model.layers[L]) if mode != "none" else None
    out = []
    try:
        for enc in batches:
            enc = {k: v.to(device) for k, v in enc.items()}
            if mode == "first":
                _ABL["first"] = (enc["attention_mask"] == 0).sum(1)
            out.append(model.model(**enc, use_cache=False,
                                   output_hidden_states=False
                                   ).last_hidden_state[:, -1, :].float().cpu().numpy())
    finally:
        if hk: hk.detach()
    return np.concatenate(out, 0)[inv]
 
def predict(X): return clf.predict(scaler.transform(X))
def recovery(pred, fn): return 0.0 if len(fn)==0 else float((pred[fn]==HATE).mean())
 
def mcnemar(y, p0, p1):
    c0 = (p0==y); c1 = (p1==y)
    b = int(np.sum(c0 & ~c1)); c = int(np.sum(~c0 & c1))
    p = binomtest(min(b,c), b+c, 0.5).pvalue if (b+c)>0 else 1.0
    return dict(b_only_B0=b, c_only_E1=c, p_value=float(p))
 
def boot_delta_f1(y, p0, p1, n=None, seed=None):
    n = n or args.bootstrap_n; seed = seed or args.seed
    rng = np.random.RandomState(seed); idx = np.arange(len(y)); d = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        d.append(f1_score(y[s], p1[s], average="macro") -
                 f1_score(y[s], p0[s], average="macro"))
    lo, med, hi = np.percentile(d, [2.5, 50, 97.5])
    return dict(lo=float(lo), med=float(med), hi=float(hi))
 
# ─────────────────────────── 1. Best 셋업 ───────────────────────────
def best_of(df, name):
    sub = df[df.vector==name]; b = sub.loc[sub.macro_f1.idxmax()]
    return int(b.layer), float(b.alpha), float(b.macro_f1)
 
BEST = {tag: {n: best_of(SW[tag], n) for n in VECTORS} for tag in SW}
 
# 강제 지정 override
def override(tag, name, L_arg, a_arg):
    if L_arg is None and a_arg is None: return
    L, a, _ = BEST[tag][name]
    L = L_arg if L_arg is not None else L
    a = a_arg if a_arg is not None else a
    row = SW[tag][(SW[tag].vector==name) & (SW[tag].layer==L) & (SW[tag].alpha==a)]
    if len(row) == 0:
        print(f"⚠️  [{tag}] {name} L={L} α={a} sweep에 없음 — 강제 무시")
        return
    BEST[tag][name] = (int(L), float(a), float(row.iloc[0].macro_f1))
    print(f"   [{tag}] {name} → L={L} α={a} (CLI override)")
 
for tag in TAGS:
    override(tag, "v_AB", args.best_layer_AB, args.best_alpha_AB)
    override(tag, "v_AC", args.best_layer_AC, args.best_alpha_AC)
 
for tag in BEST:
    print(f"[best {tag}] B0={b0m[tag]['macro_f1']:.4f}")
    for n,(L,a,f) in BEST[tag].items():
        print(f"   {n:9s} L={L:2d} α={a:>5} F1={f:.4f} (Δ{f-b0m[tag]['macro_f1']:+.4f})")
 
@torch.inference_mode()
def extract_all_layers(texts):
    out = []
    for s in range(0, len(texts), args.batch):
        enc = tok([str(t) for t in texts[s:s+args.batch]], return_tensors="pt",
                  padding=True, truncation=True, max_length=args.max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        hs = model.model(**enc, use_cache=False,
                         output_hidden_states=True).hidden_states
        out.append(np.stack([h[:, -1, :].float().cpu().numpy() for h in hs], axis=1))
    return np.concatenate(out, 0)

# ─────────────────────────── 2. Ablation ───────────────────────────
def run_ablation(tag):
    texts, labels, _ = EV[tag]; fn = FN[tag]; base = b0m[tag]["macro_f1"]
    batches, inv = prebatch(texts); rows = []
    for name in ["v_AB","v_AC"]:
        L, a, _ = BEST[tag][name]
        settings = [
            ("main (last,+α)", a,   "last"),
            ("α=0",           0.0, "last"),
            ("sign flip (−α)",-a,  "last"),
            ("first-token",    a,  "first"),
            ("all-token",      a,  "all"),
        ]
        for slabel, al, mode in settings:
            X = extract(batches, inv, VECTORS[name], L, al, mode)
            pred = predict(X); m = evaluate(pred, labels)
            rows.append(dict(vector=name, L=L, setting=slabel,
                             macro_f1=round(m["macro_f1"], 4),
                             d_f1=round(m["macro_f1"]-base, 4),
                             fn_rec=round(recovery(pred, fn), 4)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(args.out, f"ablation_{tag}.csv"), index=False)
    print(f"\n===== ABLATION [{tag}] (B0={base:.4f}) =====\n{t.to_string(index=False)}")
    return t
 
if not _FAST_ONLY:
    for tag in TAGS:
        run_ablation(tag)
 
# ─────────────────────────── 3. 유의성 검정 ───────────────────────────
sig = {}
if not _FAST_ONLY:
    for tag in TAGS:
        texts, labels, _ = EV[tag]; batches, inv = prebatch(texts)
        p0 = predict(extract(batches, inv, mode="none"))
        L, a, _ = BEST[tag]["v_AB"]
        p1 = predict(extract(batches, inv, VECTORS["v_AB"], L, a, "last"))
        mc = mcnemar(labels, p0, p1); bt = boot_delta_f1(labels, p0, p1)
        f0 = f1_score(labels, p0, average="macro")
        f1v = f1_score(labels, p1, average="macro")
        sig[tag] = dict(B0_macro_f1=round(f0,4), E1_macro_f1=round(f1v,4),
                        delta=round(f1v-f0, 4), mcnemar=mc, bootstrap_delta_ci=bt,
                        best=dict(layer=L, alpha=a))
        print(f"[sig {tag}] B0={f0:.4f} → E1={f1v:.4f} (Δ{f1v-f0:+.4f}) | "
              f"McNemar p={mc['p_value']:.2e} (b={mc['b_only_B0']}, c={mc['c_only_E1']}) | "
              f"ΔF1 95%CI [{bt['lo']:+.4f}, {bt['hi']:+.4f}]")
    json.dump(sig, open(os.path.join(args.out, "significance.json"), "w"), indent=2)
 
# ─────────────────────────── 4. Axis attribution ───────────────────────────
_harm_path = os.path.join(args.out, "harm_gap_z_lasttoken.npy")
if _FAST_ONLY and os.path.exists(_harm_path):
    harm_gap_z = np.load(_harm_path)
    print(f"\n[axis] 기존 harm_gap_z 로드 ({_harm_path})")
else:
    print("\n[axis] last-token harm-축 곡선 재산출")
    _dfc = pd.read_csv(args.cell_c).dropna(subset=["cell_a","cell_c_modified"])
    Atx = _dfc["cell_a"].tolist(); Ctx = _dfc["cell_c_modified"].tolist()
    hA = extract_all_layers(Atx); hC = extract_all_layers(Ctx)
    vh = VECTORS["v_harm"]
    pa = np.einsum("nlh,lh->nl", hA, vh)
    pc = np.einsum("nlh,lh->nl", hC, vh)
    pooled = np.concatenate([pa, pc], 0).std(0) + 1e-8
    harm_gap_z = (pa.mean(0) - pc.mean(0)) / pooled
    np.save(_harm_path, harm_gap_z)
    order = np.argsort(-np.abs(harm_gap_z))
    print(f"   last-token harm |z| top10: {order[:10].tolist()}")
    print(f"   0528 mean-pool reference  : {args.harm_layers}")
    for tag in BEST:
        for n in ["v_AB","v_AC"]:
            L = BEST[tag][n][0]
            rank = int(np.where(order==L)[0][0]) + 1
            print(f"   [{tag}] {n} best L={L} | last-token |z| rank {rank} (z={harm_gap_z[L]:+.3f})")
 
# ─────────────────────────── 5. 그래프 ───────────────────────────
def graphA_axis(tag):
    df = SW[tag]; base = b0m[tag]["macro_f1"]
    fig, ax = plt.subplots(figsize=(11,5))
    for n, lab in [("v_AB","E1 v_AB"),("v_AC","E2 v_AC"),
                   ("v_random","B1 rand"),("v_harm","B2 harm")]:
        sub = df[df.vector==n]
        ba = sub.loc[sub.macro_f1.idxmax(), "alpha"]
        s = sub[sub.alpha==ba].sort_values("layer")
        ax.plot(s.layer, s.macro_f1, marker="o", ms=3, label=f"{lab} (α={ba})")
    ax.axhline(base, color="k", ls="--", label="B0")
    ax.set_xlabel("layer"); ax.set_ylabel("macro F1")
    ax.legend(fontsize=8, loc="upper left")
    ax2 = ax.twinx()
    xs = range(len(harm_gap_z))
    ax2.plot(xs, np.abs(harm_gap_z), color="orange", lw=1.6, ls=":", alpha=0.8)
    ax2.fill_between(xs, np.abs(harm_gap_z), color="orange", alpha=0.08)
    ax2.set_ylabel("harm-axis |A-C z| (last-token)", color="orange")
    ax.set_title(f"Graph A + harm-axis [{tag}]")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"graphA_axis_{tag}.png"), dpi=150)
    plt.close()
 
def graphB(tag):
    df = SW[tag]; base = b0m[tag]["macro_f1"]
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    for ax, n, title in zip(axes, ["v_AB","v_AC"],
                             ["E1 v_AB (target)","E2 v_AC (cue)"]):
        sub = df[df.vector==n]
        L = int(sub.loc[sub.macro_f1.idxmax(), "layer"])
        s = sub[(sub.layer==L) & (sub.alpha > 0)].sort_values("alpha")
        ax.plot(s.alpha, s.macro_f1, marker="o")
        ax.axhline(base, color="k", ls="--", label="B0")
        ax.set_title(f"{title}\nbest layer {L} [{tag}]")
        ax.set_xlabel("steering strength α"); ax.set_ylabel("macro F1")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"graphB_{tag}.png"), dpi=150)
    plt.close()
 
if not _FAST_ONLY:
    for tag in TAGS:
        graphA_axis(tag); graphB(tag)
        print(f"[graph] graphA_axis_{tag}.png, graphB_{tag}.png 저장")

# ── §1 Layer별 Linear Probe + probe 학습 데이터 교체 ────────────────
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler as SS

_TRAIN_DIR = os.path.join(os.path.dirname(args.data_eval.rstrip(os.sep)), "train")
PROBE_SOURCES = {
    "hatexplain": args.hatexplain_csv,
    "latent":     os.path.join(_TRAIN_DIR, "latent_train.csv"),
    "toxigen":    os.path.join(_TRAIN_DIR, "toxigen_train.csv"),
}
SWAP_EVAL_TAGS = ["eval_latent_v2", "eval_toxigen_v1"]

def load_probe_df(path):
    df = pd.read_csv(path)
    if not pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(str).str.strip().str.lower().map(
            {"hate": 1, "non-hate": 0, "1": 1, "0": 0})
    df["label"] = df["label"].astype(int)
    return df

def layer_probe_scores(h_train, y_train, h_eval, labels):
    scores = {}
    for L in range(N_LAYERS + 1):
        sc_L = SS().fit(h_train[:, L, :])
        clf_L = LR(C=1.0, max_iter=1000)
        clf_L.fit(sc_L.transform(h_train[:, L, :]), y_train)
        scores[L] = evaluate(
            clf_L.predict(sc_L.transform(h_eval[:, L, :])), labels
        )["macro_f1"]
    return scores

def save_layer_probe_plot(layer_scores, tag, steer_L, out_png, title):
    peak_L = max(layer_scores, key=layer_scores.get)
    layers = list(layer_scores.keys())
    f1s = list(layer_scores.values())
    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    ax1.plot(layers, f1s, color="#2563eb", lw=2, marker="o", ms=5,
             label="Layer probe macro F1", zorder=3)
    ax1.scatter([peak_L], [layer_scores[peak_L]], s=140, c="#16a34a",
                marker="*", zorder=5, label=f"Probe peak L{peak_L} ({layer_scores[peak_L]:.3f})")
    ax1.scatter([steer_L], [layer_scores[steer_L]], s=90, c="#dc2626",
                marker="D", zorder=5, edgecolors="white", linewidths=0.8,
                label=f"Steering best L{steer_L} ({layer_scores[steer_L]:.3f})")
    ax1.axvline(steer_L, color="#dc2626", ls="--", lw=2, alpha=0.85, zorder=2)
    ax1.annotate(f"Steering best\nL{steer_L}",
                 xy=(steer_L, layer_scores[steer_L]),
                 xytext=(steer_L + 1.8, layer_scores[steer_L] + 0.018),
                 fontsize=9, color="#dc2626", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2))
    ax1.annotate(f"Probe peak\nL{peak_L}",
                 xy=(peak_L, layer_scores[peak_L]),
                 xytext=(peak_L - 5.5, layer_scores[peak_L] + 0.012),
                 fontsize=9, color="#16a34a", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.2))
    ax1.set_xlabel("Layer"); ax1.set_ylabel("Probe macro F1")
    ax1.set_xlim(-0.5, N_LAYERS + 0.5)
    ax1.grid(True, alpha=0.25, ls=":")
    ax2 = ax1.twinx()
    ax2.plot(range(len(harm_gap_z)), np.abs(harm_gap_z),
             color="#ea580c", lw=1.8, ls=":", alpha=0.85, label="harm |z|")
    ax2.set_ylabel("harm-axis |z|", color="#ea580c")
    ax1.set_title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return peak_L

def run_layer_probe_for_source(probe_name, probe_path, eval_tags, csv_prefix, png_prefix):
    summary = []
    pending = [t for t in eval_tags
               if not (os.path.exists(os.path.join(args.out, f"{csv_prefix}_{probe_name}_{t}.csv"))
                       and os.path.exists(os.path.join(args.out, f"{png_prefix}_{probe_name}_{t}.png")))]
    h_pr = y_pr = None
    if pending:
        pr_df = load_probe_df(probe_path)
        print(f"[layer_probe] {probe_name} hidden 추출 (n={len(pr_df)})")
        h_pr = extract_all_layers(pr_df["text"].tolist())
        y_pr = pr_df["label"].to_numpy()
        print(f"   shape={h_pr.shape}")
    for tag in eval_tags:
        csv_path = os.path.join(args.out, f"{csv_prefix}_{probe_name}_{tag}.csv")
        png_path = os.path.join(args.out, f"{png_prefix}_{probe_name}_{tag}.png")
        steer_L = BEST[tag]["v_AB"][0]
        if os.path.exists(csv_path) and os.path.exists(png_path):
            df = pd.read_csv(csv_path)
            layer_scores = dict(zip(df.layer.astype(int), df.macro_f1))
            peak_L = int(df.loc[df.macro_f1.idxmax(), "layer"])
            print(f"[layer_probe] {probe_name} → {tag} 기존 결과 재사용")
            summary.append(dict(probe_train=probe_name, eval_set=tag,
                                peak_layer=peak_L, peak_f1=round(layer_scores[peak_L], 4),
                                steer_layer=steer_L,
                                steer_probe_f1=round(layer_scores[steer_L], 4),
                                last_f1=round(layer_scores[N_LAYERS], 4)))
            continue
        texts, labels, _ = EV[tag]
        print(f"[layer_probe] {probe_name} → {tag} eval hidden (N={len(labels)})")
        h_ev = extract_all_layers(texts)
        layer_scores = layer_probe_scores(h_pr, y_pr, h_ev, labels)
        pd.DataFrame({"layer": list(layer_scores.keys()),
                      "macro_f1": list(layer_scores.values())}).to_csv(csv_path, index=False)
        peak_L = save_layer_probe_plot(
            layer_scores, tag, steer_L, png_path,
            f"Layer probe ({probe_name} train) → {tag}")
        summary.append(dict(probe_train=probe_name, eval_set=tag,
                            peak_layer=peak_L, peak_f1=round(layer_scores[peak_L], 4),
                            steer_layer=steer_L,
                            steer_probe_f1=round(layer_scores[steer_L], 4),
                            last_f1=round(layer_scores[N_LAYERS], 4)))
        print(f"   peak L={peak_L} F1={layer_scores[peak_L]:.4f} | "
              f"steer L={steer_L} F1={layer_scores[steer_L]:.4f}")
    return summary

if args.probe_swap_only:
    print("\n[probe_swap] probe 학습 데이터 교체 실험")
    all_rows = []
    for probe_name, probe_path in PROBE_SOURCES.items():
        if not os.path.exists(probe_path):
            print(f"[probe_swap] {probe_name} 파일 없음 — 스킵: {probe_path}")
            print(f"             먼저 실행: python build_probe_train_csvs.py")
            continue
        all_rows.extend(run_layer_probe_for_source(
            probe_name, probe_path, SWAP_EVAL_TAGS,
            "probe_swap", "probe_swap"))
    if all_rows:
        t = pd.DataFrame(all_rows)
        t.to_csv(os.path.join(args.out, "probe_swap_summary.csv"), index=False)
        print(f"\n===== PROBE SWAP SUMMARY =====\n{t.to_string(index=False)}")
    print("\n[probe_swap done]")
    sys.exit(0)

if not args.layer_probe_only and not args.random_seed_only:
    hx_tr = load_probe_df(args.hatexplain_csv)
    print("\n[layer_probe] HateXplain train hidden 추출")
    h_hx_all = extract_all_layers(hx_tr["text"].tolist())
    y_hx = hx_tr["label"].to_numpy()
    print(f"   train={len(y_hx)}  shape={h_hx_all.shape}")
    for tag in TAGS:
        texts, labels, _ = EV[tag]
        print(f"[layer_probe] {tag} eval hidden 추출 (N={len(labels)})")
        h_ev_all = extract_all_layers(texts)
        layer_scores = layer_probe_scores(h_hx_all, y_hx, h_ev_all, labels)
        pd.DataFrame({"layer": list(layer_scores.keys()),
                      "macro_f1": list(layer_scores.values())}).to_csv(
            os.path.join(args.out, f"layer_probe_{tag}.csv"), index=False)
        steer_L = BEST[tag]["v_AB"][0]
        peak_L = save_layer_probe_plot(
            layer_scores, tag, steer_L,
            os.path.join(args.out, f"layer_probe_{tag}.png"),
            f"Layer probe accuracy + harm axis [{tag}]")
        print(f"[layer_probe] {tag} 저장 완료 → layer_probe_{tag}.csv/.png")
        print(f"   peak L={peak_L}  F1={layer_scores[peak_L]:.4f} | "
              f"last L={N_LAYERS}  F1={layer_scores[N_LAYERS]:.4f}")
elif args.layer_probe_only:
    run_layer_probe_for_source(
        "hatexplain", args.hatexplain_csv, TAGS, "layer_probe", "layer_probe")
    print("\n[layer_probe done]")
    sys.exit(0)
 
# ─────────────────────────── 6. 최종 표 ───────────────────────────
def final_table(tag):
    df = SW[tag]; base = b0m[tag]
    label = {"v_random":"B1 Random","v_harm":"B2 v_harm",
             "v_AB":"E1 v_AB (target)","v_AC":"E2 v_AC (cue)"}
    rows = [dict(Setup="B0 No steering", layer="—", alpha="—",
                 macro_f1=round(base["macro_f1"],4),
                 hate_recall=round(base["hate_recall"],4),
                 fn_recovery="—", d_f1=0.0)]
    for n in ["v_random","v_harm","v_AB","v_AC"]:
        L, a, f = BEST[tag][n]
        r = df[(df.vector==n)&(df.layer==L)&(df.alpha==a)].iloc[0]
        rows.append(dict(Setup=label[n], layer=L, alpha=a,
                         macro_f1=round(f, 4),
                         hate_recall=round(r.hate_recall, 4),
                         fn_recovery=round(r.fn_recovery, 4),
                         d_f1=round(f - base["macro_f1"], 4)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(args.out, f"final_table_{tag}.csv"), index=False)
    print(f"\n===== FINAL [{tag}] =====\n{t.to_string(index=False)}")
    return t
 
if not args.random_seed_only:
    for tag in TAGS:
        final_table(tag)

# ── B1 Random Seed 반복 실험: 추가이득 평균 ± SD ──────────────────
def run_random_seed_experiment(tag):
    texts, labels, _ = EV[tag]
    batches, inv = prebatch(texts)

    L_ab, a_ab, _ = BEST[tag]["v_AB"]
    L_ac, a_ac, _ = BEST[tag]["v_AC"]
    X_e1 = extract(batches, inv, VECTORS["v_AB"], L_ab, a_ab, "last")
    X_e2 = extract(batches, inv, VECTORS["v_AC"], L_ac, a_ac, "last")
    f1_e1 = evaluate(predict(X_e1), labels)["macro_f1"]
    f1_e2 = evaluate(predict(X_e2), labels)["macro_f1"]

    records = []
    for seed in tqdm(range(args.rand_seeds), desc=f"rand_seed [{tag}]", unit="seed"):
        rng = np.random.default_rng(seed)
        v_rand = rng.standard_normal((N_LAYERS + 1, model.config.hidden_size)).astype(np.float32)
        norms = np.linalg.norm(v_rand, axis=-1, keepdims=True)
        v_rand = v_rand / np.where(norms > 0, norms, 1.0)

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

    b1_arr  = df["b1_f1"].values
    de1_arr = df["delta_e1_b1"].values
    de2_arr = df["delta_e2_b1"].values
    pct_e1 = (de1_arr > 0).mean() * 100
    pct_e2 = (de2_arr > 0).mean() * 100

    print(f"\n===== [{tag}] RANDOM SEED SUMMARY ({args.rand_seeds} seeds) =====")
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
    ax.set_title(f"[{tag}] B1 distribution vs fixed E1/E2")
    ax.legend(fontsize=8)

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
    ax2.set_title(f"[{tag}] Incremental gain (E - B1)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(args.out, f"random_seed_experiment_{tag}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[rand_seed plot] {out_path}")

if not (args.layer_probe_only or args.probe_swap_only):
    for tag in TAGS:
        df_rand = run_random_seed_experiment(tag)
        plot_random_seed_results(df_rand, tag)

print("\n[all done]")