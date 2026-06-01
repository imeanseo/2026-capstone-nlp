"""
Cell B 후처리 게이트 (post-processing gate)
==============================================
Hard fail  → 재생성 필요   (자동 플래그)
Soft fail  → 수동 검토 필요 (자동 플래그)
통과       → OK

판정 기준
----------
[Hard 1] Verbatim cue 누락 (비폭력·비사망 polarity만)
  - VERBATIM_PRESERVE 토큰 중 원문에 있는 것이 출력에 없으면 Hard fail.
  - kill / die / stomp / rape 등 **대인 폭력·사망 화행**은 새 Cell B 목표상
    출력에서 빠질 수 있음 → 이 목록에는 넣지 않음.

[Hard 1b] 대인 폭력·사망 화행 잔존
  - 출력에 여전히 stomp…to death, hope…die 등 패턴이 남으면 Hard fail (목표 위반).

[Hard 2] Identity leakage (HurtLex)
  - HurtLex stereotype=yes 고위험 카테고리 어휘가 Cell B에 잔존하면 Hard fail.
  - HURTLEX_EXCLUSION으로 비인간 맥락 false positive 제외 (savage, political, advocate)

[Soft 1] VADER compound diff
  - |compound(B) - compound(A)| > VADER_DIFF_THRESH (기본 0.50)
  - 감성 강도가 크게 약화됐음을 의미.

[Soft 2] NRC anger+disgust 커버리지 감소
  - 원문 대비 출력의 NRC 부정 감정(anger, disgust) 토큰 비율이
    NRC_DROP_THRESH(기본 0.75) 이상 감소하면 Soft fail.
  - orig_cov==1이면 완전 소실(0)일 때만 SOFT.

[Soft 3] Generic phrase 단조 반복
  - 동일 generic phrase 3회 이상 등장 시 Soft fail.
  - 대상: "certain people", "some people", "those people", "that person",
           "some individuals", "certain individuals"

사용법
------
python postprocess.py --input cell_b_test_v3.csv --output cell_b_postcheck.csv
"""

import argparse
import re
import warnings
warnings.filterwarnings("ignore")

from nltk.stem import PorterStemmer

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

# ── 설정 ────────────────────────────────────────────────────────────────────
VADER_DIFF_THRESH = 0.50   # VADER compound 차이 임계값 (soft fail)
NRC_DROP_THRESH   = 0.75   # NRC anger+disgust 커버리지 감소율 (soft fail, 0.60→0.75)
GENERIC_REPEAT_N  = 3      # generic phrase 반복 임계 횟수 (soft fail)

REFUSAL_TRIGGERS = [
    "i'm sorry",
    "i cannot",
    "i can't assist",
    "i can't help",
    "i am unable",
    "i apologize",
    "as an ai",
    "i'm not able",
]

# HurtLex identity check에서 제외 — 비인간 도메인 맥락 false positive
IDENTITY_LEAKAGE_EXCLUSIONS = {"political", "advocate", "savage"}
HURTLEX_EXCLUSION = IDENTITY_LEAKAGE_EXCLUSIONS

# 비폭력 polarity만 verbatim 강제 (대인 살해·폭행 동사는 Cell B에서 제거될 수 있음)
VERBATIM_PRESERVE = {
    "dumb", "stupid", "idiot", "fool", "retarded", "scum", "garbage",
    "worthless", "awful", "evil",
    "shit", "slop", "stench", "traitor", "skank",
}

# 출력에 남아 있으면 안 되는 **대인 폭력·사망** 패턴 (소문자 기준)
RESIDUAL_VIOLENCE_RES = [
    r"stomp\w*.{0,40}\bto\s+death\b",
    r"beat\w*.{0,40}\bto\s+death\b",
    r"\bkick\w*.{0,20}\bto\s+death\b",
    r"\bhope\b.{0,50}\b(all\s+)?die\b",
    r"\bhope\b.{0,40}\bdie\s+from\b",
    r"\bgas\s+these\b",
    r"\braping\b",
    r"\bexterminate\b",
]

GENERIC_PHRASES = [
    "certain people", "some people", "those people",
    "that person", "some individuals", "certain individuals",
    "those individuals",
]

# HurtLex 고위험 카테고리 — 진짜 identity-bound 어휘만
# pa  = personal attack (인종·집단 공격성 명사)
# pr  = profanity with identity connotation
# rci = racial, cultural, immigration slurs
# asm = attacks on sexual minorities
# asf = attacks on gender/sex minorities (female-targeted)
# ddf, ddp, is 등 일반 부정어 카테고리는 false positive가 너무 많아 제외
HURTLEX_HIGH_RISK_CATS = {"pa", "pr", "rci", "asm", "asf"}


# ── 사전 로딩 ────────────────────────────────────────────────────────────────

def load_hurtlex(path: Path) -> set:
    """stereotype=yes 고위험 카테고리 lemma 집합 반환"""
    df = pd.read_csv(path, sep="\t")
    mask = (df["category"].isin(HURTLEX_HIGH_RISK_CATS)) & (df["stereotype"] == "yes")
    words = set(df[mask]["lemma"].str.lower().str.strip())
    return words


def load_nrc(path: Path) -> tuple[set, set]:
    """NRC에서 anger=1, disgust=1 단어 집합 반환"""
    anger, disgust = set(), set()
    for line in open(path):
        parts = line.strip().split("\t")
        if len(parts) != 3 or parts[2] != "1":
            continue
        if parts[1] == "anger":
            anger.add(parts[0].lower())
        elif parts[1] == "disgust":
            disgust.add(parts[0].lower())
    return anger, disgust


def load_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        print("[WARNING] vaderSentiment 미설치 (pip install vaderSentiment) → VADER 검사 스킵")
        return None


# ── 토크나이저 (단순 소문자 단어 분리) ──────────────────────────────────────

_stemmer = PorterStemmer()

# Porter stemmer가 다루지 못하는 -y 형용사형 등 수동 매핑
VERBATIM_VARIANTS: dict[str, str] = {
    "scummy": "scum",
    "shitty": "shit",
    "crappy": "crap",
    "dumbass": "dumb",
    "stupidly": "stupid",
    "evilness": "evil",
    "worthlessly": "worthless",
    "scummiest": "scum",
    "shittiest": "shit",
}

COMPOUND_SPLIT_MAP = {
    "dumbass": ["dumb"],
    "asshole": ["hole"],
    "shithead": ["shit"],
    "fuckface": ["fuck"],
    "dipshit": ["shit"],
    "jackass": ["ass"],
}


def tokenize(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower()))


def tokenize_stemmed(text: str) -> set:
    """소문자 토큰 + Porter stem + 수동 변형 매핑 (scummy→scum 등)."""
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    stems = {_stemmer.stem(t) for t in tokens}
    variant_bases = {VERBATIM_VARIANTS[t] for t in tokens if t in VERBATIM_VARIANTS}
    return tokens | stems | variant_bases


# ── 검사 함수 ────────────────────────────────────────────────────────────────

def check_verbatim(orig: str, out: str) -> dict:
    """[Hard 1] VERBATIM_PRESERVE: 원문 cue가 출력에 exact 또는 stem으로 있는지"""
    orig_toks = tokenize(orig)
    out_toks_full = tokenize_stemmed(out)
    cues_in_orig = VERBATIM_PRESERVE & orig_toks
    missing = {
        w
        for w in cues_in_orig
        if w not in out_toks_full and _stemmer.stem(w) not in out_toks_full
    }
    return {
        "verbatim_cues_orig": sorted(cues_in_orig),
        "verbatim_cues_missing": sorted(missing),
        "hard_verbatim": bool(missing),
    }


def check_residual_violence(out: str) -> dict:
    """[Hard 1b] 출력에 대인 폭력·사망 화행이 남았는지"""
    low = out.lower()
    if "can't assist" in low or "cannot assist" in low or "i'm sorry" in low:
        return {"residual_violence_patterns": [], "hard_violence": False}
    hits = [pat for pat in RESIDUAL_VIOLENCE_RES if re.search(pat, low, re.DOTALL)]
    return {
        "residual_violence_patterns": hits,
        "hard_violence":              bool(hits),
    }


def check_refusal(out: str) -> dict:
    """API 거절 출력 여부 (verbatim/violence gate와 별도 집계)."""
    low = out.lower().strip()
    detected = any(trigger in low for trigger in REFUSAL_TRIGGERS)
    return {"refusal_flag": detected, "refusal_detected": detected}


def check_identity_leakage(out: str, hurtlex_set: set) -> dict:
    """[Hard 2] HurtLex identity 어휘가 출력에 잔존하는지 확인
    VERBATIM_PRESERVE·HURTLEX_EXCLUSION은 검사 대상에서 제외."""
    out_toks = tokenize(out)
    check_set = hurtlex_set - VERBATIM_PRESERVE - HURTLEX_EXCLUSION
    leaked = out_toks & check_set
    return {
        "identity_leaked_tokens": sorted(leaked),
        "hard_identity":          bool(leaked),
    }


def check_vader(orig: str, out: str, analyzer) -> dict:
    """[Soft 1] VADER compound 차이"""
    if analyzer is None:
        return {"vader_orig": None, "vader_out": None,
                "vader_diff": None, "soft_vader": False}
    va = analyzer.polarity_scores(orig)["compound"]
    vb = analyzer.polarity_scores(out)["compound"]
    diff = abs(vb - va)
    return {
        "vader_orig": round(va, 4),
        "vader_out":  round(vb, 4),
        "vader_diff": round(diff, 4),
        "soft_vader": diff > VADER_DIFF_THRESH,
    }


def nrc_tokens_expanded(text: str) -> list[str]:
    """토큰화 후 합성어는 분해하여 NRC 카운트 확장."""
    toks = re.findall(r"[a-z]+", text.lower())
    expanded: list[str] = []
    for t in toks:
        expanded.append(t)
        if t in COMPOUND_SPLIT_MAP:
            expanded.extend(COMPOUND_SPLIT_MAP[t])
    return expanded


def nrc_negative_token_set(text: str) -> set:
    return set(nrc_tokens_expanded(text))


def check_nrc_coverage(orig: str, out: str,
                        nrc_anger: set, nrc_disgust: set) -> dict:
    """[Soft 2] NRC anger+disgust 커버리지 감소율"""
    nrc_neg = nrc_anger | nrc_disgust
    orig_toks = nrc_negative_token_set(orig)
    out_toks = nrc_negative_token_set(out)
    orig_cov = len(orig_toks & nrc_neg)
    out_cov = len(out_toks & nrc_neg)
    if orig_cov == 0:
        drop_rate = 0.0
        soft_nrc = False
    elif orig_cov == 1:
        # 단일 NRC hit edge case: 완전 소실만 SOFT
        drop_rate = round(1 - out_cov / orig_cov, 4)
        soft_nrc = out_cov == 0
    else:
        drop_rate = round(1 - out_cov / orig_cov, 4)
        soft_nrc = drop_rate > NRC_DROP_THRESH
    return {
        "nrc_angry_disgust_orig": orig_cov,
        "nrc_angry_disgust_out":  out_cov,
        "nrc_drop_rate":          drop_rate,
        "soft_nrc":               soft_nrc,
    }


def check_generic_repeat(out: str) -> dict:
    """[Soft 3] generic phrase 3회 이상 반복"""
    out_l   = out.lower()
    repeats = {p: out_l.count(p) for p in GENERIC_PHRASES if out_l.count(p) >= GENERIC_REPEAT_N}
    return {
        "generic_repeat": repeats,
        "soft_repeat":    bool(repeats),
    }


# ── 종합 판정 ────────────────────────────────────────────────────────────────

TYPE_STRICT_VIOLENCE = {"THREAT_VIOLENCE"}
TYPE_STRICT_VERBATIM = {"DIRECT_INSULT"}


def judge(row: dict, hate_type: str = "") -> str:
    flags = []
    ht = (hate_type or row.get("hate_type") or "").strip().upper()

    if row.get("refusal_flag"):
        flags.append("REFUSAL:api_refusal")

    if ht in TYPE_STRICT_VIOLENCE and row.get("hard_violence"):
        flags.append("HARD:residual_violence(strict)")
    if ht in TYPE_STRICT_VERBATIM and row.get("verbatim_cues_missing"):
        flags.append(
            f"HARD:verbatim_missing({','.join(row['verbatim_cues_missing'])})"
        )

    if row["hard_verbatim"] and ht not in TYPE_STRICT_VERBATIM:
        flags.append(f"HARD:verbatim_missing({','.join(row['verbatim_cues_missing'])})")
    if row.get("hard_violence"):
        flags.append("HARD:residual_person_violence")
    if row["hard_identity"]:
        flags.append(f"HARD:identity_leaked({','.join(row['identity_leaked_tokens'][:5])})")
    if row["soft_vader"]:
        flags.append(f"SOFT:vader_diff={row['vader_diff']}")
    if row["soft_nrc"]:
        flags.append(f"SOFT:nrc_drop={row['nrc_drop_rate']}")
    if row["soft_repeat"]:
        flags.append(f"SOFT:generic_repeat={row['generic_repeat']}")
    return " | ".join(flags) if flags else "PASS"


# ── 메인 ────────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str | None = None):
    hurtlex_path = REPO / "lexicons" / "hurtlex_EN.tsv"
    nrc_path     = REPO / "NRC-Emotion-Lexicon" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

    print("사전 로딩 중...")
    hurtlex_set         = load_hurtlex(hurtlex_path)
    nrc_anger, nrc_dis  = load_nrc(nrc_path)
    vader               = load_vader()
    print(f"  HurtLex identity words : {len(hurtlex_set)}")
    print(f"  NRC anger              : {len(nrc_anger)}")
    print(f"  NRC disgust            : {len(nrc_dis)}")
    print()

    df = pd.read_csv(input_path)
    hate_types = None
    if "text_clean" in df.columns:
        cols = ["text_clean", "generated_text"]
        if "hate_type" in df.columns:
            cols.append("hate_type")
        work = df[cols].copy()
        if "hate_type" in work.columns:
            hate_types = work["hate_type"].tolist()
            work = work[["text_clean", "generated_text"]]
        work.columns = ["original", "output"]
        df = work
    elif "original_text" in df.columns:
        df = df[["original_text", "output_text"]].copy()
        df.columns = ["original", "output"]
    df = df.dropna(subset=["original", "output"])

    results = []
    for pos, (_, row) in enumerate(df.iterrows()):
        orig = str(row["original"])
        out  = str(row["output"])
        idx  = f"S{pos+1}"
        ht = ""
        if hate_types is not None and pos < len(hate_types):
            ht = str(hate_types[pos]) if pd.notna(hate_types[pos]) else ""

        r = {"idx": idx}
        r["text_clean"] = orig
        r["generated_text"] = out

        r.update(check_verbatim(orig, out))
        r.update(check_refusal(out))
        r.update(check_residual_violence(out))
        r.update(check_identity_leakage(out, hurtlex_set))
        r.update(check_vader(orig, out, vader))
        r.update(check_nrc_coverage(orig, out, nrc_anger, nrc_dis))
        r.update(check_generic_repeat(out))
        r["verdict"] = judge(r, hate_type=ht)

        results.append(r)
        status = "🚫 REFUSAL" if r.get("refusal_flag") else (
            "❌ HARD" if r.get("hard_verbatim") or r.get("hard_violence") or r.get("hard_identity")
            else ("⚠️  SOFT" if r["soft_vader"] or r["soft_nrc"] or r["soft_repeat"] else "✅ PASS")
        )
        print(f"[{idx}] {status}  {r['verdict'][:80]}")

    result_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("요약")
    print("=" * 80)
    total = len(result_df)
    hard  = result_df["verdict"].str.contains("HARD", na=False).sum()
    soft  = ((~result_df["verdict"].str.contains("HARD", na=False)) &
              result_df["verdict"].str.contains("SOFT", na=False)).sum()
    ok    = (result_df["verdict"] == "PASS").sum()
    refusal = result_df["refusal_flag"].sum() if "refusal_flag" in result_df.columns else 0
    print(f"  PASS      : {ok}/{total}")
    print(f"  SOFT fail : {soft}/{total}  (수동 검토)")
    print(f"  HARD fail : {hard}/{total}  (재생성 필요)")
    print(f"  Refusal   : {refusal}/{total}  (별도 플래그)")

    if output_path:
        save_cols = [
            "idx", "text_clean", "generated_text",
            "verbatim_cues_orig", "verbatim_cues_missing", "hard_verbatim",
            "refusal_flag", "refusal_detected",
            "residual_violence_patterns", "hard_violence",
            "identity_leaked_tokens", "hard_identity",
            "vader_orig", "vader_out", "vader_diff", "soft_vader",
            "nrc_angry_disgust_orig", "nrc_angry_disgust_out",
            "nrc_drop_rate", "soft_nrc",
            "generic_repeat", "soft_repeat",
            "verdict",
        ]
        save_cols = [c for c in save_cols if c in result_df.columns]
        result_df[save_cols].to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n결과 저장: {output_path}")

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell B 후처리 게이트")
    parser.add_argument("--input",  required=True,  help="cell_b_test*.csv")
    parser.add_argument("--output", default=None,   help="결과 저장 경로 (선택)")
    args = parser.parse_args()
    run(args.input, args.output)
