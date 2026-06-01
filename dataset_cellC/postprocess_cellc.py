# postprocess_cellc.py
# Cell C 품질 게이트
# 판정 기준:
#   HARD: 명시적 혐오 마커 잔존 (slur, 폭력 동사, 강한 욕설)
#   HARD: 타겟 그룹 미복구 ("certain people" 잔존)
#   SOFT: NRC 부정 감정 단어 합산 < NRC_NEG_THRESHOLD → Cell D 근접
#         (근거: Mohammad & Turney, ACL 2013, NRC Word-Emotion Association Lexicon)
#   SOFT: Cell A와 너무 유사 (BERTScore > 0.92)
#   PASS: 위 조건 모두 통과

import argparse
import re
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

BERTSCORE_THRESHOLD_HIGH = 0.92
BERTSCORE_THRESHOLD_LOW = 0.70

NRC_NEG_THRESHOLD = 1
NRC_NEGATIVE_EMOTIONS = ["anger", "fear", "disgust", "negative", "sadness"]
FRAME_FALLBACK_WORDS = ["concern", "issue", "problem", "challenge", "affect"]

EXPLICIT_CUE_PATTERNS = [
    r"\b(dumb|stupid|idiot|retarded|worthless|scum|traitor|evil|awful)\b",
    r"\bgarbage\b(?!\s+can\b)",
    r"\b(shit|crap|slop|stench)\b",
    r"\b(stomp|kill|beat|rape|exterminate|gas)\b",
    r"reckless destructive",
    r"that whole mess",
    r"ruining everything",
]

NEUTRAL_PLACEHOLDER_PATTERNS = [
    r"\bcertain people\b",
    r"\bsome people\b",
    r"\bthose individuals\b",
    r"\bcertain individuals\b",
    r"\bthat person\b",
]


def check_explicit_cues(cell_c: str) -> dict:
    low = cell_c.lower()
    hits = [pat for pat in EXPLICIT_CUE_PATTERNS if re.search(pat, low)]
    return {
        "explicit_cue_hits": hits,
        "hard_explicit_cue": bool(hits),
    }


def check_target_restored(cell_c: str) -> dict:
    low = cell_c.lower()
    hits = [pat for pat in NEUTRAL_PLACEHOLDER_PATTERNS if re.search(pat, low)]
    return {
        "placeholder_hits": hits,
        "hard_target_not_restored": bool(hits),
    }


def check_cell_d_proximity(cell_c: str) -> dict:
    """NRC Lexicon 기반 부정 프레임 잔존 여부 검사."""
    try:
        from nrclex import NRCLex
        tokens = re.findall(r"[a-z']+", cell_c.lower())
        emotion = NRCLex()
        emotion.load_token_list(tokens)
        raw = emotion.raw_emotion_scores
    except Exception:
        return {
            "nrc_neg_total": None,
            "nrc_emotion_scores": {},
            "frame_fallback_hit": False,
            "soft_cell_d_proximity": False,
        }

    neg_total = sum(raw.get(e, 0) for e in NRC_NEGATIVE_EMOTIONS)
    fallback_hit = any(w in cell_c.lower() for w in FRAME_FALLBACK_WORDS)
    soft = (neg_total < NRC_NEG_THRESHOLD) and not fallback_hit

    return {
        "nrc_neg_total": neg_total,
        "nrc_emotion_scores": {e: raw.get(e, 0) for e in NRC_NEGATIVE_EMOTIONS},
        "frame_fallback_hit": fallback_hit,
        "soft_cell_d_proximity": soft,
    }


def check_bertscore_similarity(cell_a: str, cell_c: str) -> dict:
    try:
        from bert_score import score as bscore
        _, _, F1 = bscore([cell_c], [cell_a], lang="en", verbose=False)
        score = round(float(F1[0]), 4)
    except Exception:
        score = None
    if score is None:
        return {
            "bertscore_a_c": None,
            "soft_too_similar": False,
            "soft_too_different": False,
        }
    return {
        "bertscore_a_c": score,
        "soft_too_similar": score > BERTSCORE_THRESHOLD_HIGH,
        "soft_too_different": score < BERTSCORE_THRESHOLD_LOW,
    }


def judge(row: dict) -> str:
    flags = []
    if row.get("hard_explicit_cue"):
        flags.append(f"HARD:explicit_cue({row['explicit_cue_hits'][0][:30]})")
    if row.get("hard_target_not_restored"):
        flags.append("HARD:target_not_restored")
    if row.get("soft_cell_d_proximity"):
        flags.append("SOFT:cell_d_proximity(no_frame_signal)")
    if row.get("soft_too_similar"):
        flags.append(f"SOFT:too_similar_to_A(bs={row['bertscore_a_c']})")
    if row.get("soft_too_different"):
        flags.append(f"SOFT:too_different_from_A(bs={row['bertscore_a_c']})")
    return " | ".join(flags) if flags else "PASS"


def run(input_path: str, output_path: str | None = None, use_bertscore: bool = False):
    df = pd.read_csv(input_path)
    cols = [c for c in ["idx", "cell_a", "cell_b", "cell_c"] if c in df.columns]
    df = df[cols].copy()
    df = df[df["cell_c"].notna() & (df["cell_c"] != "")].reset_index(drop=True)

    print(f"분석 행: {len(df)}")
    results = []

    for i, row in df.iterrows():
        cell_a = str(row["cell_a"])
        cell_b = str(row.get("cell_b", ""))
        cell_c = str(row["cell_c"])
        idx = str(row.get("idx", f"S{i + 1}"))

        r = {"idx": idx, "cell_a": cell_a, "cell_b": cell_b, "cell_c": cell_c}
        r.update(check_explicit_cues(cell_c))
        r.update(check_target_restored(cell_c))
        r.update(check_cell_d_proximity(cell_c))

        if use_bertscore:
            r.update(check_bertscore_similarity(cell_a, cell_c))
        else:
            r["bertscore_a_c"] = None
            r["soft_too_similar"] = False
            r["soft_too_different"] = False

        r["verdict"] = judge(r)
        results.append(r)

        nrc_info = (
            f" nrc_neg={r.get('nrc_neg_total', '?')}"
            if r.get("nrc_neg_total") is not None
            else ""
        )
        status = (
            "❌ HARD"
            if r["hard_explicit_cue"] or r["hard_target_not_restored"]
            else ("⚠️  SOFT" if "SOFT" in r["verdict"] else "✅ PASS")
        )
        print(f"[{idx}] {status}{nrc_info}  {r['verdict'][:80]}")

    res_df = pd.DataFrame(results)

    total = len(res_df)
    hard = res_df["verdict"].str.contains("HARD").sum()
    soft = (
        (~res_df["verdict"].str.contains("HARD"))
        & res_df["verdict"].str.contains("SOFT")
    ).sum()
    ok = (res_df["verdict"] == "PASS").sum()
    print(f"\n{'=' * 60}")
    print(f"PASS  : {ok}/{total}")
    print(f"SOFT  : {soft}/{total}  (수동 검토)")
    print(f"HARD  : {hard}/{total}  (재생성 필요)")
    if "nrc_neg_total" in res_df.columns and res_df["nrc_neg_total"].notna().any():
        avg_nrc = res_df["nrc_neg_total"].dropna().mean()
        print(f"평균 NRC 부정 점수: {avg_nrc:.2f}  (0이면 Cell D 근접)")

    if output_path:
        res_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"결과 저장: {output_path}")

    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell C 후처리 게이트")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--bertscore",
        action="store_true",
        help="BERTScore 유사도 검사 활성화 (느림)",
    )
    args = parser.parse_args()
    run(args.input, args.output, use_bertscore=args.bertscore)
