#!/usr/bin/env python3
"""
Cell A 고품질 부분집합 생성.

입력: results/cell_a_anchors_v2_framed.csv (정본)
출력: 기본 results/cell_a_high_quality.csv — UTF-8, pandas 기본 따옴표(쉼표/개행 필드 이스케이프).

관련 분석·구현 (repo scripts/):
  - hatexplain_cellA_EDA.ipynb + hatexplain_cellA_EDA_conclusions.md
      합의·길이·슬러·NRC·프레이밍 미검출 비율, OLS/군집 요약. unanimous 쪽 슬러 경향 등.
  - select_cellA.ipynb
      앵커 파이프라인 본체: Filter 1~7 + (구)Filter 8 슬러 개수 상한.
      VALID_GROUP_TARGETS, get_target_tokens, classify_framing과 동일 전제.
  - framing_detection_rules.py — 프레이밍 룰 정의(노트북과 동기화 시 참고).
  - minimal_pair_pilot_v5.ipynb — Cell A 메타 컬럼·검증(VADER 등) 사용 예.

규칙 (체크리스트 + 위 분석과 정렬):
  - Filter 1: hatespeech / offensive 만, split 제외
  - 합의: 기본 unanimous (EDA 결론: 경계 사례 완화; --allow-majority)
  - Filter 2: 토큰 수 10~60
  - Filter 3: 유효 타겟 1~3개 (VALID_GROUP_TARGETS, Other 제외)
  - Filter 5: non_slur_cue >= 1
  - Filter 6: 프레이밍 NONE_DETECTED 제외 (앵커에 이미 반영된 행만 존재)
  - Filter 7: target_tokens >= 1
  - 명시적 혐오: slur_tokens >= 1 (HurtLex 기반 태깅; EDA §7과 정합)
  - 멀티 타겟: 타겟 2개 이상이면 target_tokens도 2개 이상 (지칭어-집단 정합)
  - select_cellA Filter 8: --slur-token-cap N (기본 0=비활성; 3이면 노트북과 동일 «슬러 4개 이상 제외»)
  - 수동 제외: scripts/cell_a_high_quality_denylist.txt (post_id 한 줄에 하나, # 주석 가능)
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO / "results" / "cell_a_anchors_v2_framed.csv"
DEFAULT_OUTPUT = REPO / "results" / "cell_a_high_quality.csv"
MANUAL_DENYLIST_PATH = REPO / "scripts" / "cell_a_high_quality_denylist.txt"


def load_manual_denylist() -> frozenset[str]:
    """수동 제외 post_id 목록 (한 줄에 하나, # 으로 시작하는 줄은 무시)."""
    path = MANUAL_DENYLIST_PATH
    if not path.is_file():
        print(f"[warn] manual denylist missing: {path}", file=sys.stderr)
        return frozenset()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line.split()[0])
    return frozenset(out)


VALID_LABELS = frozenset({"hatespeech", "offensive"})
VALID_TARGETS = frozenset(
    {
        "African",
        "Islam",
        "Jewish",
        "Women",
        "Refugee",
        "Homosexual",
        "Arab",
        "Latino_Americans",
        "Asian",
    }
)


def parse_list_cell(s) -> list:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    if isinstance(s, list):
        return s
    s = str(s).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except (SyntaxError, ValueError):
        return []


def token_len(row) -> int:
    return len(parse_list_cell(row.get("tokens")))


def valid_targets_list(targets: list) -> list:
    return [t for t in targets if t in VALID_TARGETS and t != "Other"]


def passes_filters(
    row,
    allow_majority: bool,
    denylist: frozenset[str],
    slur_token_cap: int,
) -> tuple[bool, str]:
    pid = str(row.get("post_id", "")).strip()
    if pid in denylist:
        return False, "denylist"

    lab = str(row.get("majority_label", "")).strip().lower()
    if lab not in VALID_LABELS:
        return False, "label"

    agr = str(row.get("agreement", "")).strip().lower()
    if agr == "split":
        return False, "split"
    if agr != "unanimous" and not allow_majority:
        return False, "agreement_majority"

    ntok = token_len(row)
    if not (10 <= ntok <= 60):
        return False, "token_len"

    targets = parse_list_cell(row.get("targets"))
    vt = valid_targets_list(targets)
    if not vt:
        return False, "no_valid_target"
    if len(vt) > 3:
        return False, "too_many_targets"

    slurs = parse_list_cell(row.get("slur_tokens"))
    if len(slurs) < 1:
        return False, "no_slur"
    if slur_token_cap > 0 and len(slurs) > slur_token_cap:
        return False, "slur_count_cap"

    non_slur = parse_list_cell(row.get("non_slur_cue_tokens"))
    if len(non_slur) < 1:
        return False, "no_non_slur_cue"

    ttoks = parse_list_cell(row.get("target_tokens"))
    if len(ttoks) < 1:
        return False, "no_target_token"

    if len(vt) >= 2 and len(ttoks) < 2:
        return False, "multi_target_single_lex"

    framing = parse_list_cell(row.get("framing"))
    pf = str(row.get("primary_framing", "")).strip()
    if pf and framing and pf not in framing:
        return False, "primary_not_in_framing"
    if not framing or framing == ["NONE_DETECTED"]:
        return False, "framing_none"

    return True, "ok"


def main() -> int:
    p = argparse.ArgumentParser(description="Build high-quality Cell A CSV from anchors.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--allow-majority",
        action="store_true",
        help="다수결(majority) 합의 행도 포함 (기본은 unanimous만)",
    )
    p.add_argument("--max-rows", type=int, default=0, help="상한(0이면 전부)")
    p.add_argument(
        "--denylist",
        type=str,
        default="",
        help="추가 제외 post_id 쉼표구분",
    )
    p.add_argument(
        "--slur-token-cap",
        type=int,
        default=0,
        metavar="N",
        help="슬러 토큰 개수 상한(0=비활성). select_cellA.ipynb Filter 8과 맞추려면 3",
    )
    args = p.parse_args()

    deny = set(load_manual_denylist())
    if args.denylist.strip():
        deny.update(x.strip() for x in args.denylist.split(",") if x.strip())

    if not args.input.is_file():
        print(f"Missing input: {args.input}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.input, encoding="utf-8")
    if "post_id" not in df.columns:
        print("CSV missing post_id", file=sys.stderr)
        return 1

    reasons: dict[str, int] = {}
    keep_mask = []
    for _, row in df.iterrows():
        ok, reason = passes_filters(
            row,
            args.allow_majority,
            frozenset(deny),
            args.slur_token_cap,
        )
        reasons[reason] = reasons.get(reason, 0) + 1
        keep_mask.append(ok)

    out = df.loc[keep_mask].copy()
    if args.max_rows and args.max_rows > 0:
        out = out.head(args.max_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(
        args.output,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )

    n_in, n_out = len(df), len(out)
    print(f"input rows: {n_in}")
    print(f"output rows: {n_out} ({100 * n_out / n_in:.1f}% of input)")
    print(f"written: {args.output}")
    print("drop reasons (all input rows):")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    if n_out:
        print("output majority_label:", out["majority_label"].value_counts().to_dict())
        print("output agreement:", out["agreement"].value_counts().to_dict())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
