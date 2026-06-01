#!/usr/bin/env python3
"""
Latent Hatred (implicit-hate) raw 다운로드 및 사용 가능 컬럼 추출.

공식 repo: https://github.com/SALT-NLP/implicit-hate
데이터 zip: README의 Dropbox 링크 (Twitter API 대체용 *_posts.tsv 포함)

라이선스 (corpus LICENSE, CC BY 4.0):
  - post 텍스트, class / implicit_class 라벨 추출·연구 사용 가능
  - 배포·공유 시 출처 표기 필요 (ElSherief et al., EMNLP 2021)
  - stg3 target/implied_statement 등 추가 컬럼은 본 스크립트에서 추출하지 않음

출력:
  steering_vector/data/latent_hatred/latent_hatred_posts.csv  — post, class [, implicit_class]
  steering_vector/data/latent_hatred/latent_hatred_prepare.log
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "steering_vector" / "data" / "latent_hatred"
RAW_DIR = DATA_DIR / "raw"
CORPUS_DIR = RAW_DIR / "implicit-hate-corpus"
ZIP_PATH = RAW_DIR / "implicit-hate-corpus.zip"
DROPBOX_URL = (
    "https://www.dropbox.com/s/p1ctnsg3xlnupwr/implicit-hate-corpus.zip?dl=1"
)
GITHUB_LICENSE_URL = (
    "https://raw.githubusercontent.com/SALT-NLP/implicit-hate/master/LICENSE"
)

STG1_POSTS = CORPUS_DIR / "implicit_hate_v1_stg1_posts.tsv"
STG2_POSTS = CORPUS_DIR / "implicit_hate_v1_stg2_posts.tsv"
DEFAULT_OUTPUT = DATA_DIR / "latent_hatred_posts.csv"
DEFAULT_LOG = DATA_DIR / "latent_hatred_prepare.log"

LICENSE_NOTE = """\
License summary (corpus: CC BY 4.0, repo code: MIT)
- Usable for research: post, class, implicit_class (fine-grained, implicit rows only)
- Attribution required when sharing derivatives
- Citation: ElSherief et al. (2021) Latent Hatred, EMNLP
  https://aclanthology.org/2021.emnlp-main.29/
"""


def download_raw(skip: bool) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if skip and STG1_POSTS.is_file():
        print(f"[skip-download] found {STG1_POSTS}")
        return

    print(f"Downloading {DROPBOX_URL}")
    urlretrieve(DROPBOX_URL, ZIP_PATH)
    print(f"Extracting {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(RAW_DIR)

    if not (CORPUS_DIR / "LICENSE").is_file():
        raise FileNotFoundError(f"Expected corpus LICENSE under {CORPUS_DIR}")

    repo_license = DATA_DIR / "LICENSE.repo"
    try:
        urlretrieve(GITHUB_LICENSE_URL, repo_license)
    except OSError as exc:
        print(f"[warn] could not fetch repo LICENSE: {exc}", file=sys.stderr)


def extract(with_implicit_class: bool) -> pd.DataFrame:
    if not STG1_POSTS.is_file():
        raise FileNotFoundError(
            f"Missing {STG1_POSTS}. Run without --skip-download first."
        )

    stg1 = pd.read_csv(STG1_POSTS, sep="\t")
    required = {"post", "class"}
    missing = required - set(stg1.columns)
    if missing:
        raise ValueError(f"stg1_posts missing columns: {sorted(missing)}")

    out = stg1[["post", "class"]].copy()

    dup_n = int(out["post"].duplicated().sum())
    if dup_n:
        print(f"[warn] duplicate post texts in stg1: {dup_n} — keeping first")
        out = out.drop_duplicates(subset="post", keep="first").reset_index(drop=True)

    if with_implicit_class:
        stg2 = pd.read_csv(STG2_POSTS, sep="\t")
        if "implicit_class" not in stg2.columns:
            raise ValueError("stg2_posts missing implicit_class column")
        out = out.merge(
            stg2[["post", "implicit_class"]].drop_duplicates(subset="post", keep="first"),
            on="post",
            how="left",
        )

    return out


def log_stats(df: pd.DataFrame, log_path: Path) -> None:
    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines.append(f"=== Latent Hatred prepare log ({ts}) ===")
    lines.append("")
    lines.append(LICENSE_NOTE.rstrip())
    lines.append("")
    lines.append(f"source_stg1: {STG1_POSTS.relative_to(REPO)}")
    lines.append(f"rows_total: {len(df)}")
    lines.append("")
    lines.append("label_distribution (class):")
    for label, cnt in df["class"].value_counts().sort_index().items():
        pct = 100.0 * cnt / len(df)
        lines.append(f"  {label}: {cnt} ({pct:.2f}%)")

    if "implicit_class" in df.columns:
        implicit_rows = df["class"] == "implicit_hate"
        filled = df.loc[implicit_rows, "implicit_class"].notna().sum()
        lines.append("")
        lines.append(
            f"implicit_class filled: {filled}/{implicit_rows.sum()} "
            f"(implicit_hate rows)"
        )
        lines.append("implicit_class distribution (non-null only):")
        for label, cnt in (
            df.loc[df["implicit_class"].notna(), "implicit_class"]
            .value_counts()
            .sort_index()
            .items()
        ):
            lines.append(f"  {label}: {cnt}")

    text = "\n".join(lines) + "\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Latent Hatred dataset")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="raw zip이 이미 있으면 다운로드 생략",
    )
    parser.add_argument(
        "--no-implicit-class",
        action="store_true",
        help="implicit_class 컬럼 제외 (post, class 만)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    download_raw(skip=args.skip_download)
    df = extract(with_implicit_class=not args.no_implicit_class)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {args.output} ({len(df)} rows, cols={list(df.columns)})")

    log_stats(df, args.log)
    print(f"Log saved: {args.log}")


if __name__ == "__main__":
    main()
