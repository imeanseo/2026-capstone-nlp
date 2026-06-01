import argparse
import ast
import os
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

from prompts_v10_domain import PROMPT, pick_domain


class SafeDict(dict):
    def __missing__(self, key):
        return ""


def parse_args():
    parser = argparse.ArgumentParser(description="Simple inference script.")

    parser.add_argument("--input", required=True, help="input CSV path")
    parser.add_argument("--output", required=True, help="output CSV path")
    parser.add_argument("--prompt-keys", nargs="+", required=True, help="prompt keys in prompts.py, e.g. analyze generate revise")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--input-column", default="text_clean", help="input text column")
    parser.add_argument("--output-column", default="generated_text", help="output column name")
    parser.add_argument("--n", type=int, default=0, help="number of rows; 0 means all")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--max-output-tokens", type=int, default=1024)

    return parser.parse_args()


def safe_parse_list(value):
    if isinstance(value, list):
        return value

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    if value == "":
        return []

    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else [parsed]
    except (ValueError, SyntaxError):
        return [str(value)]


def get_original_token_count(row, input_text):
    if "tokens" in row.index:
        tokens = safe_parse_list(row.get("tokens"))
        if tokens:
            return len(tokens)

    try:
        token_count = int(float(row.get("token_count", 0) or 0))
        if token_count > 0:
            return token_count
    except Exception:
        pass

    return len(str(input_text).split())


def make_prompt(row, prompt_template, input_column):
    row_dict = SafeDict(row.to_dict())
    input_text = row_dict[input_column]
    row_dict["input_text"] = row_dict[input_column]

    return prompt_template.format_map(row_dict)


def call_api(client, model, system_prompt, history, max_output_tokens):
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=history,
        max_output_tokens=max_output_tokens,
    )

    return response.output_text.strip()


def extract_final_sentence(text: str) -> str:
    if not text:
        return ""

    import re

    text = text.strip()

    pattern = re.compile(
        r'(?:^|\n)\s*(?:[#*\-\s]*)'
        r'\*{0,2}'
        r'Final\s+(?:Transformed\s+)?Sentence'
        r'\*{0,2}'
        r'\s*:\s*'
        r'(.*)',
        re.IGNORECASE
    )

    match = pattern.search(text)
    if match:
        raw = match.group(1).strip()

        if not raw or raw in ('**', '*'):
            after = text[match.end():]
            lines = [line.strip() for line in after.splitlines() if line.strip()]
            if lines:
                raw = lines[0]
            else:
                raw = ""

        raw = re.sub(r'^\*{1,3}\s*|\s*\*{1,3}$', '', raw).strip()
        raw = re.sub(r'^\*{1,3}\s*|\s*\*{1,3}$', '', raw).strip()
        raw = raw.strip('"').strip("'").strip()
        return raw

    return text.strip().strip('"').strip("'")


def count_ws_tokens(text: str) -> int:
    return len(str(text).strip().split())


def run_inference(args):
    for key in args.prompt_keys:
        if key not in PROMPT:
            raise KeyError(
                f"'{key}' not found in PROMPT. "
                f"Available keys: {list(PROMPT.keys())}"
            )

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    df = pd.read_csv(args.input)

    if args.n and args.n > 0:
        infer_df = df.head(args.n).copy()
    else:
        infer_df = df.copy()

    system_prompt = PROMPT["system"]

    print(f"input rows: {len(df)}")
    print(f"inference rows: {len(infer_df)}")
    print(f"model: {args.model}")
    print(f"prompt keys: {args.prompt_keys}")

    results = []

    for i, (idx, row) in enumerate(infer_df.iterrows(), start=1):
        input_text = str(row[args.input_column])

        # ── 도메인 배정 (프레이밍 기반 블랙리스트 적용) ──
        framing = str(row.get("primary_framing", row.get("framing", "")))
        domain = pick_domain(framing=framing, seed=hash(input_text) % (2**31))

        history = [
            {
                "role": "user",
                "content": (
                    "For each input, transform that specific sentence. "
                    "Do not reuse wording from previous examples. "
                    "Preserve the input's clause structure as much as possible."
                )
            },
            {
                "role": "assistant",
                "content": "Understood."
            }
        ]
        turn_outputs = {}
        generated_text = ""

        print(f"\n[{i}/{len(infer_df)}] multi-turn inferencing...")
        print(f"  domain: {domain}")
        print(f"  input: {input_text}")

        for turn_idx, prompt_key in enumerate(args.prompt_keys, start=1):
            prompt_template = PROMPT[prompt_key]

            row_dict = SafeDict(row.to_dict())
            row_dict["input_text"] = input_text
            row_dict["domain"] = domain

            original_token_count = get_original_token_count(row, input_text)

            row_dict["target_token_count"] = original_token_count
            row_dict["min_token_count"] = max(1, round(original_token_count * 0.85))
            row_dict["max_token_count"] = max(1, round(original_token_count * 1.15))

            row_dict.update(turn_outputs)

            prompt = prompt_template.format_map(row_dict)

            history.append({
                "role": "user",
                "content": prompt
            })

            turn_text = ""

            for attempt in range(1, args.max_retries + 2):
                try:
                    turn_text = call_api(
                        client=client,
                        model=args.model,
                        system_prompt=system_prompt,
                        history=history,
                        max_output_tokens=args.max_output_tokens,
                    )
                    break

                except Exception as e:
                    error = str(e)
                    print(f"  turn {turn_idx} attempt {attempt} failed: {error}")

                    if attempt <= args.max_retries:
                        time.sleep(args.sleep_sec)

            if not turn_text:
                print(f"  turn {turn_idx} failed")
                break

            history.append({
                "role": "assistant",
                "content": turn_text
            })

            turn_outputs[f"turn_{turn_idx}"] = turn_text
            turn_outputs[prompt_key] = turn_text

            if prompt_key == "rewrite":
                original_token_count = get_original_token_count(row, input_text)
                candidate_token_count = count_ws_tokens(turn_text)
                length_diff = candidate_token_count - original_token_count
                length_ratio = candidate_token_count / original_token_count if original_token_count > 0 else 0

                turn_outputs["candidate_token_count"] = candidate_token_count
                turn_outputs["length_diff"] = length_diff
                turn_outputs["length_ratio"] = round(length_ratio, 3)

            if prompt_key == "check":
                generated_text = extract_final_sentence(turn_text)
            else:
                generated_text = turn_text

            print(f"  turn {turn_idx} success: {prompt_key}")
            print(f"  output: {turn_text.replace(chr(10), ' ')}")

            time.sleep(args.sleep_sec)

        result_row = {
            "text_clean": input_text,
            args.output_column: generated_text,
            "domain": domain,
        }

        for key in args.prompt_keys:
            if key in turn_outputs:
                result_row[key] = turn_outputs[key]

        for k in range(1, len(args.prompt_keys) + 1):
            turn_key = f"turn_{k}"
            if turn_key in turn_outputs:
                result_row[turn_key] = turn_outputs[turn_key]

        results.append(result_row)

        if generated_text:
            print("  success")
        else:
            print("  failed")

        time.sleep(args.sleep_sec)

    output_df = pd.DataFrame(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f"saved: {args.output}")
    print(f"success: {(output_df[args.output_column] != '').sum()}")
    print(f"failed: {(output_df[args.output_column] == '').sum()}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)