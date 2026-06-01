import argparse
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from prompts import PROMPT

load_dotenv()


class SafeDict(dict):
    def __missing__(self, key):
        return ""


def parse_args():
    parser = argparse.ArgumentParser(description="Simple inference script.")

    parser.add_argument("--input", required=True, help="input CSV path")
    parser.add_argument("--output", required=True, help="output CSV path")
    parser.add_argument("--prompt-key", required=True, help="prompt key in prompts.py")

    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--input-column", default="text_clean", help="input text column")
    parser.add_argument("--output-column", default="generated_text", help="output column name")
    parser.add_argument("--n", type=int, default=0, help="number of rows; 0 means all")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--max-output-tokens", type=int, default=1024)

    return parser.parse_args()


def make_prompt(row, prompt_template, input_column):
    row_dict = SafeDict(row.to_dict())
    input_text = row_dict[input_column]
    row_dict["input_text"] = row_dict[input_column]

    return prompt_template.format_map(row_dict)


def call_api(client, model,system_prompt, prompt, max_output_tokens):
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )

    return response.output_text.strip()


def run_inference(args):
    if args.prompt_key not in PROMPT:
        raise KeyError(
            f"'{args.prompt_key}' not found in PROMPT. "
            f"Available keys: {list(PROMPT.keys())}"
        )


    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY 환경 변수를 설정하세요. 예: export OPENAI_API_KEY='sk-...'"
        )
    client = OpenAI()

    df = pd.read_csv(args.input)

    if args.n and args.n > 0:
        infer_df = df.head(args.n).copy()
    else:
        infer_df = df.copy()

    system_prompt = PROMPT["system"]
    prompt_template = PROMPT[args.prompt_key]

    print(f"input rows: {len(df)}")
    print(f"inference rows: {len(infer_df)}")
    print(f"model: {args.model}")
    print(f"prompt key: {args.prompt_key}")

    results = []

    for i, (_, row) in enumerate(infer_df.iterrows(), start=1):
        input_text = str(row[args.input_column])
        prompt = make_prompt(
            row=row,
            prompt_template=prompt_template,
            input_column=args.input_column,
        )

        generated_text = ""

        print(f"\n[{i}/{len(infer_df)}] inferencing...")

        for attempt in range(1, args.max_retries + 2):
            attempts = attempt

            try:
                generated_text = call_api(
                    client=client,
                    model=args.model,
                    system_prompt = system_prompt,
                    prompt=prompt,
                    max_output_tokens=args.max_output_tokens,
                )
                break

            except Exception as e:
                error = str(e)
                print(f"  attempt {attempt} failed: {error}")

                if attempt <= args.max_retries:
                    time.sleep(args.sleep_sec)


        results.append({
                "text_clean": input_text,
                "generated_text": generated_text,
            })

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