import argparse
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from prompts import PROMPT, pre_mask

load_dotenv()


class SafeDict(dict):
    def __missing__(self, key):
        return ""


def parse_args():
    parser = argparse.ArgumentParser(description="Cell D v2 inference with pre-masking.")

    parser.add_argument("--input", required=True, help="input CSV path")
    parser.add_argument("--output", required=True, help="output CSV path")
    parser.add_argument("--prompt-keys", nargs="+", required=True, help="prompt keys in prompts.py")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--input-column", default="text_clean", help="input text column")
    parser.add_argument("--output-column", default="generated_text", help="output column name")
    parser.add_argument("--n", type=int, default=0, help="number of rows; 0 means all")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--max-output-tokens", type=int, default=1024)

    return parser.parse_args()


def call_api(client, model, system_prompt, history, max_output_tokens):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + history,
        max_tokens=max_output_tokens,
    )
    return response.choices[0].message.content.strip()


def _cell_d_few_shot_messages():
    """Priming example: pre-masked A → analyze → step1 → step2 → check."""
    demo_input_raw = "Women are disgusting parasites and ruin everything."
    demo_input = pre_mask(demo_input_raw)
    demo_turn1 = """- All Demographic Mentions: Women
- Replacement Noun Phrase: students
- Sentence Structure (clause / voice / tense / aspect / negation): declarative / active / present / simple / none
- Step 1 Cues (REMOVE): (none)
- Step 2 Cues (NEUTRALIZE): disgusting -> strong negative adjective -> different; parasites -> (frame-carrying, not a cue); ruin -> violence verb -> influence
- Frame-carrying tokens to preserve: are, everything"""
    demo_turn2 = "Students are disgusting parasites and ruin everything."
    demo_turn3 = "Students are different parasites and influence everything."
    demo_turn4 = "Students are different participants and influence everything."

    return [
        {"role": "user", "content": PROMPT["cell_d_analyze"].format_map(SafeDict(input_text=demo_input))},
        {"role": "assistant", "content": demo_turn1},
        {
            "role": "user",
            "content": PROMPT["cell_d_step1"].format_map(
                SafeDict(input_text=demo_input, turn_1=demo_turn1)
            ),
        },
        {"role": "assistant", "content": demo_turn2},
        {
            "role": "user",
            "content": PROMPT["cell_d_step2"].format_map(
                SafeDict(input_text=demo_input, turn_1=demo_turn1, turn_2=demo_turn2)
            ),
        },
        {"role": "assistant", "content": demo_turn3},
        {
            "role": "user",
            "content": PROMPT["cell_d_check"].format_map(
                SafeDict(input_text=demo_input, turn_3=demo_turn3)
            ),
        },
        {"role": "assistant", "content": demo_turn4},
    ]


def run_inference(args):
    for key in args.prompt_keys:
        if key not in PROMPT:
            raise KeyError(
                f"'{key}' not found in PROMPT. "
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

    print(f"input rows: {len(df)}")
    print(f"inference rows: {len(infer_df)}")
    print(f"model: {args.model}")
    print(f"prompt keys: {args.prompt_keys}")
    print(f"pre-masking: enabled")

    results = []

    for i, (_, row) in enumerate(infer_df.iterrows(), start=1):
        input_text_raw = str(row[args.input_column])
        input_text = pre_mask(input_text_raw)

        use_cell_d_priming = args.prompt_keys[:4] == [
            "cell_d_analyze",
            "cell_d_step1",
            "cell_d_step2",
            "cell_d_check",
        ]
        history = list(_cell_d_few_shot_messages()) if use_cell_d_priming else []
        turn_outputs = {}
        generated_text = ""

        print(f"\n[{i}/{len(infer_df)}] multi-turn inferencing...")
        print(f"  raw: {input_text_raw[:60]}...")
        print(f"  masked: {input_text[:60]}...")

        for turn_idx, prompt_key in enumerate(args.prompt_keys, start=1):
            prompt_template = PROMPT[prompt_key]

            row_dict = SafeDict(row.to_dict())
            row_dict["input_text"] = input_text

            row_dict.update(turn_outputs)

            prompt = prompt_template.format_map(row_dict)

            history.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

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

            history.append(
                {
                    "role": "assistant",
                    "content": turn_text,
                }
            )

            turn_outputs[f"turn_{turn_idx}"] = turn_text
            turn_outputs[prompt_key] = turn_text

            generated_text = turn_text

            print(f"  turn {turn_idx} success: {prompt_key}")

            time.sleep(args.sleep_sec)

        result_row = {
            "text_clean": input_text_raw,
            "text_masked": input_text,
            args.output_column: generated_text,
        }

        result_row.update(turn_outputs)

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
