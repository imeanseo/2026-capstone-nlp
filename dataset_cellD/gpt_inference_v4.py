import argparse
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from prompts_v4 import PROMPT

load_dotenv()


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

    for i, (_, row) in enumerate(infer_df.iterrows(), start=1):
        input_text = str(row[args.input_column])

        # Golden few-shot: 예시 1 (v2 #8 gold) + 예시 2 (직접 작성)
        history = [
            # --- 골든 예시 1 ---
            {"role": "user", "content": "Analyze the input: 'she not gonna love niggers that for sure if i ever saw that shit i would stomp that nigger to death'"},
            {"role": "assistant", "content": "- Target Group: niggers\n- Replacement Noun Phrase: people\n- Sentence Structure (clause / voice / tense / aspect / negation): declarative / active / present / simple / negation\n- Step 1 Cues (REMOVE): shit -> REMOVE\n- Step 2 Cues (NEUTRALIZE): love -> negative emotion verb -> notice; stomp -> violence verb -> consider; to death -> multi-word violence phrase -> REMOVE; gonna -> intensifier -> REMOVE\n- Frame-carrying tokens to preserve: she, not, that, for sure, if, i, ever, saw, that, would"},
            {"role": "user", "content": "Step 1: produce the B-like intermediate."},
            {"role": "assistant", "content": "She is not going to love people that for sure if I ever saw that I would stomp that person."},
            {"role": "user", "content": "Step 2: neutralize the remaining cues."},
            {"role": "assistant", "content": "She is not going to notice people that for sure if I ever saw that I would consider that person."},
            {"role": "user", "content": "Check and output the final Cell D sentence."},
            {"role": "assistant", "content": "She is not going to notice people that for sure if I ever saw that I would consider that person."},

            # --- 골든 예시 2 ---
            {"role": "user", "content": "Analyze the input: 'those stupid blacks always destroy everything they touch'"},
            {"role": "assistant", "content": "- Target Group: blacks\n- Replacement Noun Phrase: individuals\n- Sentence Structure (clause / voice / tense / aspect / negation): declarative / active / present / habitual / none\n- Step 1 Cues (REMOVE): (none)\n- Step 2 Cues (NEUTRALIZE): stupid -> strong negative adjective -> different; destroy -> violence verb -> approach; always -> intensifier -> REMOVE\n- Frame-carrying tokens to preserve: those, everything they touch"},
            {"role": "user", "content": "Step 1: produce the B-like intermediate."},
            {"role": "assistant", "content": "Those stupid individuals always destroy everything they touch."},
            {"role": "user", "content": "Step 2: neutralize the remaining cues."},
            {"role": "assistant", "content": "Those different individuals approach everything they touch."},
            {"role": "user", "content": "Check and output the final Cell D sentence."},
            {"role": "assistant", "content": "Those different individuals approach everything they touch."},
        ]
        turn_outputs = {}
        generated_text = ""

        print(f"\n[{i}/{len(infer_df)}] multi-turn inferencing...")

        for turn_idx, prompt_key in enumerate(args.prompt_keys, start=1):
            prompt_template = PROMPT[prompt_key]

            row_dict = SafeDict(row.to_dict())
            row_dict["input_text"] = input_text

            # 이전 turn 결과를 {turn_1}, {turn_2}, ... 형태로 prompt에서 사용 가능
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

            generated_text = turn_text

            print(f"  turn {turn_idx} success: {prompt_key}")

            time.sleep(args.sleep_sec)


        result_row = {
            "text_clean": input_text,
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
