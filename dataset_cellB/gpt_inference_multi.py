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

        history = [
            # ===== Golden 1: 폭력 화행 완화 (8번 문제 유형 — stomp…to death 금지) =====
            {"role": "user", "content": "she not gonna love niggers that for sure if i ever saw that shit i would stomp that nigger to death"},
            {"role": "assistant", "content": "- Target Span Function: [T1] plural object of love-negation; same referent singular object of conditional threat; repeated target\n- Multiple or Repeated Target Pattern: repeated same target\n- Preservable Polarity Cue Types: sarcastic not gonna love; profanity shit — PRESERVE. stomp/to death bound to harming humans — REPAIR (harm-act, not standalone cue)\n- Non-Preservable Harm Frames: physical assault toward humans (stomp to death) — REPLACE\n- Cue-Target Interaction: hostility + violence verb gains lethal force by binding to racial target; removing target must also remove credible bodily harm toward any human referent\n- Generic Human Replacement: [T1]→[PERSON_1] object then them\n- Harm-Frame Repair Plan: stomp … to death → go off on them (non-violent, high arousal)\n- Reference Coherence Plan: [PERSON_1] first object; them second mention\n- Core Sentence Frame: hostile rejection + verbal blow-up, no homicide framing\n- Length Plan: similar length\n- Transformation Risk: low\n- Minimum Negative Load: shit, for sure"},
            {"role": "user", "content": "rewrite"},
            {"role": "assistant", "content": "She's not gonna love [PERSON_1] that for sure; if I ever saw that shit, I would go off on them."},
            {"role": "user", "content": "check"},
            {"role": "assistant", "content": "She's not gonna love some people that for sure; if I ever saw that shit, I would go off on them."},

            # ===== Golden 2: death-wish → 비사망 고강도 (집단 혐오 화행 완화) =====
            {"role": "user", "content": "i hope they all die from aids today i d be so happy"},
            {"role": "assistant", "content": "- Target Span Function: [T1] plural subject of hope-clause\n- Multiple or Repeated Target Pattern: single\n- Preservable Polarity Cue Types: i hope; i'd be so happy (sarcasm) — PRESERVE. all die / disease frame toward people — REPAIR\n- Non-Preservable Harm Frames: death-wish + disease taunt toward humans — REPLACE\n- Cue-Target Interaction: schadenfreude is realized as literal death-wish; must keep sarcasm without wishing death on [PERSON]\n- Generic Human Replacement: [T1]→[PERSON_1] subject\n- Harm-Frame Repair Plan: all die from aids today → have the worst day today / blow up in their face today (no death, no disease weapon)\n- Reference Coherence Plan: single NP\n- Core Sentence Frame: sarcastic ill-wish, non-lethal\n- Length Plan: similar\n- Transformation Risk: low\n- Minimum Negative Load: hope, happy"},
            {"role": "user", "content": "rewrite"},
            {"role": "assistant", "content": "I hope [PERSON_1] have the worst day today, I'd be so happy."},
            {"role": "user", "content": "check"},
            {"role": "assistant", "content": "I hope certain people have the worst day today, I'd be so happy."},
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
