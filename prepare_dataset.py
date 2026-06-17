#!/usr/bin/env python3
"""
P1.7 §E benchmark — dataset prep.

Direct port of the upstream ntkmirror example at
``examples/gsm8k_small.py`` (commit ``da17ca3c`` on
github.com/leochlon/ntkmirror), pinned so the prompt template +
JSONL key names match exactly what ``ntkmirror.load_jsonl_examples``
parses inside the kfp container:

  {"prompt": "Question: ...\\nAnswer:", "completion": " ..."}

Paper-aligned default sizes (64 train / 32 eval) reproduce the
"small" GSM8K setup from the ntkmirror docs/method.md — small
enough to run on a single GPU in a few minutes; large enough to
get a meaningful NLL signal for the §E three-way comparison.

Usage:
  pip install datasets
  python prepare_dataset.py --out-dir runs/gsm8k_small
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=64)
    ap.add_argument("--eval-size", type=int, default=32)
    ap.add_argument("--out-dir", default="runs/gsm8k_small")
    args = ap.parse_args()

    # Late-import so the script can run --help without the heavy dep.
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def write(split: str, n: int, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in ds[split].select(range(n)):
                # Prompt template MUST match upstream verbatim — the
                # ForwardFineTuner score-and-pick step keys on the
                # token stream produced by this exact template.
                prompt = "Question: " + row["question"].strip() + "\nAnswer:"
                # Leading space on the completion separates the
                # generated answer from "Answer:" in the tokenizer.
                completion = " " + row["answer"].strip()
                f.write(
                    json.dumps(
                        {"prompt": prompt, "completion": completion},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    write("train", args.train_size, out / "train.jsonl")
    write("test", args.eval_size, out / "eval.jsonl")
    print(f"wrote {args.train_size} train + {args.eval_size} eval to {out}")


if __name__ == "__main__":
    main()
