#!/usr/bin/env python
"""Score a served endpoint for the NTK *style* demo: generate answers for the
held-out pirate eval prompts, flag pirate-register markers, and compute the
same teacher-forced completion-NLL as demo_eval.py / the §F notebook.

Usage:
    python scripts/style_eval.py --endpoint http://.../openai/v1 \
        --eval-jsonl runs/pirate/eval.jsonl --out demo_results/style_base.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
from pathlib import Path

import requests

from demo_eval import (  # noqa: E402 — same directory
    BASE_TOKENIZER,
    discover_served_model_name,
    load_examples,
    served_completion_nll,
)

PIRATE_MARKERS = [
    "arr", "ahoy", "matey", "me hearty", "avast", "ye ", "yarr", "landlubber",
    "savvy", "buccaneer", "scallywag", "shiver me timbers", "plank", "doubloon",
    "pieces of eight", "sea dog", "hoist", "davy jones", "blimey", "'tis",
]
MARKER_RES = [re.compile(r"\b" + re.escape(m.strip()) + r"\b") for m in PIRATE_MARKERS]


def pirate_markers(text: str) -> list[str]:
    low = " " + text.lower() + " "
    return [m for m, rx in zip(PIRATE_MARKERS, MARKER_RES) if rx.search(low)]


def generate(endpoint, model, prompt, api_key=None, max_tokens=120):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {"model": model, "prompt": prompt, "temperature": 0,
            "max_tokens": max_tokens, "stop": ["Question:", "\n"]}
    r = requests.post(endpoint.rstrip("/") + "/completions", headers=headers, json=body, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--eval-jsonl", default="runs/pirate/eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--skip-nll", action="store_true")
    args = ap.parse_args()

    examples = load_examples(args.eval_jsonl)
    model = args.model or discover_served_model_name(args.endpoint, args.api_key)
    print(f"endpoint={args.endpoint} model={model} items={len(examples)}")

    items, n_pirate = [], 0
    for i, ex in enumerate(examples):
        gen = generate(args.endpoint, model, ex["prompt"], args.api_key)
        markers = pirate_markers(gen)
        n_pirate += bool(markers)
        items.append({
            "question": ex["prompt"],
            "gold_completion": ex["completion"],
            "generation": gen,
            "pirate_markers": markers,
            "pirate": bool(markers),
        })
        print(f"[{i + 1}/{len(examples)}] pirate={bool(markers)} markers={markers[:4]}")

    nll = None
    if not args.skip_nll:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
        nll = served_completion_nll(args.endpoint, model, tokenizer, examples, args.api_key)
        print(f"nll={nll:.4f} (ppl {math.exp(nll):.4f})")

    payload = {
        "label": args.label or model,
        "served_model_name": model,
        "endpoint": args.endpoint,
        "eval_jsonl": str(args.eval_jsonl),
        "n_items": len(items),
        "n_pirate": n_pirate,
        "pirate_rate": n_pirate / len(items) if items else 0.0,
        "nll": nll,
        "perplexity": math.exp(nll) if nll is not None else None,
        "captured_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "items": items,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"pirate voice: {n_pirate}/{len(items)} ({payload['pirate_rate']:.0%}) -> {out}")


if __name__ == "__main__":
    main()
