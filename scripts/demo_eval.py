#!/usr/bin/env python
"""Score a served GSM8K endpoint for the NTK demo: greedy-generation accuracy
plus the same teacher-forced completion-NLL as the §F benchmark notebook.

Writes a JSON payload the demo notebook renders (per-item transcripts,
accuracy, NLL) so the "before" side can be replayed without holding the GPU.

Usage:
    python scripts/demo_eval.py --endpoint http://127.0.0.1:8080/openai/v1 \
        --eval-jsonl runs/gsm8k_small/eval.jsonl --out demo_results/base.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
from pathlib import Path

import requests

BASE_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
ANSWER_RE = re.compile(r"####\s*([-+]?[\d,]*\.?\d+)")
NUMBER_RE = re.compile(r"[-+]?[\d,]*\.?\d+")


def load_examples(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_answer(text):
    """GSM8K answer: the '#### N' marker if present, else the last number."""
    m = ANSWER_RE.search(text)
    if m:
        raw = m.group(1)
    else:
        nums = NUMBER_RE.findall(text)
        if not nums:
            return None
        raw = nums[-1]
    raw = raw.replace(",", "").rstrip(".")
    try:
        val = float(raw)
    except ValueError:
        return None
    return int(val) if val == int(val) else val


def discover_served_model_name(endpoint, api_key=None):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = requests.get(endpoint.rstrip("/") + "/models", headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        raise RuntimeError("served /v1/models returned no models")
    return data[0]["id"]


def generate(endpoint, model, prompt, api_key=None, max_tokens=256):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stop": ["Question:"],
    }
    r = requests.post(endpoint.rstrip("/") + "/completions", headers=headers, json=body, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


# --- teacher-forced NLL, identical method to notebooks/ntk_step5_served_nll.ipynb ---

def completion_token_split(tokenizer, prompt, completion):
    prompt_ids = tokenizer(prompt).input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
    return prompt_ids, completion_ids, prompt_ids + completion_ids


def served_completion_nll(endpoint, model, tokenizer, examples, api_key=None):
    url = endpoint.rstrip("/") + "/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    eff = 0
    total_neg_logprob, total_tokens = 0.0, 0
    for i, ex in enumerate(examples):
        prompt_ids, completion_ids, input_ids = completion_token_split(
            tokenizer, ex["prompt"], ex["completion"])
        if not completion_ids:
            continue
        base_body = {"model": model, "prompt": input_ids,
                     "echo": True, "logprobs": 1, "temperature": 0}
        resp = requests.post(url, headers=headers, json={**base_body, "max_tokens": eff}, timeout=180)
        if resp.status_code == 400 and eff == 0:
            eff = 1
            resp = requests.post(url, headers=headers, json={**base_body, "max_tokens": eff}, timeout=180)
        resp.raise_for_status()
        lp = resp.json()["choices"][0]["logprobs"]
        token_logprobs = lp["token_logprobs"]
        if len(token_logprobs) < len(input_ids):
            raise RuntimeError("served logprobs shorter than input; cannot align completion span")
        comp = token_logprobs[len(prompt_ids):len(input_ids)]
        if any(v is None for v in comp):
            raise RuntimeError("None inside completion-span logprobs")
        total_neg_logprob += -float(sum(comp))
        total_tokens += len(comp)
        if (i + 1) % 10 == 0:
            print(f"[nll] {i + 1} examples scored")
    if total_tokens == 0:
        raise RuntimeError("no completion tokens scored")
    return total_neg_logprob / total_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="OpenAI base URL, e.g. http://host/openai/v1")
    ap.add_argument("--eval-jsonl", default="runs/gsm8k_small/eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None, help="human label stored in the payload")
    ap.add_argument("--model", default=None, help="served model name (discovered if omitted)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--skip-nll", action="store_true")
    args = ap.parse_args()

    examples = load_examples(args.eval_jsonl)
    if args.max_items:
        examples = examples[: args.max_items]
    model = args.model or discover_served_model_name(args.endpoint, args.api_key)
    print(f"endpoint={args.endpoint} model={model} items={len(examples)}")

    items, correct = [], 0
    for i, ex in enumerate(examples):
        gold = extract_answer(ex["completion"])
        gen = generate(args.endpoint, model, ex["prompt"], args.api_key)
        pred = extract_answer(gen)
        ok = pred is not None and gold is not None and pred == gold
        correct += ok
        items.append({
            "question": ex["prompt"],
            "gold_completion": ex["completion"],
            "gold_answer": gold,
            "generation": gen,
            "pred_answer": pred,
            "correct": ok,
        })
        print(f"[{i + 1}/{len(examples)}] gold={gold} pred={pred} {'OK' if ok else 'X'}")

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
        "n_correct": correct,
        "accuracy": correct / len(items) if items else 0.0,
        "nll": nll,
        "perplexity": math.exp(nll) if nll is not None else None,
        "captured_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "items": items,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"accuracy={payload['accuracy']:.1%} ({correct}/{len(items)}) -> {out}")


if __name__ == "__main__":
    main()
