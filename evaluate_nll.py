#!/usr/bin/env python3
"""
P1.7 §E benchmark — three-way NLL eval harness.

Reproduces the plan §E comparison on the held-out GSM8K eval split:

  (1) REFERENCE: original ntkmirror with hooks attached
      - Loads the trained ``SignedLogMaskState`` directly.
      - Attaches forward hooks to ``model.model.layers[i]``.
      - Computes per-token NLL on the completion tokens only.

  (2) PRODUCTION: LoRA export served via vLLM ``--enable-lora``
      - Loads the PEFT adapter directory exported by
        ``cogflow.transformers.ntkmirror_export.controller_to_lora``.
      - The kfp component logged it via cogflow MLflow handshake
        as ``model_info(type='lora', base_model_id=<base>)``.
      - For NLL we don't need vLLM specifically — just merge the
        adapter into the HF model and compute teacher-forced NLL
        the same way as the reference path. This isolates "did the
        LoRA conversion preserve the controller?" from any vLLM
        serving-side artifacts.

  (3) FLOOR: base model only
      - Same base, no controller, no adapter.

Pass criterion (plan §E): the LoRA-export NLL is within 2-3% of the
hook-attached reference. The floor is shown for context — a wide
floor → reference gap means the controller is actually doing
something; a small reference → LoRA-export gap means the conversion
preserved the effect.

Usage:
  pip install transformers peft datasets torch ntkmirror
  python evaluate_nll.py \\
    --base Qwen/Qwen2.5-0.5B-Instruct \\
    --eval-jsonl runs/gsm8k_small/eval.jsonl \\
    --controller runs/gsm8k_small/controller.pt \\
    --adapter    runs/gsm8k_small/adapter

The ``--controller`` and ``--adapter`` paths point at artifacts the
kfp component produced. Pull them down from MinIO before running
this script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


def _load_examples(jsonl_path: Path) -> list[dict[str, Any]]:
    """Read the same prompt/completion JSONL the trainer consumes."""
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _completion_nll(
    model,
    tokenizer,
    examples: Iterable[dict[str, Any]],
    device: str,
) -> float:
    """Teacher-forced NLL averaged over completion tokens only.

    Identical methodology across the three paths so the comparison is
    apples-to-apples: each example's prompt is tokenized, the
    completion is appended, and only the completion-span tokens
    contribute to the loss. Prompt tokens are masked out so the
    reported number reflects what the model would have lost on the
    *answer* it was trained to produce, not on the prompt it was
    given.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    total_logloss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for ex in examples:
            prompt_ids = tokenizer(ex["prompt"], return_tensors="pt").input_ids
            completion_ids = tokenizer(
                ex["completion"], return_tensors="pt", add_special_tokens=False
            ).input_ids
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(device)

            logits = model(input_ids=input_ids).logits  # [1, T, V]
            # Shift for next-token prediction.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Mask out the prompt-span tokens.
            prompt_len = prompt_ids.shape[1]
            mask = torch.zeros_like(shift_labels, dtype=torch.bool)
            # Targets at positions >= prompt_len-1 correspond to the
            # completion tokens (we lost one position to the shift).
            mask[:, prompt_len - 1 :] = True

            losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view_as(shift_labels)
            selected = losses[mask]
            total_logloss += selected.sum().item()
            total_tokens += selected.numel()

    return total_logloss / max(total_tokens, 1)


def _eval_floor(base: str, eval_jsonl: Path, device: str) -> float:
    """Path (3) — base model, no controller, no adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[floor] loading base {base!r}")
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base).to(device)
    nll = _completion_nll(model, tokenizer, _load_examples(eval_jsonl), device)
    print(f"[floor] NLL = {nll:.4f}  ({math.exp(nll):.4f} ppl)")
    return nll


def _eval_reference(
    base: str, controller_path: Path, eval_jsonl: Path, device: str
) -> float:
    """Path (1) — base + ntkmirror controller applied via ControllerRuntime.

    Uses ``ControllerRuntime.apply()`` — the canonical public API for
    inference-side hook installation in the pinned upstream commit.
    The context manager installs forward hooks on the decoder layers
    selected by the controller for the duration of the ``with`` block
    and removes them in a ``finally``, so the same model object can be
    reused cleanly across calls without leaking state.
    """
    from ntkmirror import ControllerRuntime, SignedLogMaskState
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[reference] loading base {base!r} + controller {controller_path}")
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base).to(device)
    state = SignedLogMaskState.load(str(controller_path))
    runtime = ControllerRuntime(model=model, tokenizer=tokenizer)
    with runtime.apply(state):
        nll = _completion_nll(model, tokenizer, _load_examples(eval_jsonl), device)
    print(f"[reference] NLL = {nll:.4f}  ({math.exp(nll):.4f} ppl)")
    return nll


def _eval_lora_export(
    base: str, adapter_dir: Path, eval_jsonl: Path, device: str
) -> float:
    """Path (2) — production path — PEFT LoRA merged into the base."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[lora-export] loading base {base!r} + adapter {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base)
    base_model = AutoModelForCausalLM.from_pretrained(base).to(device)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir)).to(device)
    nll = _completion_nll(model, tokenizer, _load_examples(eval_jsonl), device)
    print(f"[lora-export] NLL = {nll:.4f}  ({math.exp(nll):.4f} ppl)")
    return nll


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument(
        "--eval-jsonl", type=Path, default=Path("runs/gsm8k_small/eval.jsonl")
    )
    ap.add_argument("--controller", type=Path, help="path to controller .pt")
    ap.add_argument("--adapter", type=Path, help="path to exported PEFT adapter dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="Pass band: |lora_nll - reference_nll| / reference_nll <= threshold",
    )
    args = ap.parse_args()

    if not args.eval_jsonl.exists():
        print(f"eval JSONL not found: {args.eval_jsonl}", file=sys.stderr)
        return 2

    floor_nll = _eval_floor(args.base, args.eval_jsonl, args.device)

    reference_nll: float | None = None
    if args.controller:
        reference_nll = _eval_reference(
            args.base, args.controller, args.eval_jsonl, args.device
        )

    lora_nll: float | None = None
    if args.adapter:
        lora_nll = _eval_lora_export(
            args.base, args.adapter, args.eval_jsonl, args.device
        )

    print()
    print("================== §E three-way NLL summary ==================")
    print(f"floor (base only)              : {floor_nll:.4f}")
    if reference_nll is not None:
        print(
            f"reference (hooks attached)     : {reference_nll:.4f}  "
            f"(Δ vs floor = {reference_nll - floor_nll:+.4f})"
        )
    if lora_nll is not None and reference_nll is not None:
        rel = abs(lora_nll - reference_nll) / max(reference_nll, 1e-9)
        verdict = "PASS" if rel <= args.threshold else "FAIL"
        print(
            f"production (LoRA-export merged): {lora_nll:.4f}  "
            f"(Δ vs reference = {lora_nll - reference_nll:+.4f}, "
            f"relative = {rel * 100:.2f}%, threshold {args.threshold * 100:.0f}%, "
            f"{verdict})"
        )
        return 0 if verdict == "PASS" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
