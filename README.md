# cf_finetune_ntk

Benchmark / validation tooling for **NTK fine-tuning** in the Cognitive
Framework — measuring the negative-log-likelihood (NLL) quality of an
[ntkmirror](https://github.com/leochlon/ntkmirror) signed-log-gate controller
across the export + serving paths.

This is validation tooling, **not** product code — it deliberately lives
outside the platform repos (Cog-Engine / cogflow).

## What it measures

All arms compute the **same** metric — teacher-forced NLL averaged over the
**completion tokens only** (the prompt span is masked), on a held-out GSM8K
eval split — so the numbers are apples-to-apples.

### §E — local three-way (`evaluate_nll.py`)
1. **floor** — base model, no controller.
2. **reference** — base + ntkmirror controller attached via
   `ControllerRuntime.apply` forward hooks (the canonical hook-attached number).
3. **lora-export** — the PEFT adapter produced by `controller_to_lora`, merged
   into the base (the Phase-1 production path).

The §E gap `reference → lora-export` is the *approximation error* of the LoRA
export; a wide `floor → reference` gap shows the controller is doing something.

### §F — served gate (`notebooks/ntk_step5_served_nll.ipynb`)
The **served LLM+NTK** path: a deployed KServe ISVC's native `ntk_controller`
(cf-hfserver `NTKQwen2ForCausalLM` plugin), scored over the OpenAI
`/v1/completions` endpoint (`echo` + `logprobs`). Because the served plugin
applies the *same* signed-log gate as the §E `reference` arm, served NLL should
**reproduce** the reference —

    PASS when  |served − reference| / reference ≤ threshold   (default 1%)

This is a serving-correctness gate (did the deployed pod stage + attach the real
controller and preserve the method?), not an approximation test.

#### Validated result (2026-06-19, Qwen2.5-0.5B + GSM8K)
On controller `1a62543f` / eval set `39030a5d`, under vLLM's **default compiled +
CUDA-graph mode**: floor 0.7706 / reference 0.5886 / **served 0.5895 → PASS (0.15%)**.

This gate first **caught a real serving bug**: ntkmirror applies its gate via
PyTorch `nn.Module` forward hooks (`hook_site=layer_output`), but vLLM's
`Qwen2Model` is `@support_torch_compile`d and CUDA-graph-captured, so the hooks
never fire on real requests — and even in eager mode the stock post-hook scales
only vLLM's split `hidden_states`, leaving `residual` (most of the stream)
unscaled. Served NLL sat at the base floor (0.72) in both modes. The fix
(cf-hfserver) reimplements the gate **natively in the decoder-layer forward**,
scaling both `hidden_states` and `residual` (their sum is the residual stream),
so it's captured by compile + CUDA graphs. **Requires the gate-fixed serving
image** (`hiroregistry/cfhfserver:0.18.9-gpu` or later).

Apples-to-apples by construction: the notebook tokenizes `prompt`+`completion`
with the same base tokenizer and the same split as the reference, then sends the
**exact token ids** as the completions prompt — so there's no server-side
retokenization drift, and the completion span is exactly
`token_logprobs[len(prompt_ids):]`.

## Layout

```
prepare_dataset.py                 GSM8K -> {prompt, completion} JSONL (train/eval split)
evaluate_nll.py                    §E three-way local NLL (floor / reference / lora-export)
notebooks/ntk_step5_served_nll.ipynb   §F served LLM+NTK NLL gate
scripts/api_requests.sh            end-to-end CogAPI driver (dataset upload + recommend + fine-tune)
requirements.txt                   torch / transformers / peft / datasets / requests / ntkmirror
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python prepare_dataset.py --out-dir runs/gsm8k_small     # 64 train / 32 eval
```

## Runbook

1. **§E (local, GPU):** train or pull a `controller.pt`, then
   ```bash
   python evaluate_nll.py --base Qwen/Qwen2.5-0.5B-Instruct \
     --eval-jsonl runs/gsm8k_small/eval.jsonl \
     --controller runs/gsm8k_small/controller.pt \
     --adapter    runs/gsm8k_small/adapter
   ```
   Record the **reference** NLL — it is the denominator for §F.
2. **Get a servable controller:** either fine-tune via the platform
   (`POST /cogapi/models/fine-tune`, `export="ntk_model"`) which trains + registers
   an `ntk_controller` row, **or** register an existing `controller.pt` directly
   (no training needed) by logging it to an MLflow run under `controller/controller.pt`
   and calling `cogflow.models.register_finetuned_catalog_entry(run_id, adapter_type=
   "ntk_controller", base_model_id=..., base_model_hf_id=..., ...)`.
3. **Deploy** the ntk_model ISVC (`POST /cogapi/models-serving` with
   `llm_adapter:{kind:"ntk_model", adapter_model_id:<row>}`) — **without**
   `--quantization`, for an apples-to-apples comparison.
4. **§F (served):** open `notebooks/ntk_step5_served_nll.ipynb`, set the config
   (`EVAL_DATASET_ID` = the registered cogflow eval dataset, served model name, and
   `REFERENCE_NLL` = the step-1 number for the **same** controller, computed on that
   **same** eval set), and run → it pulls the eval JSONL through cogflow
   (`download_dataset`), measures served NLL, and prints PASS/FAIL.

### Eval data comes from a registered cogflow dataset
The §F notebook is self-contained: it pulls its eval split through cogflow's dataset
channel (`cogflow.datasets.download_dataset`), the same way the fine-tune pulls its
train split — not from a local file. The two splits are GSM8K's official, disjoint
`train` / `test` splits (see `prepare_dataset.py`), so the eval set is genuinely
held-out from fine-tuning. Registered platform datasets (dev):

| split | dataset id | source |
|-------|-----------|--------|
| train (fine-tune) | `a6693b90-b11a-4af1-a31b-51cd3c1a84e8` | GSM8K `train`, first 64 |
| eval  (§F gate)    | `39030a5d-b36c-4f07-8895-667515fcaa14` | GSM8K `test`, first 32 (held-out) |

To re-register on another cluster: `python prepare_dataset.py` then upload each split
via `POST /datasets/file` (`dataset_type=5`), and point `EVAL_DATASET_ID` at the eval row.

## Notes
- **Reference must match the served controller.** `REFERENCE_NLL` is only valid
  if computed (§E reference arm) on the exact `controller.pt` the ISVC serves.
- **No quantization on the gate ISVC** — served (bf16/fp16) vs reference (fp32)
  should differ only by float noise (the 1% band). A larger gap is a serving-path
  bug to investigate, not a band to widen.
- `scripts/api_requests.sh` currently submits `export="lora"`; flip it to
  `"ntk_model"` for the Phase-2 fine-tune run.

## References
- ntkmirror — https://github.com/leochlon/ntkmirror (LoRA-free forward-pass
  fine-tuning via signed log-mask controllers).
