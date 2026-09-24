# Demo runbook — "NTK fine-tuning works on IaC" (all in the CF UI)

Story: upload a small Pulumi-for-Kubernetes dataset, fine-tune Qwen2.5-0.5B-Instruct with the
platform's NTK method, read the perplexity before/after and the training cost off the model page,
serve the exact NTK controller, and compare its answers with the base model's in the Playground.
No correctness or deploy claims are made about the generated Pulumi programs.

Serving path: **exact NTK** (`export: ntk_model`, served with `llm_adapter: {kind: ntk_model}`).
The LoRA export was tried first and rejected for the demo: the exported adapter loses the effect
(answers stay YAML even on a training prompt) because the exporter drops the residual-stream term.
One L40 hosts one inference service, so the before/after is done in two halves with the
Playground's **Pin** feature (see step 6).

## Fixed facts

| item | value |
|---|---|
| UI | `https://localhost:1443/uidev/` (local `pnpm dev --port 3000` in cog-framework-ui, `.env.local` has `NUXT_PUBLIC_API_BASE=/apidev`; haproxy `~/bypassdex` injects the session cookie) |
| API smoke test | `curl -sk https://localhost:1443/apidev/openapi.json` |
| Base LLM row | `d3cde38f-6374-44f1-a5db-2cc903250a7a` — Qwen2.5-0.5B-Instruct (`Qwen/Qwen2.5-0.5B-Instruct`) |
| Train dataset | `pulumi-k8s-train` (`e64a3d75-eea2-41b1-9101-bb9e9b1eb6c0`, 160 rows, type 5) |
| Eval dataset | `pulumi-k8s-eval` (`20cdecdc-64b5-4eb1-9356-9201ed992c0f`, 32 held-out rows, type 5) |
| Dataset files | `runs/pulumi/{train,eval}.jsonl` (generator: `scripts/make_pulumi_dataset.py`, QA: `scripts/check_pulumi_dataset.py`) |
| Knobs | gates 5000 · **max_log_gate 0.5** · steps 240 · lr 0.005 (the recommender fills 0.05; raise Max log gate to 0.5 on camera — 0.05 moves perplexity but not the generations) |
| Export | **NTK controller (exact)** — the Fine-tune dialog's Export select |
| Rehearsal rows | `iac-house-style-ntk-v3` = `53a4b505-fecb-4946-bf14-49199502bfaa` (ntk_controller, clamp 0.5); LoRA-export arms `iac-house-style-v1/v2/v3` (0.05/0.2/0.5) kept for the metrics comparison |
| GPU | one L40 on `cog-gpu`; one vLLM pod at a time |

## Before recording (off camera)

1. Free the GPU: `kubectl scale deploy ntk-pirate-predictor -n admin --replicas=0` (and any other
   `*-predictor` deployment at 1 replica: `kubectl get deploy -n admin | grep predictor`).
2. Start the UI: `cd cog-framework-ui && pnpm dev --port 3000`; open `https://localhost:1443/uidev/`.
3. Keep this watcher running in a terminal — the fine-tune GPU pod needs the storage-type toleration:
   ```bash
   cd /home/ali/project/coge
   while true; do for p in $(kubectl get pods -n admin --no-headers | grep ntk-fine-tune | awk '$3=="Pending"{print $1}'); do ./patch-pod-toleration.sh admin/$p; done; sleep 20; done
   ```
4. Do NOT merge/deploy anything to cog-api-dev while a fine-tune runs: the component registers the
   catalog row by POSTing to CogAPI at the end, and cogflow only warns if that call fails.
5. Optional: upload the two datasets fresh on camera (step 1 below) or reuse the rows above.

## On camera

1. **Datasets → New dataset → File → "JSONL (fine-tune)"** → upload `runs/pulumi/train.jsonl` as
   `pulumi-k8s-train`; repeat for `eval.jsonl` as `pulumi-k8s-eval`.
2. **Fine-tune → New fine-tune**: base *Qwen2.5-0.5B-Instruct*, training dataset *pulumi-k8s-train*,
   evaluation dataset *pulumi-k8s-eval*, name `iac-house-style`, Export *NTK controller (exact)*,
   knobs auto-filled then **Max log gate → 0.5** → Launch. Lands on *Pipelines › Runs* (auto-refreshes).
3. Open the run → DAG → the `ntk-fine-tune` node → Logs: loss falling over 240 steps
   (rehearsal: 0.51 → 0.30), "LoRA exported", "ntk_fine_tune complete". ~4–5 min on the L40
   (~2 min of it is pip + base download). Cut or time-lapse.
4. **Models → iac-house-style** (badge `ntk_controller`) → Overview → metrics:
   `eval_perplexity_before` 1.64 → `eval_perplexity_after` ≈ 1.04, `eval_nll_improvement_pct` ≈ 91,
   `train_seconds` ≈ 52, `trainable_params` 5000 vs `base_params` 494,032,768,
   `artifact_bytes` = `controller_bytes` ≈ 103 KB (rehearsal numbers at clamp 0.5).
5. **Before half — base model.** *Model serving → Serve → LLM from Hugging Face*:
   `Qwen/Qwen2.5-0.5B-Instruct`, service name `base-qwen` → wait *ready* (~2 min).
   **Playground** → service `base-qwen` → ask the three requests below → **Pin** each answer
   (they stay on screen; the base answers are Kubernetes YAML / prose).
6. **Flip (cut or time-lapse, ~4 min).** *Model serving* → delete `base-qwen`. *Serve → LLM →
   Model source: From catalog* → base *Qwen2.5-0.5B-Instruct*, adapter *iac-house-style (NTK, exact)*,
   service name `iac-ntk` → wait *ready*.
7. **After half.** *Playground* → Refresh services → `iac-ntk` → ask the same three requests: Pulumi
   Python in the house style (`import pulumi_kubernetes as k8s`, pinned image, `run_as_non_root`,
   `allow_privilege_escalation=False`, cpu/memory limits, Service) next to the pinned YAML answers.

Pre-tested requests (deterministic at temperature 0; all unseen in training, all gave clean
house-style Python from the exact model in rehearsal):
1. `Deploy nginx:1.27 on port 80 with 2 replicas in namespace web, exposed as a ClusterIP service.`
2. `Set up a Deployment for memcached:1.6.29 in the staging namespace listening on port 11211 behind a ClusterIP Service, scaled to 2 replicas.`
3. `Set up a Deployment for ghcr.io/acme/worker:v0.9.3 into the web namespace exposed as a LoadBalancer service on port 9000, with 1 replica. Limit it to 1 CPU and 1Gi memory.`
Avoid on camera: CronJob/Job requests (the 0.5B model invents `k8s.batch.v1.BackendArgs`-style
classes) and "Write Pulumi Python that …" phrasing for nginx (syntax slips). Screening on 12 unseen
requests: 11 parse as Python, 8 carry all four house-style markers — say "most", never "all".

## After recording

- Delete `iac-ntk` (and `base-qwen` if still there); restore whichever model should hold the GPU
  (`kubectl scale deploy ntk-pirate-predictor -n admin --replicas=1` or `qwen38-predictor`).
- The rehearsal rows (`iac-house-style-v1`, datasets above) can stay in the catalog.

## Honesty notes for the presenter

- Perplexity is "how well the model predicts our house-style programs on 32 unseen requests";
  it is not a claim that the programs deploy. IaC validation is the IaC team's product.
- The served model is the exact controller, the same object the pipeline measured; the
  perplexity on the model page therefore describes what is being served.
- The controller is ~100 KB and 5,000 numbers; training took ~52 s on one L40 (the pipeline's
  wall-clock is longer because of image pull, pip and the base download).
- Decoding is deterministic (temperature 0): a pre-tested request always reproduces; an untested
  one may not.
