"""NTK vs PEFT-LoRA training benchmark on identical data, steps and GPU.
Both arms: same train.jsonl, batch 8, 240 steps, max_length 1024, completion-only loss,
model loaded the same way. Reports wall-clock, trainable params, artifact bytes, peak GPU memory,
held-out NLL before/after. Runs inside the serving image (torch + transformers + ntkmirror)."""
import json, math, os, sys, time, tempfile, shutil, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ntkmirror import ForwardFineTuner, load_jsonl_examples
from ntkmirror.data import make_batch

BASE = "Qwen/Qwen2.5-0.5B-Instruct"; STEPS = 240; BS = 8; MAXLEN = 1024
train = load_jsonl_examples(sys.argv[1]); evals = load_jsonl_examples(sys.argv[2])
dev = "cuda"
tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.pad_token or tok.eos_token

def batches(xs, n):
    for i in range(0, len(xs), n): yield xs[i:i+n]

def ce_loss(logits, labels):
    logits = logits[:, :-1, :].float(); labels = labels[:, 1:]
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)

@torch.no_grad()
def eval_nll(model):
    model.eval(); tot = 0.0; n = 0
    for chunk in batches(evals, BS):
        b = make_batch(tok, chunk, device=dev, max_length=MAXLEN)
        out = model(input_ids=b["input_ids"], attention_mask=b.get("attention_mask"), use_cache=False)
        lab = b["labels"][:, 1:]; k = int((lab != -100).sum())
        tot += float(ce_loss(out.logits, b["labels"])) * k; n += k
    return tot / max(1, n)

def dir_bytes(p): return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)

def load():
    m = AutoModelForCausalLM.from_pretrained(BASE).to(dev)  # same call as the platform component
    return m

results = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "steps": STEPS, "batch": BS, "train_examples": len(train), "eval_examples": len(evals)}

# ---- Arm A: NTK controller (as the platform pipeline does) ----
torch.cuda.reset_peak_memory_stats(); model = load()
nll0 = eval_nll(model)
tr = ForwardFineTuner(model=model, tokenizer=tok, gates=5000, max_log_gate=0.5)
t0 = time.perf_counter(); stats = tr.fit(train, steps=STEPS, lr=5e-3, batch_size=BS, max_length=MAXLEN, verbose=False); wall = time.perf_counter() - t0
nll1 = tr.evaluate_nll(evals)["nll"]
d = tempfile.mkdtemp(); tr.save(os.path.join(d, "controller.pt"))
results["ntk"] = {"wall_seconds_total": wall, "train_seconds": stats.get("train_seconds"), "select_seconds": stats.get("select_seconds"),
    "trainable_params": 5000, "artifact_bytes": dir_bytes(d), "peak_gpu_mem_gb": torch.cuda.max_memory_allocated()/1e9,
    "eval_nll_before": nll0, "eval_nll_after": nll1, "loss_first": stats.get("loss_first"), "loss_last": stats.get("loss_last")}
print(json.dumps({"ntk": results["ntk"]}), flush=True)
del tr, model; gc.collect(); torch.cuda.empty_cache()

# ---- Arm B: PEFT LoRA r=8 on all linear layers ----
from peft import LoraConfig, get_peft_model
torch.cuda.reset_peak_memory_stats(); model = load()
cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
pm = get_peft_model(model, cfg)
n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad)
opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=2e-4)
tb = list(batches(train, BS)); losses = []
pm.train(); t0 = time.perf_counter()
for step in range(STEPS):
    b = make_batch(tok, tb[step % len(tb)], device=dev, max_length=MAXLEN)
    opt.zero_grad(set_to_none=True)
    out = pm(input_ids=b["input_ids"], attention_mask=b.get("attention_mask"), use_cache=False)
    loss = ce_loss(out.logits, b["labels"]); loss.backward(); opt.step(); losses.append(float(loss))
torch.cuda.synchronize(); wall = time.perf_counter() - t0
nll1 = eval_nll(pm)
d = tempfile.mkdtemp(); pm.save_pretrained(d)
results["lora"] = {"wall_seconds_total": wall, "train_seconds": wall, "trainable_params": n_train, "artifact_bytes": dir_bytes(d),
    "peak_gpu_mem_gb": torch.cuda.max_memory_allocated()/1e9, "eval_nll_before": nll0, "eval_nll_after": nll1,
    "loss_first": losses[0], "loss_last": losses[-1], "r": 8, "lr": 2e-4}
print(json.dumps({"lora": results["lora"]}), flush=True)
print("RESULT " + json.dumps(results), flush=True)
