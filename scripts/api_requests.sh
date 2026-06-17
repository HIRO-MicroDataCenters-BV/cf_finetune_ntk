#!/usr/bin/env bash
# P1.7 §E benchmark — CogAPI request driver.
#
# Walks the full end-to-end Phase-1 surface:
#   1. POST /datasets/file       — multipart upload train.jsonl as a
#                                   DatasetTypeEnum.JSONL (=5) row.
#   2. POST /fine-tune/recommend — returns paper-aligned ntkmirror
#                                   defaults (gates / max_log_gate /
#                                   train_steps / lr) for the base.
#   3. POST /models/fine-tune    — submits the kfp run that lands the
#                                   trained LoRA adapter as
#                                   model_info(type='lora',
#                                              base_model_id=<BASE>).
#
# The base model row (Qwen2.5-0.5B-Instruct) must already exist as
# model_info(type='llm', hf_model_id='Qwen/Qwen2.5-0.5B-Instruct').
# Register it via the existing /models/log route or the UI before
# running this script.
#
# Required env:
#   COGAPI_URL         e.g. https://dashboard.cog.hiro-develop.nl/cogapi
#   KUBEFLOW_USERID    your kubeflow user id (header)
#   AUTHSESSION_COOKIE the dashboard session cookie value
#   BASE_MODEL_ID      UUID of the pre-registered Qwen2.5-0.5B-Instruct row
#
# Optional env:
#   TRAIN_JSONL        path to the prepared JSONL (default: runs/gsm8k_small/train.jsonl)
#   OUTPUT_NAME        name for the resulting LoRA row (default: ntk-gsm8k-small-<ts>)
#   DATASET_NAME       name for the dataset row (default: gsm8k-small-<ts>)
#
# Exits non-zero on any HTTP error.

set -euo pipefail

: "${COGAPI_URL:?set COGAPI_URL, e.g. https://dashboard.cog.hiro-develop.nl/cogapi}"
: "${KUBEFLOW_USERID:?set KUBEFLOW_USERID — your kubeflow user id}"
: "${AUTHSESSION_COOKIE:?set AUTHSESSION_COOKIE — dashboard session cookie value}"
: "${BASE_MODEL_ID:?set BASE_MODEL_ID — UUID of the registered Qwen2.5-0.5B-Instruct row}"

TS=$(date -u +%Y%m%d-%H%M%S)
TRAIN_JSONL=${TRAIN_JSONL:-runs/gsm8k_small/train.jsonl}
OUTPUT_NAME=${OUTPUT_NAME:-ntk-gsm8k-small-$TS}
DATASET_NAME=${DATASET_NAME:-gsm8k-small-$TS}
BASE_HF_ID=${BASE_HF_ID:-Qwen/Qwen2.5-0.5B-Instruct}

if [[ ! -f $TRAIN_JSONL ]]; then
  echo "TRAIN_JSONL not found: $TRAIN_JSONL — run prepare_dataset.py first" >&2
  exit 1
fi

CURL=(curl --fail-with-body -sS \
  -H "kubeflow-userid: $KUBEFLOW_USERID" \
  -H "Cookie: authservice_session=$AUTHSESSION_COOKIE")

# ---------------------------------------------------------------------------
# 1. Register + upload the JSONL dataset.
#    DatasetTypeEnum.JSONL = 5.
# ---------------------------------------------------------------------------
echo ">>> [1/3] uploading $TRAIN_JSONL as dataset '$DATASET_NAME'"
DATASET_RESP=$(
  "${CURL[@]}" \
    -X POST "$COGAPI_URL/datasets/file" \
    -F "dataset_type=5" \
    -F "name=$DATASET_NAME" \
    -F "description=GSM8K small — P1.7 §E benchmark train split" \
    -F "files=@$TRAIN_JSONL"
)
echo "$DATASET_RESP" | python3 -m json.tool
DATASET_ID=$(echo "$DATASET_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['id'])")
echo "dataset_id=$DATASET_ID"

# ---------------------------------------------------------------------------
# 2. Recommender — paper-aligned defaults.
#    Phase 1 returns static ntkmirror defaults regardless of hf_model_id.
# ---------------------------------------------------------------------------
echo ">>> [2/3] requesting recommender knobs for $BASE_HF_ID"
RECO_RESP=$(
  "${CURL[@]}" \
    -X POST "$COGAPI_URL/fine-tune/recommend" \
    -H 'content-type: application/json' \
    -d "{\"hf_model_id\":\"$BASE_HF_ID\"}"
)
echo "$RECO_RESP" | python3 -m json.tool
RECO_DATA=$(echo "$RECO_RESP" | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin)['data']))")

# ---------------------------------------------------------------------------
# 3. Fine-tune submission.
#    Spread the recommender's knob dict directly into hyperparams (the
#    field names match by design — see NtkHyperparams docstring).
# ---------------------------------------------------------------------------
echo ">>> [3/3] submitting fine-tune run output_name=$OUTPUT_NAME"
FT_BODY=$(python3 - <<PY
import json
reco = json.loads('''$RECO_DATA''')
hp = {
  "gates": reco["gates"],
  "max_log_gate": reco["max_log_gate"],
  "train_steps": reco["train_steps"],
  "lr": reco["lr"],
}
print(json.dumps({
  "base_model_id": "$BASE_MODEL_ID",
  "dataset_id": "$DATASET_ID",
  "output_name": "$OUTPUT_NAME",
  "method": "ntk",
  "export": "lora",
  "hyperparams": hp,
}))
PY
)
FT_RESP=$(
  "${CURL[@]}" \
    -X POST "$COGAPI_URL/models/fine-tune" \
    -H 'content-type: application/json' \
    -d "$FT_BODY"
)
echo "$FT_RESP" | python3 -m json.tool

MODEL_ID=$(echo "$FT_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['model_id'])")
RUN_ID=$(echo "$FT_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'].get('run_id') or '')")
echo
echo "=================================================================="
echo " Fine-tune submitted."
echo " reserved model_id : $MODEL_ID"
echo " kfp run_id        : $RUN_ID"
echo
echo " Watch the run in the Kubeflow Pipelines UI. On success the"
echo " kfp component creates a model_info(type='lora',"
echo " base_model_id='$BASE_MODEL_ID', id='$MODEL_ID') row via"
echo " cogflow's MLflow handshake — at which point evaluate_nll.py"
echo " can be pointed at it for the §E three-way comparison."
echo "=================================================================="
