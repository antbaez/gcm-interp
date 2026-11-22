#!/bin/bash
#SBATCH --job-name=pyreft
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

CONFIG="${1:-${CONFIG:-pyreft.yaml}}"
if [ ! -f "$CONFIG" ]; then
  echo "Config file not found: $CONFIG" >&2
  exit 1
fi

while IFS= read -r line || [ -n "$line" ]; do
  line="${line%%#*}"
  [[ "$line" != *:* ]] && continue
  key="$(echo "${line%%:*}" | xargs)"
  value="$(echo "${line#*:}" | xargs)"
  [ -z "$key" ] && continue
  var="$(echo "$key" | tr '[:lower:]' '[:upper:]')"
  value="${value/#\~/$HOME}"
  if [ -z "${!var:-}" ]; then
    printf -v "$var" '%s' "$value"
  fi
done < "$CONFIG"

for required in MODEL_ID SOURCE BASE PATCH_ALGO DEVICE VENV REPO; do
  if [ -z "${!required:-}" ]; then
    echo "Missing '$(echo "$required" | tr '[:upper:]' '[:lower:]')' in $CONFIG" >&2
    exit 1
  fi
done
BATCH_SIZE="${BATCH_SIZE:-1}"
N="${N:-0}"

if [ -f "$REPO/.env" ]; then
  while IFS='=' read -r key value || [ -n "$key" ]; do
    key="${key#export }"
    key="$(echo "$key" | xargs)"
    [[ -z "$key" || "$key" == \#* ]] && continue
    value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
    eval "value=\"$value\""
    if [ -n "$value" ] && [ -z "${!key:-}" ]; then
      export "$key=$value"
    fi
  done < "$REPO/.env"
fi

export HF_HOME="${HF_HOME:-$HOME/orcd/pool/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/orcd/pool/.cache/uv}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Add it to $REPO/.env as HF_TOKEN=<your token>" >&2
  exit 1
fi

source "$VENV/bin/activate"
cd "$REPO"

case "$PATCH_ALGO" in
  random) PREFIX="random_random" ;;
  probes) PREFIX="probes_targeted" ;;
  *)      PREFIX="numerator_1_targeted" ;;
esac

MODEL_NAME="${MODEL_ID##*/}"
TASK_DIR="from_${SOURCE}_to_${BASE}/${PATCH_ALGO}"
RUNS_EVAL="runs/${MODEL_NAME}/${TASK_DIR}/eval"
STEER_EVAL="normalized-results/${MODEL_NAME}/${TASK_DIR}/eval_test"

missing=0
mkdir -p "$RUNS_EVAL"
for topk in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.5 1; do
  csv="${PREFIX}_${topk}.csv"
  if [ ! -f "$RUNS_EVAL/$csv" ]; then
    if [ -f "$STEER_EVAL/$csv" ]; then
      cp "$STEER_EVAL/$csv" "$RUNS_EVAL/$csv"
    else
      echo "Missing top-k heads file: $RUNS_EVAL/$csv" >&2
      missing=1
    fi
  fi
done
if [ "$missing" -eq 1 ]; then
  echo "Run patching (--patch_model) and steering eval (--eval_model --eval_test --steering) for this model/task/algo first." >&2
  exit 1
fi

echo "host=$(hostname) model=$MODEL_ID task=${SOURCE}->${BASE} algo=$PATCH_ALGO HF_HOME=$HF_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python run.py \
  --model_id "$MODEL_ID" \
  --batch_size "$BATCH_SIZE" \
  --patch_algo "$PATCH_ALGO" \
  --source "$SOURCE" \
  --base "$BASE" \
  ${JUDGE_ID:+--judge_id "$JUDGE_ID"} \
  --device "$DEVICE" \
  --eval_model \
  --eval_test \
  --N "$N" \
  --pyreft
