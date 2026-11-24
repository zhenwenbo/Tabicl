#!/usr/bin/env bash
# Run the full three-stage TabICLv2 classifier recipe on one node with eight GPUs.
# Usage: bash scripts/train_v2_clf_8gpu.sh
# CHECKPOINT_ROOT overrides the default Model/tabiclv2-clf-8gpu directory.
# N_JOBS controls the number of synthetic-data workers per GPU (default: 16).
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(dirname -- "$script_dir")

export NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0,1,2,3,4,5,6,7}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$repo_dir/Model/tabiclv2-clf-8gpu}
export STAGE3_RECOMPUTE=${STAGE3_RECOMPUTE:-True}
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

if ! command -v torchrun >/dev/null; then
    printf 'torchrun was not found. Activate the TabICL pretraining environment.\n' >&2
    exit 1
fi

mkdir -p -- "$CHECKPOINT_ROOT"
exec 9>"$CHECKPOINT_ROOT/.train.lock"
if ! flock -n 9; then
    printf 'A training job already holds the lock for %s\n' "$CHECKPOINT_ROOT" >&2
    exit 1
fi

final_steps=(500000 40000 10000)
for stage in 1 2 3; do
    checkpoint="$CHECKPOINT_ROOT/stage$stage/step-${final_steps[stage-1]}.ckpt"
    done_file="$CHECKPOINT_ROOT/stage$stage/.complete"
    if [[ -s "$checkpoint" && -f "$done_file" ]]; then
        printf 'Stage %s already complete (%s)\n' "$stage" "$checkpoint"
        continue
    fi

    printf '[%s] Starting stage %s on %s\n' "$(date -u +%FT%TZ)" "$stage" "$CUDA_VISIBLE_DEVICES"
    if ! METRICS_CSV="$CHECKPOINT_ROOT/stage$stage.csv" bash "$script_dir/train_v2_clf_stage$stage.sh" 2>&1 \
        | tr '\r' '\n' | tee -a "$CHECKPOINT_ROOT/stage$stage.log"; then
        printf 'Stage %s failed. See %s/stage%s.log\n' "$stage" "$CHECKPOINT_ROOT" "$stage" >&2
        exit 1
    fi
    if [[ ! -s "$checkpoint" ]]; then
        printf 'Stage %s did not produce %s\n' "$stage" "$checkpoint" >&2
        exit 1
    fi
    touch -- "$done_file"
    printf '[%s] Finished stage %s (%s)\n' "$(date -u +%FT%TZ)" "$stage" "$checkpoint"
done
