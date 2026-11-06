#!/bin/bash
#SBATCH -J evaluate
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00

# Closed-loop eval of a semantic_diffusion MoDE checkpoint, with the example policy server in
# scripts/policies/mode_server.py. One LIBERO task per array task; without --array, one job runs the
# tasks listed in configs/eval.yaml.
# Usage: SEMANTIC_DIFFUSION=<semantic_diffusion>/src/diffusion_policy \
#     sbatch --array=0-9 slurm/evaluate.sbatch <run dir> <name> <configs/render/*.yaml | mujoco> [configs/lighting/*.yaml]
set -e
PORT=$((20000 + SLURM_JOB_ID % 20000))
# HF_HOME unset so the policy's own settings choose its model cache.
env -u HF_HOME $SEMANTIC_DIFFUSION/.venv/bin/python scripts/policies/mode_server.py --run-dir $1 --port $PORT \
    > policy-${SLURM_JOB_ID}.log 2>&1 &
source .venv/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python scripts/evaluate.py --render $3 ${4:+--lighting $4} ${SLURM_ARRAY_TASK_ID:+--task-id $SLURM_ARRAY_TASK_ID} \
    --policy-port $PORT --policy "$(basename $1)" --name $2
kill %1
