#!/bin/bash
#SBATCH -J render_dataset
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/%x_%A_%a.out

# Re-render LIBERO demo files with Blender, one file per array task. Task i takes the i-th file of
# <libero_datasets>/libero_*/*.hdf5: libero_10 is 0-9, libero_90 10-99, then goal, object and spatial.
# Usage: sbatch --array=0-129%4 slurm/render_dataset.sbatch configs/render/eevee.yaml
set -e
source .venv/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
DATA=$(python -c "from libero_blender.paths import LIBERO_DATASETS; print(LIBERO_DATASETS)")
FILE=$(cd $DATA && ls libero_*/*.hdf5 | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
python scripts/render_dataset.py --render $1 --suite $(dirname $FILE) --file $(basename $FILE) \
    --local-dir ${TMPDIR:-/tmp}/render_dataset_$SLURM_JOB_ID
