# LIBERO-Blender: datasets, Blender-image evaluation and policy results

LIBERO with its MuJoCo physics, init states and success checks unchanged, and the two camera images
(agentview, wrist) rendered by Blender instead. Two renderers: EEVEE (rasterised, deterministic) and
Cycles (path traced, 256 samples, OptiX denoiser). Render settings: `configs/render/eevee.yaml`,
`configs/render/cycles.yaml` (both version 1).

## Datasets

The full original LIBERO release re-rendered, in the original HDF5 layout: same files, demos, states,
actions and low-dim observations; only `obs/agentview_rgb` and `obs/eye_in_hand_rgb` are replaced
(same 128 x 128 shape, dtype, chunking and compression). Each file's render config is in
`data.attrs["blender_render_config"]`.

| | EEVEE | Cycles |
|---|---|---|
| Root | `datasets/eevee` | `datasets/cycles` |
| Files | 130 / 130 | 130 / 130 |
| Demos / frames | 6,500 / 1,007,618 | 6,500 / 1,007,618 |
| Size | 101 GB | 101 GB |
| Render time (summed) | 91 GPU-h (L40S, some A30) | 182 GPU-h (A30) |

Suites: libero_10, libero_90, libero_spatial, libero_object, libero_goal, under `<root>/<suite>/`.
`<root>/manifest.json` has one entry per file (source, sha256, render config, git commit, checks) and
`<root>/summary.md` has the per-suite checks. Per file:

- the stored LIBERO images were compared with MuJoCo re-renders of the stored states, to confirm
  that replaying states reproduces the demos and to find the stored orientation (no flip, all files);
- 12 random camera-frames got a geom-id alignment check against MuJoCo segmentation; all files
  passed, worst mean IoU 0.9968.

EEVEE renders are bit-identical across runs and between L40S and A30 GPUs. Cycles varies by ±1
grey level between runs.

## Evaluation protocol

`scripts/evaluate.py` with `configs/eval.yaml`, which follows semantic_diffusion's LIBERO eval: 20 trials
per task with init states 0-19, seed 100, 10 settle steps, a new action chunk every 10 actions. The MoDE
policy server (`scripts/policies/mode_server.py`) rotates the 128 px images by 180 degrees, uses 10
denoising steps and binarizes the gripper action. On an A30 an EEVEE eval takes about 0.1 GPU-h per task,
a Cycles eval about 0.2.

## Policy results (LIBERO-10)

MoDE (semantic_diffusion, `libero_mode_edm_train`, 60k steps, seed 42, trained on LIBERO-90 +
LIBERO-10), evaluated on LIBERO-10, 200 episodes per row. ± is the binomial standard error.

| Policy trained on | Eval images | Success |
|---|---|---|
| MuJoCo images (`l100_mode_edm_60k_emafix_8038394`) | MuJoCo | 0.960 ± 0.014 |
| MuJoCo images | EEVEE | 0.000 |
| MuJoCo images | Cycles | 0.000 |
| MuJoCo images + colour jitter (`l100_mode_60k_colorjit_698450`) | EEVEE | 0.005 ± 0.005 |
| **EEVEE dataset, from scratch** (`l100_mode_scratch_eevee_698451`) | **EEVEE** | **0.915 ± 0.020** |
| **Cycles dataset, from scratch** (`l100_mode_scratch_cycles_705430`) | **Cycles** | **0.810 ± 0.028** |

Per task (20 trials each):

| task | MuJoCo-trained on MuJoCo | EEVEE-trained on EEVEE | Cycles-trained on Cycles |
|---|---|---|---|
| 0 | 0.90 | 0.95 | 1.00 |
| 1 | 1.00 | 1.00 | 1.00 |
| 2 (stove + moka pot) | 1.00 | 0.95 | 0.40 |
| 3 | 1.00 | 0.80 | 0.85 |
| 4 | 0.95 | 0.90 | 0.85 |
| 5 | 1.00 | 1.00 | 1.00 |
| 6 | 1.00 | 0.90 | 0.95 |
| 7 | 1.00 | 0.95 | 0.95 |
| 8 (two moka pots on stove) | 0.75 | 0.70 | 0.20 |
| 9 | 1.00 | 1.00 | 0.90 |

Findings:

- A policy trained on MuJoCo images fails completely on Blender images, with either renderer. Colour jitter
  in training, which makes MoDE robust to colour and brightness shifts within MuJoCo, does not
  help. The gap is shading, shadows and materials, not only colour.
- Training on the re-rendered data recovers most of the performance: EEVEE is 4.5 points below the MuJoCo
  baseline. Cycles is 15 points below, almost all from the two moka-pot/stove tasks. This is
  one training seed per renderer, so the EEVEE vs Cycles gap is indicative only.

## Random lighting

`scripts/evaluate.py --lighting configs/lighting/random.yaml` gives every trial its own lighting:
- **colour:** the key light, the fill light or both get one random hue, at saturation 0-0.6;
- **key light:** placed 30-90 degrees above the table in any direction and aimed at the scene;
- **fill light:** the HDRI is turned 0-360 degrees, which moves the main shadows;
- **brightness:** x0.6-2.0.

The lighting is drawn from (seed, task, trial), so every policy sees the same 200 lightings, and each
trial's lighting is saved in its result. Example renders: `docs/images/`.

| Policy | Images | Fixed lighting | Random lighting |
|---|---|---|---|
| EEVEE-trained | EEVEE | 0.915 ± 0.020 | 0.010 ± 0.007 (2/200) |
| Cycles-trained | Cycles | 0.810 ± 0.028 | 0.010 ± 0.007 (2/200) |

Both policies collapse. The 4 successes all had near-white light, and 3 of them brightness 0.99-1.13.
As a control, the random-lighting code path with its ranges collapsed to the fixed lighting gives
57/60 on LIBERO-10 tasks 0-2, against 58/60 with fixed lighting, so the collapse is the policies' and
not the eval's. This matches a separate robustness study in semantic_diffusion, where plain MoDE fails
at brightness x0.6 even on MuJoCo images.

### Training augmentation against random lighting

Three more runs, each from scratch with the same 60k recipe and seed 42, with two augmentation
options:
- `pi0_augment=true` is semantic_diffusion's existing full augmentation: agentview crop and rotation,
  plus brightness x0.7-1.3, contrast and saturation jitter.
- `light_jitter=true` is a new opt-in flag, off by default, that modifies `pi0_augment`. It widens
  brightness to x0.6-2.0 and adds a light-colour tint with the same hue and saturation model as the eval
  lights (`_jitter_light` in semantic_diffusion's `pi0_preprocessor.py`). The existing augmentation
  options are unchanged.

| Policy | Images | Fixed lighting | Random lighting |
|---|---|---|---|
| EEVEE, plain | EEVEE | 0.915 ± 0.020 | 0.010 ± 0.007 |
| EEVEE, `pi0_augment` (`l100_mode_scratch_eevee_pi0aug_8285896`) | EEVEE | 0.915 ± 0.020 | 0.185 ± 0.027 |
| EEVEE, `pi0_augment` + `light_jitter` (`..._eevee_pi0aug_light_8285897`) | EEVEE | 0.925 ± 0.019 | 0.400 ± 0.035 |
| Cycles, plain | Cycles | 0.810 ± 0.028 | 0.010 ± 0.007 |
| Cycles, `pi0_augment` + `light_jitter` (`..._cycles_pi0aug_light_8285898`) | Cycles | 0.900 ± 0.021 | 0.650 ± 0.034 |

- The augmentations do not cost fixed-lighting performance. On Cycles, fixed lighting goes up from
  0.810 to 0.900, mostly on the two stove tasks: task 2 goes from 0.40 to 1.00 and task 8 from 0.20 to 0.85.
- The light jitter is what helps against random lighting. Both EEVEE runs saw the same 200 lightings:
  52 trials succeeded only with `light_jitter`, against 9 only without it (McNemar p < 1e-6).
- Random lighting still costs a lot: 0.40 on EEVEE and 0.65 on Cycles. Image augmentation cannot
  produce moving shadows or a change of light direction. The Cycles vs EEVEE difference is from one
  seed each.

Per-task results: `results/eval/<run>/libero_10/task<id>.json`.

Full table: `results/eval/summary.md`. Training used semantic_diffusion's `libero_mode_edm_train` recipe, with
`LIBERO_DATA_SRC` pointed at each dataset root.
