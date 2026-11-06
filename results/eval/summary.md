# Phase 4 eval summary

semantic_diffusion MoDE policies, replan 10, 10 denoising steps, seed 100. ± is the binomial standard error.

| run | policy | images |
|---|---|---|
| l10_colorjit_eevee | l100_mode_60k_colorjit_698450, ckpt 60000 | eevee |
| l10_cycles | l100_mode_edm_60k_emafix_8038394, ckpt 60000 | cycles |
| l10_cycles_pi0aug_light | l100_mode_scratch_cycles_pi0aug_light_8285898, ckpt 60000 | cycles |
| l10_cycles_pi0aug_light_randlight | l100_mode_scratch_cycles_pi0aug_light_8285898, ckpt 60000 | cycles + random_light v3 |
| l10_eevee | l100_mode_edm_60k_emafix_8038394, ckpt 60000 | eevee |
| l10_eevee_pi0aug | l100_mode_scratch_eevee_pi0aug_8285896, ckpt 60000 | eevee |
| l10_eevee_pi0aug_light | l100_mode_scratch_eevee_pi0aug_light_8285897, ckpt 60000 | eevee |
| l10_eevee_pi0aug_light_randlight | l100_mode_scratch_eevee_pi0aug_light_8285897, ckpt 60000 | eevee + random_light v3 |
| l10_eevee_pi0aug_randlight | l100_mode_scratch_eevee_pi0aug_8285896, ckpt 60000 | eevee + random_light v3 |
| l10_mujoco | semantic_diffusion MoDE l100_mode_edm_60k_emafix_8038394, ckpt 60000 | mujoco |
| l10_scratch_cycles | l100_mode_scratch_cycles_705430, ckpt 60000 | cycles |
| l10_scratch_cycles_randlight | l100_mode_scratch_cycles_705430, ckpt 60000 | cycles + random_light v3 |
| l10_scratch_eevee | l100_mode_scratch_eevee_698451, ckpt 60000 | eevee |
| l10_scratch_eevee_neutrallight | l100_mode_scratch_eevee_698451, ckpt 60000 | eevee + neutral_light v1 |
| l10_scratch_eevee_randlight | l100_mode_scratch_eevee_698451, ckpt 60000 | eevee + random_light v3 |

## libero_10 (reference 0.965)

| task | l10_colorjit_eevee | l10_cycles | l10_cycles_pi0aug_light | l10_cycles_pi0aug_light_randlight | l10_eevee | l10_eevee_pi0aug | l10_eevee_pi0aug_light | l10_eevee_pi0aug_light_randlight | l10_eevee_pi0aug_randlight | l10_mujoco | l10_scratch_cycles | l10_scratch_cycles_randlight | l10_scratch_eevee | l10_scratch_eevee_neutrallight | l10_scratch_eevee_randlight |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.85 | 0.55 | 0.00 | 0.90 | 1.00 | 0.25 | 0.10 | 0.90 | 1.00 | 0.00 | 0.95 | 0.90 | 0.00 |
| 1 | 0.00 | 0.00 | 0.95 | 0.75 | 0.00 | 0.95 | 1.00 | 0.45 | 0.20 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 0.05 |
| 2 | 0.00 | 0.00 | 1.00 | 0.95 | 0.00 | 1.00 | 1.00 | 0.90 | 0.55 | 1.00 | 0.40 | 0.00 | 0.95 | 0.95 | 0.00 |
| 3 | 0.00 | 0.00 | 0.80 | 0.65 | 0.00 | 0.70 | 0.85 | 0.45 | 0.10 | 1.00 | 0.85 | 0.00 | 0.80 | - | 0.00 |
| 4 | 0.00 | 0.00 | 0.90 | 0.60 | 0.00 | 0.95 | 0.80 | 0.20 | 0.05 | 0.95 | 0.85 | 0.05 | 0.90 | - | 0.00 |
| 5 | 0.05 | 0.00 | 0.90 | 0.95 | 0.00 | 1.00 | 0.95 | 0.80 | 0.25 | 1.00 | 1.00 | 0.05 | 1.00 | - | 0.00 |
| 6 | 0.00 | 0.00 | 0.95 | 0.50 | 0.00 | 0.95 | 0.95 | 0.35 | 0.20 | 1.00 | 0.95 | 0.00 | 0.90 | - | 0.00 |
| 7 | 0.00 | 0.00 | 0.95 | 0.45 | 0.00 | 1.00 | 1.00 | 0.05 | 0.10 | 1.00 | 0.95 | 0.00 | 0.95 | - | 0.00 |
| 8 | 0.00 | 0.00 | 0.85 | 0.55 | 0.00 | 0.85 | 0.75 | 0.30 | 0.15 | 0.75 | 0.20 | 0.00 | 0.70 | - | 0.00 |
| 9 | 0.00 | 0.00 | 0.85 | 0.55 | 0.00 | 0.85 | 0.95 | 0.25 | 0.15 | 1.00 | 0.90 | 0.00 | 1.00 | - | 0.05 |
| all | **0.005 ± 0.005** (200 episodes, 1.88 GPU-h) | **0.000 ± 0.000** (200 episodes, 2.59 GPU-h) | **0.900 ± 0.021** (200 episodes, 1.55 GPU-h) | **0.650 ± 0.034** (200 episodes, 1.96 GPU-h) | **0.000 ± 0.000** (200 episodes, 1.32 GPU-h) | **0.915 ± 0.020** (200 episodes, 1.59 GPU-h) | **0.925 ± 0.019** (200 episodes, 1.48 GPU-h) | **0.400 ± 0.035** (200 episodes, 1.52 GPU-h) | **0.185 ± 0.027** (200 episodes, 1.79 GPU-h) | **0.960 ± 0.014** (200 episodes, 0.39 GPU-h) | **0.810 ± 0.028** (200 episodes, 1.99 GPU-h) | **0.010 ± 0.007** (200 episodes, 2.59 GPU-h) | **0.915 ± 0.020** (200 episodes, 1.10 GPU-h) | **0.950 ± 0.028** (60 episodes, 0.34 GPU-h) | **0.010 ± 0.007** (200 episodes, 1.85 GPU-h) |

