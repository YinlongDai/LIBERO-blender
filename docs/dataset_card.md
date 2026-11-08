---
license: cc-by-4.0
pretty_name: LIBERO-Blender
viewer: false   # HDF5 files; the viewer would only find the JSON manifests
task_categories:
  - robotics
tags:
  - libero
  - robot-learning
  - imitation-learning
  - blender
  - rendering
size_categories:
  - 1M<n<10M
---

# LIBERO-Blender

The LIBERO demonstration datasets, re-rendered with Blender. All five suites of the original release
are included (libero_spatial, libero_object, libero_goal, libero_10 and libero_90: 130 files, 6,500
demos, 1,007,618 frames), each in two versions:

- `eevee/`: Blender EEVEE, a rasteriser with ray-traced shadows (64 samples). Deterministic.
- `cycles/`: Blender Cycles, a path tracer (256 samples, OptiX denoiser).

The states, actions, low-dim observations and attributes are LIBERO's own, copied unchanged. Only
the two camera images, `obs/agentview_rgb` and `obs/eye_in_hand_rgb`, are replaced by Blender renders
of the stored states. They have the same 128 x 128 shape, dtype, orientation, chunking and compression
as the originals, so any LIBERO data loader reads these files as they are.

![The middle frame of demo_0 in one task per suite: LIBERO's original image, EEVEE and Cycles](images/comparison_agentview.jpg)

The same frames from the wrist camera: [images/comparison_wrist.jpg](images/comparison_wrist.jpg). Shown
upscaled from the stored 128 x 128.

Code, render settings and policy results: https://github.com/YinlongDai/LIBERO-blender

## Layout

```
eevee/
  libero_10/<task>_demo.hdf5            one file per task, as in LIBERO
  libero_10/<task>_demo.manifest.json   how that file was made and checked
  ...                                   libero_90, libero_goal, libero_object, libero_spatial
  manifest.json                         all 130 per-file manifests, without the alignment samples
  summary.md                            per-suite checks, frames and render time
cycles/                                 the same
images/                                 the comparison images above
```

Each suite folder is about 6 GB (spatial, object, goal), 13 GB (libero_10) or 63 GB (libero_90).
Each renderer is about 101 GB in all.

## How the images were made

For every frame, the stored MuJoCo state is set in LIBERO's env, and the world poses of all bodies
and both cameras are sent to Blender. Blender renders the scene exported from that task's MuJoCo
model: LIBERO's own meshes, textures and materials. The lighting is an HDRI environment shipped with
Blender plus one soft area light above the table. The full render settings are in
`data.attrs["blender_render_config"]` of every file, and in `configs/render/eevee.yaml` and
`configs/render/cycles.yaml` (version 1) of the code repository.

## Checks

Every file passed these checks; the results are in its manifest:

- **Replay:** MuJoCo re-renders of the stored states match LIBERO's stored images, so replaying states
  reproduces the demos. This also confirms the stored image orientation, which is unflipped in every file.
- **Alignment:** a geom-id render from Blender is compared with MuJoCo's segmentation on 12 random
  camera-frames per file. All passed; the worst mean IoU is 0.9968.
- **Integrity:** each manifest records the file's sha256.

EEVEE renders are bit-identical across runs and between GPU models (L40S and A30 tested). Cycles
renders vary by up to ±1 grey level between runs.

## Usage

```python
import h5py

with h5py.File("eevee/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5") as f:
    demo = f["data/demo_0"]
    images = demo["obs/agentview_rgb"][()]   # (T, 128, 128, 3) uint8, Blender EEVEE
    actions = demo["actions"][()]            # LIBERO's original actions
```

To train on one renderer, point your LIBERO data path at `eevee/` or `cycles/` in place of LIBERO's
`datasets/` folder.

## Results

MoDE policies trained from scratch on LIBERO-90 + LIBERO-10 of each version, evaluated closed-loop on
LIBERO-10 with images from the same renderer (200 episodes each): 0.915 on EEVEE and 0.810 on Cycles.
For comparison, the same recipe trained on the original MuJoCo images scores 0.960 on MuJoCo images
and 0.000 on Blender images. More results, including a random-lighting eval, are in the code
repository's REPORT.md.

## License and citation

CC BY 4.0, like the LIBERO datasets these are rendered from. Please cite LIBERO:

```bibtex
@article{liu2023libero,
  title   = {LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author  = {Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal = {arXiv preprint arXiv:2306.03310},
  year    = {2023}
}
```
