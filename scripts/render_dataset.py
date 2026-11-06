"""Re-render one LIBERO demo file with Blender, keeping LIBERO's HDF5 layout.

    python scripts/render_dataset.py --render configs/render/eevee.yaml --suite libero_10 \
        --file KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5

Reads <libero_datasets>/<suite>/<file> and writes <rendered_datasets>/<render name>/<suite>/<file> (paths
from configs/paths.yaml): a copy in which only obs/agentview_rgb and obs/eye_in_hand_rgb are replaced
by Blender renders of the stored states, with the same shape, dtype, chunking and compression. Next to it,
<file stem>.manifest.json records how the file was made and the results of the checks in configs/dataset.yaml:
  - MuJoCo's own render of a few stored states must match the stored images, which also tells the
    stored image orientation;
  - random frames get the geom-id alignment check against MuJoCo segmentation.
Resumable: finished demos are marked with the attribute blender_rendered and skipped on a restart.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

from libero_blender.alignment import FLIPS, compare
from libero_blender.paths import LIBERO_DATASETS, RENDERED_DATASETS, ROOT
from libero_blender.renderer import Renderer
from libero_blender.scene import find_task_id, make_env, poses, rebuild_render_context, set_sim_state, write_scene

IMAGE_KEYS = {"agentview": "agentview_rgb", "robot0_eye_in_hand": "eye_in_hand_rgb"}  # camera -> dataset key


def stored_orientation(env, demo, image_size, checks):
    """Which flip turns a raw robosuite image into the stored image, and how close they are."""
    frames = np.linspace(0, len(demo["states"]) - 1, checks["orientation_check_frames"]).astype(int)
    errors = {name: [] for name in FLIPS}
    for t in frames:
        set_sim_state(env, demo["states"][t])
        for camera, key in IMAGE_KEYS.items():
            raw = env.sim.render(image_size, image_size, camera_name=camera).astype(np.float32)
            for name, flip in FLIPS.items():
                errors[name].append(float(np.abs(flip(raw) - demo["obs"][key][t]).mean()))
    means = {name: round(float(np.mean(values)), 2) for name, values in errors.items()}
    best = min(means, key=means.get)
    assert means[best] < checks["max_orientation_diff"], f"stored images do not match MuJoCo re-renders: {means}"
    return best, means


def render_demo(env, renderer, source_demo, target_group, flip, image_size):
    for key, value in source_demo.attrs.items():
        target_group.attrs[key] = value
    for name, item in source_demo.items():
        if name != "obs":
            source_demo.copy(name, target_group)
    obs = target_group.create_group("obs")
    for name in source_demo["obs"]:
        if name not in IMAGE_KEYS.values():
            source_demo["obs"].copy(name, obs)

    images = {key: [] for key in IMAGE_KEYS.values()}
    for state in source_demo["states"]:
        set_sim_state(env, state)
        rendered = renderer.render(poses(env), list(IMAGE_KEYS), image_size)
        for camera, key in IMAGE_KEYS.items():
            # Blender gives top row first, raw robosuite images are that flipped, and the stored
            # images are raw robosuite images with `flip` applied.
            images[key].append(flip(rendered[camera][::-1]))
    for key, frames in images.items():
        original = source_demo["obs"][key]
        obs.create_dataset(key, data=np.stack(frames).astype(original.dtype), chunks=original.chunks,
                           compression=original.compression, compression_opts=original.compression_opts)
    target_group.attrs["blender_rendered"] = True


def alignment_samples(env, renderer, source, demo_names, image_size, scene, checks):
    rng = np.random.default_rng(checks["seed"])
    samples = []
    for demo_name in rng.choice(demo_names, size=min(checks["alignment_demos"], len(demo_names)), replace=False):
        states = source["data"][demo_name]["states"]
        for t in rng.choice(len(states), size=checks["alignment_frames_per_demo"], replace=False):
            set_sim_state(env, states[t])
            geom_ids = renderer.render(poses(env), list(IMAGE_KEYS), image_size, render_pass="geom_id")
            for camera in IMAGE_KEYS:
                seg = env.sim.render(image_size, image_size, camera_name=camera, segmentation=True)
                result = compare(seg, geom_ids[camera][::-1], scene)
                samples.append({"demo": str(demo_name), "frame": int(t), "camera": camera,
                                "mean_iou": round(result["mean_iou"], 4), "passed": result["passed"]})
    return samples


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", required=True, help="configs/render/*.yaml")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--file", required=True, help="demo file name inside the suite's folder")
    parser.add_argument("--checks", default=str(ROOT / "configs" / "dataset.yaml"))
    parser.add_argument("--local-dir", help="node-local disk to work on; the result is copied out at the end")
    args = parser.parse_args()
    checks = yaml.safe_load(Path(args.checks).read_text())
    start = time.time()

    renderer = Renderer(args.render)
    out_dir = RENDERED_DATASETS / renderer.config["name"]
    source_path = LIBERO_DATASETS / args.suite / args.file
    target_path = out_dir / args.suite / args.file
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Shared network filesystems now and then return I/O errors, which can kill hour-long renders
    # mid-file. With --local-dir, the shared filesystem is only touched by whole-file copies.
    work_source, work_target = source_path, target_path
    if args.local_dir:
        local = Path(args.local_dir)
        local.mkdir(parents=True, exist_ok=True)
        work_source, work_target = local / f"source_{args.file}", local / args.file
        shutil.copyfile(source_path, work_source)
        if target_path.exists():
            shutil.copyfile(target_path, work_target)  # resume a file an earlier run copied out unfinished

    with h5py.File(work_source, "r") as source:
        image_size = source["data/demo_0/obs/agentview_rgb"].shape[1]
        env, task, _ = make_env(args.suite, find_task_id(args.suite, args.file), image_size)
        flip_name, orientation_errors = stored_orientation(env, source["data/demo_0"], image_size, checks)

        scene_dir = out_dir / "scenes" / args.suite / Path(args.file).stem
        if not (scene_dir / "scene.json").exists():
            set_sim_state(env, source["data/demo_0/states"][0])
            write_scene(env, scene_dir, {"suite": args.suite, "task": task.name, "demo_file": args.file})
        renderer.load(scene_dir)

        demo_names = sorted(source["data"].keys(), key=lambda name: int(name.split("_")[1]))
        skipped = 0
        with h5py.File(work_target, "a") as target:
            data = target.require_group("data")
            for key, value in source["data"].attrs.items():
                data.attrs[key] = value
            data.attrs["blender_render_config"] = json.dumps(renderer.tag)
            for demo_name in demo_names:
                if demo_name in data and data[demo_name].attrs.get("blender_rendered", False):
                    skipped += 1
                    continue
                if demo_name in data:
                    del data[demo_name]  # half-written by an interrupted run
                render_demo(env, renderer, source["data"][demo_name], data.create_group(demo_name),
                            FLIPS[flip_name], image_size)
                target.flush()
                print(f"{args.file} {demo_name} done", flush=True)

        scene = json.loads((scene_dir / "scene.json").read_text())
        rebuild_render_context(env, offsamples=0)  # multisampling would blend geom ids at edges
        samples = alignment_samples(env, renderer, source, demo_names, image_size, scene, checks)
        num_frames = sum(len(source["data"][name]["states"]) for name in demo_names)
    env.close()
    renderer.close()
    if args.local_dir:
        partial = target_path.with_name(target_path.name + ".partial")
        shutil.copyfile(work_target, partial)
        os.replace(partial, target_path)

    manifest = {
        "suite": args.suite, "demo_file": args.file, "source": f"LIBERO/{args.suite}/{args.file}",
        "output": f"{args.suite}/{args.file}", "render_config": renderer.tag,
        "git_commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip(),
        "image_size": image_size, "num_demos": len(demo_names), "num_frames": num_frames,
        "demos_skipped_as_done": skipped, "stored_orientation_flip": flip_name,
        "orientation_mean_abs_diff": orientation_errors, "alignment_samples": samples,
        "alignment_all_passed": all(s["passed"] for s in samples), "sha256": sha256(work_target),
        "wall_seconds": round(time.time() - start, 1),
    }
    (target_path.parent / f"{Path(args.file).stem}.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ("demo_file", "num_frames", "alignment_all_passed")}))


if __name__ == "__main__":
    main()
