"""Teaser video for the README: a 3D title, LIBERO demo replays in MuJoCo and in Blender, then random lighting.

    python docs/make_teaser.py --out docs     # writes teaser.mp4

The title is rendered by docs/teaser_title.py; every other clip replays stored demo states. The Cycles
and EEVEE parts are short clips of many tasks, with MuJoCo on one side of a wipe that sweeps across each
clip; the last part changes the lighting every second.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import h5py
import imageio
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from libero_blender.lighting import sample_lighting
from libero_blender.paths import BLENDER, LIBERO_DATASETS, ROOT
from libero_blender.renderer import Renderer
from libero_blender.scene import find_task_id, make_env, poses, rebuild_render_context, set_sim_state, write_scene

SIZE = 960
FPS = 30
CLIP_FRAMES = 45          # 1.5 s per task
CLIP_START = 0.3          # where in the demo a clip starts
STRIDE = 2                # demo states per video frame
LIGHTING_FRAMES = 150     # the random-lighting part
LIGHTING_EVERY = 30
CROSSFADE = 6
TITLE_FRAMES = 75
CLIPS = {
    "cycles": [
        ("libero_10", "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5"),
        ("libero_goal", "put_the_wine_bottle_on_the_rack_demo.hdf5"),
        ("libero_object", "pick_up_the_ketchup_and_place_it_in_the_basket_demo.hdf5"),
        ("libero_10", "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5"),
        ("libero_90", "KITCHEN_SCENE7_open_the_microwave_demo.hdf5"),
    ],
    "eevee": [
        ("libero_10", "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5"),
        ("libero_goal", "open_the_middle_drawer_of_the_cabinet_demo.hdf5"),
        ("libero_spatial", "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"),
        ("libero_10", "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5"),
        ("libero_90", "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_demo.hdf5"),
    ],
}
CAPTIONS = {"cycles": "Blender Cycles", "eevee": "Blender EEVEE"}
LIGHTING_CLIP = ("libero_10", "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5")


def label(draw, text, xy, size=44, alpha=1.0, anchor="la"):
    """White text on a rounded dark pill."""
    font = ImageFont.load_default(size=size)
    left, top, right, bottom = draw.textbbox(xy, text, font=font, anchor=anchor)
    draw.rounded_rectangle((left - 20, top - 12, right + 20, bottom + 12), radius=24, fill=(0, 0, 0, int(140 * alpha)))
    draw.text(xy, text, font=font, anchor=anchor, fill=(255, 255, 255, int(255 * alpha)))


def overlay(frame, draw_fn):
    image = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", image.size)
    draw_fn(ImageDraw.Draw(layer))
    return np.array(Image.alpha_composite(image, layer).convert("RGB"))


def wipe(mujoco, blender, position):
    """MuJoCo left of `position` (0-1), Blender right of it, with a soft white divider."""
    x = int(position * SIZE)
    frame = blender.copy()
    frame[:, :x] = mujoco[:, :x]
    for offset, strength in [(0, 1.0), (1, 1.0), (-1, 0.6), (2, 0.6), (-2, 0.25), (3, 0.25)]:
        column = np.clip(x + offset, 0, SIZE - 1)
        frame[:, column] = (frame[:, column] * (1 - strength) + 255 * strength).astype(np.uint8)
    return frame


def crossfade(frames, new_frames, length=CROSSFADE):
    for k in range(min(length, len(frames))):
        mix = (k + 1) / (length + 1)
        new_frames[k] = (frames[-length + k] * (1 - mix) + new_frames[k] * mix).astype(np.uint8)
    return frames[:-length] + new_frames if frames else new_frames


def open_clip(suite, file, renderer, num_frames, scene_root):
    """The env and the demo states to play, with the renderer loaded on this task's scene."""
    with h5py.File(LIBERO_DATASETS / suite / file, "r") as f:
        states = f["data/demo_0/states"][()]
    env, _, _ = make_env(suite, find_task_id(suite, file), SIZE)
    # MuJoCo's offscreen buffer is 640 x 480 by default; make room for the larger frames.
    model = env.sim.model._model
    model.vis.global_.offwidth = model.vis.global_.offheight = SIZE
    rebuild_render_context(env, offsamples=model.vis.quality.offsamples)
    set_sim_state(env, states[0])  # the scene is dumped from the first state, as for the datasets
    write_scene(env, scene_root / Path(file).stem, {})
    renderer.load(scene_root / Path(file).stem)
    start = min(int(CLIP_START * len(states)), max(0, len(states) - num_frames * STRIDE))
    return env, states[start::STRIDE][:num_frames]


def wipe_clip(clip, index, renderer, caption, scene_root):
    env, states = open_clip(*clip, renderer, CLIP_FRAMES, scene_root)
    frames = []
    for i, state in enumerate(states):
        set_sim_state(env, state)
        t = (i / (len(states) - 1)) if index % 2 == 0 else 1 - i / (len(states) - 1)  # alternate directions
        position = 0.12 + 0.76 * (0.5 - 0.5 * np.cos(np.pi * t))  # eased sweep
        mujoco = env.sim.render(SIZE, SIZE, camera_name="agentview")[::-1]
        blender = renderer.render(poses(env), ["agentview"], SIZE)["agentview"]
        frames.append(overlay(wipe(mujoco, blender, position), lambda d: (
            label(d, "MuJoCo", (32, SIZE - 40), anchor="ld"), label(d, caption, (SIZE - 32, SIZE - 40), anchor="rd"))))
    env.close()
    print(f"{clip[1][:50]}: {len(frames)} frames", flush=True)
    return frames


def lighting_clip(renderer, ranges, scene_root):
    env, states = open_clip(*LIGHTING_CLIP, renderer, LIGHTING_FRAMES, scene_root)
    rng = np.random.default_rng(3)
    lightings = [sample_lighting(rng, ranges) for _ in range(len(states) // LIGHTING_EVERY + 2)]
    frames = []
    for i, state in enumerate(states):
        set_sim_state(env, state)
        current, phase = divmod(i, LIGHTING_EVERY)
        renderer.set_lighting(lightings[current])
        frame = renderer.render(poses(env), ["agentview"], SIZE)["agentview"]
        if phase >= LIGHTING_EVERY - CROSSFADE:  # fade into the next lighting
            mix = (phase - (LIGHTING_EVERY - CROSSFADE) + 1) / (CROSSFADE + 1)
            renderer.set_lighting(lightings[current + 1])
            frame = (frame * (1 - mix) + renderer.render(poses(env), ["agentview"], SIZE)["agentview"] * mix)
        frames.append(overlay(frame.astype(np.uint8), lambda d: label(d, "Random lighting", (SIZE - 32, SIZE - 40),
                                                                      anchor="rd")))
    env.close()
    return frames


def title_frames(scene_root):
    """The 3D title rendered by docs/teaser_title.py, faded in from black, with a subtitle."""
    out = scene_root / "title"
    subprocess.run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "--python",
                    str(ROOT / "docs" / "teaser_title.py"), "--", "--out", str(out), "--frames", str(TITLE_FRAMES),
                    "--size", str(SIZE)], check=True, stdout=subprocess.DEVNULL)
    frames = [imageio.imread(path)[:, :, :3] for path in sorted(out.glob("*.png"))]
    for i, frame in enumerate(frames):
        frame = (frame * min(1.0, i / 15)).astype(np.uint8)
        alpha = min(1.0, max(0.0, (i - 30) / 15))
        frames[i] = overlay(frame, lambda d: d.text((SIZE // 2, SIZE - 90), "same physics  ·  Blender rendering",
                                                    font=ImageFont.load_default(size=40), anchor="mm",
                                                    fill=(235, 235, 240, int(255 * alpha))))
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "docs"))
    parser.add_argument("--stills", type=int, default=0, help="also save this many evenly spaced frames as PNG")
    args = parser.parse_args()
    out = Path(args.out)
    ranges = yaml.safe_load((ROOT / "configs" / "lighting" / "random.yaml").read_text())
    renderers = {name: Renderer(ROOT / "configs" / "render" / f"{name}.yaml") for name in ("eevee", "cycles")}

    with tempfile.TemporaryDirectory() as scene_root:
        frames = title_frames(Path(scene_root))
        for name in ("cycles", "eevee"):
            for index, clip in enumerate(CLIPS[name]):
                clip_frames = wipe_clip(clip, index, renderers[name], CAPTIONS[name], Path(scene_root))
                frames = crossfade(frames, clip_frames, length=15 if (name, index) == ("cycles", 0) else CROSSFADE)
        frames = crossfade(frames, lighting_clip(renderers["eevee"], ranges, Path(scene_root)))
    for renderer in renderers.values():
        renderer.close()

    # 4 Mbit/s keeps the video under GitHub's 10 MB limit for videos in a README.
    imageio.mimwrite(out / "teaser.mp4", frames, fps=FPS, codec="libx264", bitrate="4M", pixelformat="yuv420p")
    print(f"wrote {out / 'teaser.mp4'}: {len(frames) / FPS:.1f} s, {(out / 'teaser.mp4').stat().st_size / 1e6:.1f} MB")
    for i in np.linspace(0, len(frames) - 1, args.stills).astype(int):
        imageio.imwrite(out / f"teaser_{i:04d}.png", frames[i])


if __name__ == "__main__":
    main()
