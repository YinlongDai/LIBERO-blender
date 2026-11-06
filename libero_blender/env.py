"""BlenderEnv: a LIBERO env whose camera images come from Blender.

    env, task, init_states = make_env("libero_10", 0, image_size=128)
    env.set_init_state(init_states[0])
    env = BlenderEnv(env, "configs/render/eevee.yaml", lighting="configs/lighting/random.yaml")
    obs = env.reset()
    obs = env.set_init_state(init_states[0])
    obs, reward, done, info = env.step(action)

Everything goes straight to the wrapped env, so physics, init states and success checks are LIBERO's;
only `agentview_image` and `robot0_eye_in_hand_image` in the observations are replaced. They keep
robosuite's raw orientation, so existing preprocessing applies unchanged.

The scene is dumped from the env's state when the wrapper is made; the key light hangs above it. With
`lighting`, every reset draws a new random lighting, from `lighting_seed` if given. With
render_now=False an observation comes without images, which saves the render when a policy does not
look at it.
"""
import tempfile
from pathlib import Path

import numpy as np
import yaml

from libero_blender.lighting import sample_lighting
from libero_blender.renderer import Renderer
from libero_blender.scene import poses, write_scene

IMAGE_KEYS = {"agentview": "agentview_image", "robot0_eye_in_hand": "robot0_eye_in_hand_image"}


class BlenderEnv:
    def __init__(self, env, render_config, lighting=None, scene_dir=None, image_size=128, seed=0):
        self.env = env
        self.image_size = image_size
        self.lighting = yaml.safe_load(Path(lighting).read_text()) if lighting else None
        self.rng = np.random.default_rng(seed)
        self.lightings = []  # one per reset, when lighting is random
        self.temporary_scene = None if scene_dir else tempfile.TemporaryDirectory(prefix="libero_blender_scene_")
        self.scene_dir = scene_dir or self.temporary_scene.name
        write_scene(env, self.scene_dir, {})
        self.render_config = render_config
        self.renderer = Renderer(render_config)
        self.renderer.load(self.scene_dir)
        self.video_renderer = None  # started by the first render_camera

    def __getattr__(self, name):
        # Only called for attributes the wrapper does not have, e.g. check_success, sim, seed.
        return getattr(self.env, name)

    def reset(self, lighting_seed=None, render_now=True):
        if self.lighting:
            rng = np.random.default_rng(lighting_seed) if lighting_seed is not None else self.rng
            self.lightings.append(sample_lighting(rng, self.lighting))
            self.renderer.set_lighting(self.lightings[-1])
            if self.video_renderer:
                self.video_renderer.set_lighting(self.lightings[-1])
        return self.observe(self.env.reset(), render_now)

    def set_init_state(self, init_state, render_now=True):
        return self.observe(self.env.set_init_state(init_state), render_now)

    def render_camera(self, camera, size):
        """One camera at any size, upright, e.g. for videos.

        A second Blender process renders these: EEVEE keeps some state between renders, so extra
        renders in the first one would change the observations slightly.
        """
        if self.video_renderer is None:
            self.video_renderer = Renderer(self.render_config)
            self.video_renderer.load(self.scene_dir)
            if self.lightings:
                self.video_renderer.set_lighting(self.lightings[-1])
        return self.video_renderer.render(poses(self.env), [camera], size)[camera]

    def step(self, action, render_now=True):
        obs, reward, done, info = self.env.step(action)
        return self.observe(obs, render_now), reward, done, info

    def observe(self, obs, render_now):
        if not render_now:
            for key in IMAGE_KEYS.values():
                obs.pop(key, None)
            return obs
        images = self.renderer.render(poses(self.env), list(IMAGE_KEYS), self.image_size)
        for camera, key in IMAGE_KEYS.items():
            obs[key] = images[camera][::-1].copy()  # Blender gives top row first; raw robosuite is upside down
        return obs

    def close(self):
        self.renderer.close()
        if self.video_renderer:
            self.video_renderer.close()
        self.env.close()
        if self.temporary_scene:
            self.temporary_scene.cleanup()
