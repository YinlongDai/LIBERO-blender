"""Renderer: a Blender process (libero_blender/blender/server.py) that renders LIBERO scenes for this process.

    renderer = Renderer("configs/render/eevee.yaml")
    renderer.load(scene_dir)                      # a scene dump from libero_blender.scene.write_scene
    images = renderer.render(poses(env))          # {camera: uint8 (H, W, 3)}, top row first
    renderer.close()

Blender is started on a free localhost port and stopped by close(); its output goes to a log file
named in any startup error.
"""
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

from libero_blender.paths import BLENDER, ROOT
from libero_blender.protocol import recv_message, send_message

CAMERAS = ["agentview", "robot0_eye_in_hand"]
POSE_KEYS = ["body_xpos", "body_xquat", "cam_xpos", "cam_xmat", "site_alpha"]
STARTUP_TIMEOUT_S = 300


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Renderer:
    def __init__(self, render_config):
        self.config = yaml.safe_load(Path(render_config).read_text())
        self.tag = {"name": self.config["name"], "version": self.config["version"]}
        port = free_port()
        self.log = Path(tempfile.gettempdir()) / f"libero_blender_{port}.log"
        command = [BLENDER, "-b", "--factory-startup", "--python-exit-code", "1",
                   "--python", str(ROOT / "libero_blender" / "blender" / "server.py"), "--", "--port", str(port)]
        with open(self.log, "w") as log:
            self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        self.sock = self.connect(port)

    def connect(self, port):
        deadline = time.time() + STARTUP_TIMEOUT_S
        while True:
            try:
                sock = socket.create_connection(("127.0.0.1", port))
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return sock
            except ConnectionRefusedError:
                if self.process.poll() is not None or time.time() > deadline:
                    raise RuntimeError(f"Blender render server did not start; see {self.log}")
                time.sleep(0.5)

    def request(self, header, arrays=None):
        send_message(self.sock, header, arrays)
        reply, reply_arrays = recv_message(self.sock)
        if not reply["ok"]:
            raise RuntimeError(f"render server failed:\n{reply['error']}")
        return reply_arrays

    def load(self, scene_dir):
        """Build the scene (once per scene_dir) with the fixed lighting of the render config."""
        self.request({"cmd": "load", "scene_dir": str(scene_dir), "config": self.config})

    def set_lighting(self, sample):
        """Change the lighting until the next load: a sample from libero_blender.lighting.sample_lighting."""
        self.request({"cmd": "lighting", "sample": sample})

    def render(self, poses, cameras=CAMERAS, size=128, render_pass="rgb"):
        """render_pass="geom_id" gives int32 geom-id images (-1 = background) for alignment checks."""
        header = {"cmd": "render", "cameras": list(cameras), "width": size, "height": size, "pass": render_pass}
        return self.request(header, {key: poses[key] for key in POSE_KEYS})

    def close(self):
        self.request({"cmd": "quit"})
        self.sock.close()
        self.process.wait()
