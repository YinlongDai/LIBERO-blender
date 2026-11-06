"""Blender render server: holds one LIBERO scene and renders camera images for MuJoCo poses.

Started by libero_blender.renderer.Renderer:
    blender -b --factory-startup --python-exit-code 1 --python libero_blender/blender/server.py -- --port 5555

Listens on localhost and serves one client at a time. Requests (format in libero_blender/protocol.py):
    load      {scene_dir, config}  build the scene unless it is already loaded, apply the render config
    lighting  {sample}             a random-lighting sample (libero_blender/lighting.py), kept until the next load
    render    {cameras, width, height, pass} + body_xpos, body_xquat, cam_xpos, cam_xmat, site_alpha
              -> one uint8 (H, W, 3) image per camera, top row first. pass="geom_id" returns int32 (H, W)
              geom ids instead (-1 = background), for alignment checks.
    quit
"""
import argparse
import socket
import sys
import tempfile
import traceback
from pathlib import Path

import bpy
import OpenImageIO as oiio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from libero_blender.blender.lighting import apply_lighting_sample  # noqa: E402
from libero_blender.blender.scene import ID_DIVISOR, build_scene, set_state  # noqa: E402
from libero_blender.blender.settings import apply_config, read_geom_ids, setup_geom_id_pass  # noqa: E402
from libero_blender.protocol import recv_message, send_message  # noqa: E402


class RenderServer:
    def __init__(self):
        self.frames_dir = Path(tempfile.mkdtemp(prefix="libero_blender_"))
        self.scene_dir = None
        self.config = None
        self.lighting_sample = None

    def load(self, header):
        scene_dir = str(Path(header["scene_dir"]).resolve())
        if scene_dir != self.scene_dir:
            self.info, self.body_empties, cameras = build_scene(scene_dir)
            self.cameras = {camera.name: camera for camera in cameras}
            self.scene_dir = scene_dir
        self.config = header["config"]
        self.lighting_sample = None
        self.apply_config()
        return {}

    def set_lighting(self, header):
        self.lighting_sample = header["sample"]
        self.apply_config()
        return {}

    def apply_config(self):
        scene = bpy.context.scene
        apply_config(scene, self.config, self.info["light_anchor"])
        if self.lighting_sample is not None:
            apply_lighting_sample(scene, self.info["light_anchor"], self.config["lighting"], self.lighting_sample)
        # PNG without compression: the frame only goes from Blender to this process.
        scene.render.image_settings.compression = 0

    def render(self, header, poses):
        scene = bpy.context.scene
        scene.render.resolution_x = header["width"]
        scene.render.resolution_y = header["height"]
        set_state(poses, self.body_empties, list(self.cameras.values()))
        if header["pass"] == "geom_id":
            return self.render_geom_ids(header["cameras"])

        images = {}
        for name in header["cameras"]:
            path = self.frames_dir / f"{name}.png"
            scene.camera = self.cameras[name]
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            images[name] = oiio.ImageBuf(str(path)).get_pixels(oiio.UINT8)[:, :, :3]
        return images

    def render_geom_ids(self, camera_names):
        scene = bpy.context.scene
        setup_geom_id_pass(scene)
        geom_ids = {}
        for name in camera_names:
            path = self.frames_dir / f"{name}_geom_id.exr"
            scene.camera = self.cameras[name]
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            geom_ids[name] = read_geom_ids(path, ID_DIVISOR).astype("int32")
        self.apply_config()  # back to the RGB settings
        return geom_ids


def serve(port, server):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(1)
    print(f"render server listening on 127.0.0.1:{port}", flush=True)
    handlers = {"load": server.load, "lighting": server.set_lighting}

    while True:
        connection, _ = listener.accept()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with connection:
            while True:
                try:
                    header, arrays = recv_message(connection)
                except ConnectionError:
                    break  # client went away; wait for the next one
                if header["cmd"] == "quit":
                    send_message(connection, {"ok": True})
                    return
                try:
                    if header["cmd"] == "render":
                        reply, reply_arrays = {"ok": True}, server.render(header, arrays)
                    else:
                        reply, reply_arrays = {"ok": True, **handlers[header["cmd"]](header)}, {}
                except Exception:
                    # The client raises this; the server keeps serving.
                    reply, reply_arrays = {"ok": False, "error": traceback.format_exc()}, {}
                send_message(connection, reply, reply_arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    serve(args.port, RenderServer())


if __name__ == "__main__":
    main()
