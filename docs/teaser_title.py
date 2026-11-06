"""Title shot for the teaser: 3D "LIBERO / Blender" lettering on a dark mirror floor, rendered with Cycles.

Run by docs/make_teaser.py:
    blender -b --factory-startup --python docs/teaser_title.py -- --out <dir> --frames 75 --size 960

A bright light sweeps across the letters while the camera slowly moves in. Writes <dir>/0001.png ...
"""
import argparse
import math
import sys

import bpy

SILVER = (0.80, 0.82, 0.86, 1.0)
ORANGE = (0.92, 0.30, 0.03, 1.0)


def material(name, color, metallic, roughness):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def add_light(name, kind, energy, color, size, location):
    light = bpy.data.lights.new(name, kind)
    light.energy = energy
    light.color = color
    if kind == "AREA":
        light.size = size
    obj = bpy.data.objects.new(name, light)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    return obj


def aim(obj, target):
    constraint = obj.constraints.new("TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"


def keyframes(obj, path, values, frames):
    for value, frame in zip(values, frames):
        setattr(obj, path, value)
        obj.keyframe_insert(path, frame=frame)
    for fcurve in obj.animation_data.action.fcurves:
        for point in fcurve.keyframe_points:
            point.interpolation = "SINE"
            point.easing = "EASE_IN_OUT"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=75)
    parser.add_argument("--size", type=int, default=960)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    collection = scene.collection

    text = bpy.data.curves.new("title", "FONT")
    text.body = "LIBERO\nBlender"
    text.align_x, text.align_y = "CENTER", "CENTER"
    text.size = 0.95
    text.space_line = 0.9
    text.extrude = 0.14
    text.bevel_depth = 0.025
    text.bevel_resolution = 4
    text.materials.append(material("silver", SILVER, 1.0, 0.22))
    text.materials.append(material("orange", ORANGE, 0.3, 0.18))
    for i, char in enumerate(text.body_format):
        char.material_index = 0 if i < len("LIBERO") else 1
    title = bpy.data.objects.new("title", text)
    title.rotation_euler = (math.radians(90), 0, 0)  # stand the letters up, facing -Y
    collection.objects.link(title)

    floor = bpy.data.meshes.new("floor")
    floor.from_pydata([(-30, -30, 0), (30, -30, 0), (30, 30, 0), (-30, 30, 0)], [], [(0, 1, 2, 3)])
    floor.materials.append(material("mirror", (0.015, 0.016, 0.02, 1.0), 0.0, 0.12))
    floor_obj = bpy.data.objects.new("floor", floor)
    floor_obj.location.z = -1.05
    collection.objects.link(floor_obj)

    world = bpy.data.worlds.new("dark")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.005, 0.009, 1.0)
    scene.world = world

    target = bpy.data.objects.new("target", None)
    target.location = (0, 0, -0.1)
    collection.objects.link(target)
    key = add_light("key", "AREA", 250, (1.0, 0.97, 0.92), 4.0, (-2.5, -4.5, 4.0))
    rim = add_light("rim", "AREA", 600, (0.45, 0.6, 1.0), 6.0, (0.0, 4.0, 3.0))
    sweep = add_light("sweep", "AREA", 900, (1.0, 0.95, 0.9), 0.6, (-7.0, -3.0, 1.2))
    for light in (key, rim, sweep):
        aim(light, target)
    keyframes(sweep, "location", [(-7.0, -3.0, 1.2), (7.0, -3.0, 1.2)], [1, args.frames])

    camera_data = bpy.data.cameras.new("camera")
    camera_data.lens = 40
    camera = bpy.data.objects.new("camera", camera_data)
    collection.objects.link(camera)
    scene.camera = camera
    aim(camera, target)
    keyframes(camera, "location", [(-1.2, -7.5, 0.9), (0.5, -5.6, 0.45)], [1, args.frames])

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "OPTIX"
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.resolution_x = scene.render.resolution_y = args.size
    scene.render.image_settings.file_format = "PNG"
    scene.frame_start, scene.frame_end = 1, args.frames
    scene.render.filepath = f"{args.out}/"
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
