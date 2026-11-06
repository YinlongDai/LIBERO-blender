"""Scene lighting for the Blender renders, from the `lighting` section of a render config.

set_studio_lighting gives fixed values and a fixed light position per scene, so renders repeat exactly.
apply_lighting_sample changes it for one episode, from a sample of libero_blender/lighting.py.
"""
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector

AMBIENT_NODES = ["ambient_coordinates", "ambient_rotation", "ambient_tint"]


def set_studio_lighting(scene, light_anchor, lighting):
    """HDRI world plus one soft area light above light_anchor = (x, y, top_z) from scene.json.

    Safe to call again with another config: the world and the light are reused, not duplicated, and
    anything a lighting sample changed is put back.
    """
    hdri_path = Path(bpy.utils.system_resource("DATAFILES", path="studiolights/world")) / lighting["hdri"]
    assert hdri_path.exists(), f"missing {hdri_path}"

    world = bpy.data.worlds.get("studio")
    if world is None:
        world = bpy.data.worlds.new("studio")
        world.use_nodes = True
        nodes, links = world.node_tree.nodes, world.node_tree.links
        environment = nodes.new("ShaderNodeTexEnvironment")
        links.new(environment.outputs["Color"], nodes["Background"].inputs["Color"])
    nodes = world.node_tree.nodes
    environment = next(node for node in nodes if node.type == "TEX_ENVIRONMENT")
    environment.image = bpy.data.images.load(str(hdri_path), check_existing=True)
    nodes["Background"].inputs["Strength"].default_value = lighting["hdri_strength"]
    if nodes.get("ambient_tint") is not None:
        # Remove a lighting sample's nodes, so the world is exactly the fixed config's again.
        for name in AMBIENT_NODES:
            nodes.remove(nodes[name])
        world.node_tree.links.new(environment.outputs["Color"], nodes["Background"].inputs["Color"])
    scene.world = world

    light = bpy.data.lights.get("key_light") or bpy.data.lights.new("key_light", type="AREA")
    light.energy = lighting["key_light_power_w"]
    light.size = lighting["key_light_size_m"]
    light.color = (1.0, 1.0, 1.0)
    light.use_shadow_jitter = lighting["key_light_shadow_jitter"]
    light.shadow_jitter_overblur = lighting["key_light_shadow_overblur_percent"]
    light_obj = bpy.data.objects.get("key_light")
    if light_obj is None:
        light_obj = bpy.data.objects.new("key_light", light)
        scene.collection.objects.link(light_obj)
    x, y, top_z = light_anchor
    light_obj.location = (x, y, top_z + lighting["key_light_height_above_scene_m"])  # area lights face -Z
    light_obj.rotation_mode = "QUATERNION"
    light_obj.rotation_quaternion = Quaternion()


def socket(sockets, identifier):
    # The Mix node has float, vector and colour sockets that share display names, so look up by identifier.
    return next(s for s in sockets if s.identifier == identifier)


def ambient_nodes(world):
    """Rotation and tint nodes around the HDRI environment texture, added the first time a sample is applied."""
    nodes, links = world.node_tree.nodes, world.node_tree.links
    if nodes.get("ambient_tint") is None:
        environment = next(node for node in nodes if node.type == "TEX_ENVIRONMENT")
        coordinates = nodes.new("ShaderNodeTexCoord")
        coordinates.name = "ambient_coordinates"
        rotation = nodes.new("ShaderNodeMapping")
        rotation.name = "ambient_rotation"
        links.new(coordinates.outputs["Generated"], rotation.inputs["Vector"])
        links.new(rotation.outputs["Vector"], environment.inputs["Vector"])
        tint = nodes.new("ShaderNodeMix")
        tint.name = "ambient_tint"
        tint.data_type = "RGBA"
        tint.blend_type = "MULTIPLY"
        tint.inputs["Factor"].default_value = 1.0
        links.new(environment.outputs["Color"], socket(tint.inputs, "A_Color"))
        links.new(socket(tint.outputs, "Result_Color"), nodes["Background"].inputs["Color"])
    return nodes["ambient_rotation"], nodes["ambient_tint"]


def apply_lighting_sample(scene, light_anchor, lighting, sample):
    """Move, aim and recolour the key light, turn and tint the fill light, scale both; the config's values are the base."""
    light_obj = bpy.data.objects["key_light"]
    target = Vector(light_anchor)
    light_obj.location = target + Vector(sample["key_light_offset_m"])
    light_obj.rotation_mode = "QUATERNION"
    light_obj.rotation_quaternion = (target - light_obj.location).to_track_quat("-Z", "Y")
    # Same light reaching the scene centre as the fixed light at key_light_height_above_scene_m straight above.
    distance_ratio = (light_obj.location - target).length / lighting["key_light_height_above_scene_m"]
    light_obj.data.energy = lighting["key_light_power_w"] * distance_ratio ** 2 * sample["brightness"]
    light_obj.data.color = sample["key_light_color"]

    rotation, tint = ambient_nodes(scene.world)
    rotation.inputs["Rotation"].default_value = (0.0, 0.0, np.radians(sample["ambient_rotation_deg"]))
    socket(tint.inputs, "B_Color").default_value = (*sample["ambient_color"], 1.0)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = \
        lighting["hdri_strength"] * sample["brightness"]
