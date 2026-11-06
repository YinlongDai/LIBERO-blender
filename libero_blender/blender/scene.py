"""Build a Blender scene from a scene dump (libero_blender/scene.py), and move it to a MuJoCo state.

Runs inside Blender. Every MuJoCo body with visual geoms becomes an empty named body_<id>, and
every visual geom becomes a mesh object parented to its body with the geom's local offset.
Moving to a new state only sets the body empties and cameras; nothing is rebuilt.
"""
import json
import math
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

# Geom ids are written into each object's color for the id render; this keeps the values in [0, 1].
ID_DIVISOR = 65536.0
INFINITE_PLANE_HALF_SIZE = 10.0
SPHERE_SEGMENTS = 32
SPHERE_RINGS = 16
CYLINDER_SEGMENTS = 32


def load_dump(scene_dir):
    scene_dir = Path(scene_dir)
    info = json.loads((scene_dir / "scene.json").read_text())
    meshes = dict(np.load(scene_dir / "meshes.npz"))
    return info, meshes


def srgb_to_linear(value):
    # MuJoCo colors are display (sRGB) values; Blender shaders work in linear.
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def primitive_mesh(name, geom_type, size):
    bm = bmesh.new()
    if geom_type == "box":
        bmesh.ops.create_cube(bm, size=2.0)
        bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    elif geom_type == "sphere":
        bmesh.ops.create_uvsphere(bm, u_segments=SPHERE_SEGMENTS, v_segments=SPHERE_RINGS, radius=size[0])
    elif geom_type == "ellipsoid":
        bmesh.ops.create_uvsphere(bm, u_segments=SPHERE_SEGMENTS, v_segments=SPHERE_RINGS, radius=1.0)
        bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    elif geom_type == "cylinder":
        radius, half_length = size[0], size[1]
        bmesh.ops.create_cone(bm, cap_ends=True, segments=CYLINDER_SEGMENTS,
                              radius1=radius, radius2=radius, depth=2 * half_length)
    elif geom_type == "capsule":
        # A sphere whose top half is pushed up and bottom half pushed down by the half length.
        radius, half_length = size[0], size[1]
        bmesh.ops.create_uvsphere(bm, u_segments=SPHERE_SEGMENTS, v_segments=SPHERE_RINGS, radius=radius)
        for vert in bm.verts:
            vert.co.z += half_length if vert.co.z >= 0 else -half_length
    elif geom_type == "plane":
        half_x = size[0] or INFINITE_PLANE_HALF_SIZE
        half_y = size[1] or INFINITE_PLANE_HALF_SIZE
        corners = [(-half_x, -half_y, 0), (half_x, -half_y, 0), (half_x, half_y, 0), (-half_x, half_y, 0)]
        bm.faces.new([bm.verts.new(c) for c in corners])
    else:
        raise ValueError(f"no primitive for geom type {geom_type}")

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    if geom_type in ("sphere", "ellipsoid", "capsule"):
        mesh.shade_smooth()
    return mesh


def mesh_from_arrays(name, verts, faces, normals, uvs):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    # Faces are triangles in MuJoCo's order, so loop i of face f is corner 3 * f + i.
    mesh.shade_smooth()
    mesh.normals_split_custom_set(normals.reshape(-1, 3).tolist())
    if uvs is not None:
        # MuJoCo flipped v when it loaded the mesh; flip it back for Blender's image convention.
        blender_uvs = uvs.reshape(-1, 2).copy()
        blender_uvs[:, 1] = 1.0 - blender_uvs[:, 1]
        mesh.uv_layers.new(name="UVMap").data.foreach_set("uv", blender_uvs.ravel())
    mesh.update()
    return mesh


def add_texture(material, bsdf, mat_info, color, geom, has_uvs, scene_dir):
    nodes, links = material.node_tree.nodes, material.node_tree.links
    image = nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(Path(scene_dir) / mat_info["texture"]["file"]), check_existing=True)

    repeat_x, repeat_y = mat_info["texrepeat"]
    mapping = nodes.new("ShaderNodeMapping")
    coords = nodes.new("ShaderNodeTexCoord")
    if has_uvs:
        links.new(coords.outputs["UV"], mapping.inputs["Vector"])
        mapping.inputs["Scale"].default_value = (repeat_x, repeat_y, 1.0)
    elif mat_info["texture"]["type"] == "cube":
        links.new(coords.outputs["Object"], mapping.inputs["Vector"])
        image.projection = "BOX"
    else:
        # MuJoCo maps 2D textures on primitives linearly from the geom's local x and y. With
        # texuniform the repeat is per meter, otherwise it is spread over the geom's full size.
        links.new(coords.outputs["Object"], mapping.inputs["Vector"])
        if mat_info["texuniform"]:
            mapping.inputs["Scale"].default_value = (repeat_x, repeat_y, 1.0)
        else:
            size_x = geom["size"][0] or INFINITE_PLANE_HALF_SIZE
            size_y = geom["size"][1] or INFINITE_PLANE_HALF_SIZE
            mapping.inputs["Scale"].default_value = (repeat_x / (2 * size_x), repeat_y / (2 * size_y), 1.0)
    links.new(mapping.outputs["Vector"], image.inputs["Vector"])

    # MuJoCo multiplies the texture by the material color.
    tint = nodes.new("ShaderNodeVectorMath")
    tint.operation = "MULTIPLY"
    tint.inputs[1].default_value = color[:3]
    links.new(image.outputs["Color"], tint.inputs[0])
    links.new(tint.outputs["Vector"], bsdf.inputs["Base Color"])


def make_material(geom, mat_info, has_uvs, scene_dir):
    material = bpy.data.materials.new(f"geom_{geom['id']}")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    color = [srgb_to_linear(c) for c in geom["rgba"][:3]] + [1.0]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Alpha"].default_value = geom["rgba"][3]

    if mat_info is not None:
        bsdf.inputs["Roughness"].default_value = min(1.0, max(0.05, 1.0 - mat_info["shininess"]))
        bsdf.inputs["Specular IOR Level"].default_value = mat_info["specular"]
        if mat_info["emission"] > 0:
            bsdf.inputs["Emission Color"].default_value = color
            bsdf.inputs["Emission Strength"].default_value = mat_info["emission"]
        if mat_info["texture"] is not None:
            add_texture(material, bsdf, mat_info, color, geom, has_uvs, scene_dir)
    return material


def geom_mesh(geom, meshes):
    name = f"geom_{geom['id']}"
    if geom["type"] != "mesh":
        return primitive_mesh(name, geom["type"], geom["size"]), False
    key = f"mesh{geom['mesh']}"
    uvs = meshes.get(f"{key}_uvs")
    mesh = mesh_from_arrays(name, meshes[f"{key}_verts"], meshes[f"{key}_faces"], meshes[f"{key}_normals"], uvs)
    return mesh, uvs is not None


def build_scene(scene_dir):
    """Build the scene into an empty Blender file. Returns (info, body_empties, cameras)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    info, meshes = load_dump(scene_dir)
    collection = bpy.context.scene.collection

    body_empties = {}
    for body in info["bodies"]:
        empty = bpy.data.objects.new(f"body_{body['id']}", None)
        empty["mujoco_name"] = body["name"]
        empty.rotation_mode = "QUATERNION"
        collection.objects.link(empty)
        body_empties[body["id"]] = empty

    for geom in info["geoms"]:
        mesh, has_uvs = geom_mesh(geom, meshes)
        mat_info = info["materials"].get(str(geom["material"]))
        mesh.materials.append(make_material(geom, mat_info, has_uvs, scene_dir))

        obj = bpy.data.objects.new(f"geom_{geom['id']}", mesh)
        obj["mujoco_name"] = geom["name"]
        obj.parent = body_empties[geom["body"]]
        obj.rotation_mode = "QUATERNION"
        obj.location = geom["pos"]
        obj.rotation_quaternion = geom["quat"]  # MuJoCo and Blender both store w, x, y, z
        obj.color = ((geom["id"] + 1) / ID_DIVISOR, 0.0, 0.0, 1.0)
        collection.objects.link(obj)

    # Sites get their own collection; set_state shows or hides each one from MuJoCo's site alpha.
    sites = bpy.data.collections.new("sites")
    collection.children.link(sites)
    for site in info["sites"]:
        mesh = primitive_mesh(f"site_{site['id']}", site["type"], site["size"])
        # Drawn opaque when visible: MuJoCo toggles these between alpha 0 and 1.
        color = {"id": f"site{site['id']}", "rgba": site["rgba"][:3] + [1.0]}
        mesh.materials.append(make_material(color, None, False, scene_dir))
        obj = bpy.data.objects.new(f"site_{site['id']}", mesh)
        obj["mujoco_name"] = site["name"]
        obj["site_id"] = site["id"]
        obj.parent = body_empties[site["body"]]
        obj.rotation_mode = "QUATERNION"
        obj.location = site["pos"]
        obj.rotation_quaternion = site["quat"]
        obj.color = (0.0, 0.0, 0.0, 1.0)  # decodes to background in the geom-id pass, which ignores sites
        sites.objects.link(obj)

    cameras = []
    for cam in info["cameras"]:
        cam_data = bpy.data.cameras.new(cam["name"])
        # MuJoCo's fovy is the vertical field of view.
        cam_data.sensor_fit = "VERTICAL"
        cam_data.angle_y = math.radians(cam["fovy"])
        cam_data.clip_start, cam_data.clip_end = cam["znear"], cam["zfar"]
        cam_obj = bpy.data.objects.new(cam["name"], cam_data)
        collection.objects.link(cam_obj)
        cameras.append(cam_obj)

    scene = bpy.context.scene
    scene.render.resolution_x = info["image_size"]
    scene.render.resolution_y = info["image_size"]
    scene.render.resolution_percentage = 100
    return info, body_empties, cameras


def set_state(state, body_empties, cameras):
    for body_id, empty in body_empties.items():
        empty.location = state["body_xpos"][body_id]
        empty.rotation_quaternion = state["body_xquat"][body_id]

    for obj in bpy.data.collections["sites"].objects:
        obj.hide_render = bool(state["site_alpha"][obj["site_id"]] <= 0)

    # Both MuJoCo and Blender cameras look down their local -Z with +Y up, so the MuJoCo camera
    # frame can be used as the Blender camera's world matrix directly.
    for i, cam_obj in enumerate(cameras):
        rotation = state["cam_xmat"][i].reshape(3, 3)
        position = state["cam_xpos"][i]
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = position
        cam_obj.matrix_world = Matrix(matrix.tolist())
