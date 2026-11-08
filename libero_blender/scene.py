"""The MuJoCo side: LIBERO envs, demo states, and the scene files Blender builds its copy of the scene from.

A scene dump (write_scene) holds the visual geoms, materials, textures and cameras of one task, plus
`light_anchor`, the point the key light hangs above. poses() is what Blender needs per frame.
"""
import json
from pathlib import Path

import imageio
import mujoco
import numpy as np

import libero_blender.paths  # noqa: F401  (before libero, which reads its config on import)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

IMAGE_SIZE = 256
CAMERAS = ["agentview", "robot0_eye_in_hand"]
DEFAULT_GEOM_RGBA = np.array([0.5, 0.5, 0.5, 1.0])

GEOM_TYPES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}
TEXTURE_TYPES = {int(mujoco.mjtTexture.mjTEXTURE_2D): "2d", int(mujoco.mjtTexture.mjTEXTURE_CUBE): "cube"}


def name_of(model, obj_type, obj_id):
    return mujoco.mj_id2name(model, obj_type, obj_id) or ""


def geom_rgba(model, geom_id):
    # MuJoCo draws the material color when the geom keeps the default rgba.
    rgba = model.geom_rgba[geom_id]
    mat_id = model.geom_matid[geom_id]
    if mat_id >= 0 and np.allclose(rgba, DEFAULT_GEOM_RGBA):
        return model.mat_rgba[mat_id]
    return rgba


def skip_reason(model, geom_id, rendered_groups):
    """Why this geom would not show up in robosuite's camera images, or None if it would."""
    group = model.geom_group[geom_id]
    if group >= len(rendered_groups) or not rendered_groups[group]:
        return f"group {group} not rendered"
    if geom_rgba(model, geom_id)[3] == 0:
        return "fully transparent"
    if model.geom_type[geom_id] not in GEOM_TYPES:
        return f"unsupported geom type {model.geom_type[geom_id]}"
    return None


def texture_pixels(model, tex_id):
    height, width, channels = model.tex_height[tex_id], model.tex_width[tex_id], model.tex_nchannel[tex_id]
    start = model.tex_adr[tex_id]
    return model.tex_data[start:start + height * width * channels].reshape(height, width, channels)


def save_texture(model, tex_id, texture_dir):
    tex_type = model.tex_type[tex_id]
    assert tex_type in TEXTURE_TYPES, f"texture {tex_id} has type {tex_type} (skybox?) on a geom material"
    pixels = texture_pixels(model, tex_id)
    if TEXTURE_TYPES[tex_type] == "cube":
        # Cube textures are six square faces stacked vertically. Blender gets the first face and
        # projects it onto the box sides, which is only an approximation of MuJoCo's cube mapping.
        pixels = pixels[:model.tex_width[tex_id]]
    filename = f"tex{tex_id}_{name_of(model, mujoco.mjtObj.mjOBJ_TEXTURE, tex_id) or 'unnamed'}.png"
    imageio.imwrite(texture_dir / filename, pixels)
    return {"file": f"textures/{filename}", "type": TEXTURE_TYPES[tex_type]}


def material_info(model, mat_id, textures, texture_dir):
    tex_id = model.mat_texid[mat_id, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)]
    if tex_id >= 0 and tex_id not in textures:
        textures[tex_id] = save_texture(model, tex_id, texture_dir)
    return {
        "name": name_of(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id),
        "texture": textures.get(tex_id),
        "texrepeat": model.mat_texrepeat[mat_id].tolist(),
        "texuniform": bool(model.mat_texuniform[mat_id]),
        "specular": float(model.mat_specular[mat_id]),
        "shininess": float(model.mat_shininess[mat_id]),
        "reflectance": float(model.mat_reflectance[mat_id]),
        "emission": float(model.mat_emission[mat_id]),
    }


def mesh_arrays(model, mesh_id):
    """Vertices (V,3), faces (F,3), per-corner normals (F,3,3) and UVs (F,3,2) or None."""
    vert_start, num_verts = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    face_start, num_faces = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    face_slice = slice(face_start, face_start + num_faces)

    verts = model.mesh_vert[vert_start:vert_start + num_verts]
    faces = model.mesh_face[face_slice]
    normals = model.mesh_normal[model.mesh_normaladr[mesh_id] + model.mesh_facenormal[face_slice]]
    uvs = None
    if model.mesh_texcoordadr[mesh_id] >= 0:
        uvs = model.mesh_texcoord[model.mesh_texcoordadr[mesh_id] + model.mesh_facetexcoord[face_slice]]
    return verts, faces, normals, uvs


def write_scene(env, out, meta):
    """Write scene.json, meshes.npz and textures/ for the env's model. meta goes into scene.json."""
    out = Path(out)
    model = env.sim.model._model
    rendered_groups = env.sim._render_context_offscreen.vopt.geomgroup
    texture_dir = out / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)

    geoms, skipped, materials, textures, mesh_data = [], [], {}, {}, {}
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        reason = skip_reason(model, geom_id, rendered_groups)
        if reason is not None:
            skipped.append({"id": geom_id, "name": name_of(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                            "body": body_id, "reason": reason})
            continue

        mat_id = int(model.geom_matid[geom_id])
        if mat_id >= 0 and mat_id not in materials:
            materials[mat_id] = material_info(model, mat_id, textures, texture_dir)

        mesh_id = int(model.geom_dataid[geom_id]) if model.geom_type[geom_id] == int(mujoco.mjtGeom.mjGEOM_MESH) else -1
        if mesh_id >= 0 and f"mesh{mesh_id}_verts" not in mesh_data:
            verts, faces, normals, uvs = mesh_arrays(model, mesh_id)
            mesh_data[f"mesh{mesh_id}_verts"] = verts
            mesh_data[f"mesh{mesh_id}_faces"] = faces
            mesh_data[f"mesh{mesh_id}_normals"] = normals
            if uvs is not None:
                mesh_data[f"mesh{mesh_id}_uvs"] = uvs

        geoms.append({
            "id": geom_id,
            "name": name_of(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
            "body": body_id,
            "type": GEOM_TYPES[model.geom_type[geom_id]],
            "size": model.geom_size[geom_id].tolist(),
            "pos": model.geom_pos[geom_id].tolist(),
            "quat": model.geom_quat[geom_id].tolist(),
            "rgba": geom_rgba(model, geom_id).tolist(),
            "mesh": mesh_id,
            "material": mat_id,
        })

    # Sites MuJoCo draws (e.g. the stove's red burner disc). Their visibility changes at runtime:
    # LIBERO's set_visualization toggles site_rgba alpha, so every render request carries site_alpha.
    rendered_site_groups = env.sim._render_context_offscreen.vopt.sitegroup
    sites = []
    for site_id in range(model.nsite):
        group = model.site_group[site_id]
        if group < len(rendered_site_groups) and rendered_site_groups[group]:
            assert model.site_type[site_id] in GEOM_TYPES, f"site {site_id} has unsupported type"
            sites.append({
                "id": site_id,
                "name": name_of(model, mujoco.mjtObj.mjOBJ_SITE, site_id),
                "body": int(model.site_bodyid[site_id]),
                "type": GEOM_TYPES[model.site_type[site_id]],
                "size": model.site_size[site_id].tolist(),
                "pos": model.site_pos[site_id].tolist(),
                "quat": model.site_quat[site_id].tolist(),
                "rgba": model.site_rgba[site_id].tolist(),
            })

    used_bodies = sorted({g["body"] for g in geoms} | {s["body"] for s in sites})
    # MuJoCo clips at these distances (relative to the model's extent). The wrist camera often gets
    # within a few cm of objects, so Blender has to clip at exactly the same place.
    znear = float(model.vis.map.znear * model.stat.extent)
    zfar = float(model.vis.map.zfar * model.stat.extent)
    # The key light hangs above the middle of the scene's bodies at dump time. Storing it here keeps
    # the light in the same place for every frame and episode rendered from this dump.
    positions = env.sim.data._data.xpos[used_bodies]
    light_anchor = [float(positions[:, 0].mean()), float(positions[:, 1].mean()), float(positions[:, 2].max())]
    scene = dict(meta)
    scene.update({
        "image_size": IMAGE_SIZE,
        "bodies": [{"id": b, "name": name_of(model, mujoco.mjtObj.mjOBJ_BODY, b)} for b in used_bodies],
        "geoms": geoms,
        "sites": sites,
        "materials": {str(k): v for k, v in materials.items()},
        "light_anchor": light_anchor,
        "cameras": [{"name": name, "fovy": float(model.cam_fovy[model.camera(name).id]),
                     "znear": znear, "zfar": zfar} for name in CAMERAS],
        "geom_body": model.geom_bodyid.tolist(),
        "body_names": [name_of(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)],
        "site_names": [name_of(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in range(model.nsite)],
        "skipped_geoms": skipped,
    })
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    np.savez(out / "meshes.npz", **mesh_data)

    num_meshes = len([key for key in mesh_data if key.endswith("_verts")])
    print(f"exported {len(geoms)} geoms on {len(used_bodies)} bodies, {num_meshes} meshes, "
          f"{len(textures)} textures; skipped {len(skipped)} geoms")
    return scene


def poses(env):
    """World poses Blender needs for the current MuJoCo state, plus which sites are visible."""
    model, data = env.sim.model._model, env.sim.data._data
    camera_ids = [model.camera(name).id for name in CAMERAS]
    return {
        "body_xpos": data.xpos.copy(),
        "body_xquat": data.xquat.copy(),
        "cam_xpos": data.cam_xpos[camera_ids].copy(),
        "cam_xmat": data.cam_xmat[camera_ids].copy(),
        "site_alpha": model.site_rgba[:, 3].copy(),
    }


def rebuild_render_context(env, offsamples):
    """Recreate robosuite's MuJoCo render context with a different number of multisamples.

    Segmentation needs offsamples=0: with multisampling, the id colors of two geoms get averaged at
    object edges and the average decodes to a third, unrelated geom.
    """
    context = env.sim._render_context_offscreen
    env.sim.model._model.vis.quality.offsamples = offsamples
    context.gl_ctx.make_current()
    context.con.free()
    context._set_mujoco_context_and_buffers()


def make_env(suite_name, task_id, image_size, seed=0):
    """A LIBERO env made the way openpi's LIBERO eval makes it. Returns (env, task, init_states)."""
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_file), camera_heights=image_size, camera_widths=image_size)
    env.seed(seed)
    env.reset()
    return env, task, suite.get_task_init_states(task_id)


def num_tasks(suite_name):
    return benchmark.get_benchmark_dict()[suite_name]().n_tasks


def find_task_id(suite_name, demo_file):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task_name = Path(demo_file).name[:-len("_demo.hdf5")]
    for task_id in range(suite.n_tasks):
        if suite.get_task(task_id).name == task_name:
            return task_id
    raise ValueError(f"no task named {task_name} in {suite_name}")


def set_sim_state(env, state):
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    # As LIBERO's own regenerate_obs_from_state: the success check updates object states such as
    # "stove on", and _post_process shows or hides their visualization sites (the stove's red burner disc).
    env.check_success()
    env._post_process()
