"""Render settings from a render config (configs/render/*.yaml), plus the geom-id pass. Runs inside Blender."""
import bpy
import numpy as np

from libero_blender.blender.lighting import set_studio_lighting

ID_PASS_SEED = 0


def apply_config(scene, config, light_anchor):
    """Set lighting and engine for a render config. Cheap to call again; nothing is rebuilt."""
    set_studio_lighting(scene, light_anchor, config["lighting"])
    if config["engine"] == "eevee":
        setup_eevee(scene, config["eevee"])
    else:
        setup_cycles(scene, config["cycles"])


def eevee_engine_name():
    # EEVEE was renamed EEVEE Next in 4.2 and back to EEVEE in 5.0.
    if (4, 2, 0) <= bpy.app.version < (5, 0, 0):
        return "BLENDER_EEVEE_NEXT"
    return "BLENDER_EEVEE"


def use_optix_gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "OPTIX"
    assert any(d.use for d in prefs.devices), "no OptiX device found"


def setup_rgb_png(scene):
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_layers[0].material_override = None


def setup_eevee(scene, settings):
    scene.render.engine = eevee_engine_name()
    scene.render.use_persistent_data = False
    scene.eevee.taa_render_samples = settings["samples"]
    scene.eevee.use_raytracing = settings["raytracing"]
    setup_rgb_png(scene)


def setup_cycles(scene, settings):
    use_optix_gpu()
    scene.render.engine = "CYCLES"
    scene.render.use_persistent_data = settings["persistent_data"]
    scene.cycles.device = "GPU"
    scene.cycles.samples = settings["samples"]
    scene.cycles.use_adaptive_sampling = settings["adaptive_sampling"]
    scene.cycles.seed = settings["seed"]
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = settings["denoiser"]
    scene.cycles.pixel_filter_type = "BLACKMAN_HARRIS"
    scene.cycles.filter_width = 1.5
    scene.cycles.max_bounces = settings["max_bounces"]
    scene.cycles.sample_clamp_indirect = settings["clamp_indirect"]
    scene.cycles.blur_glossy = settings["blur_glossy"]
    setup_rgb_png(scene)


def setup_geom_id_pass(scene):
    """Cycles at one sample per pixel, no filtering, every object drawn as flat emission of its
    object color, which holds the encoded geom id. Written to 32-bit EXR so ids stay exact."""
    use_optix_gpu()
    scene.render.engine = "CYCLES"
    scene.render.use_persistent_data = False
    scene.cycles.device = "GPU"
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.seed = ID_PASS_SEED
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01  # sample at the pixel center, like a rasterizer
    scene.cycles.max_bounces = 0
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Raw"
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.color_mode = "RGBA"

    material = bpy.data.materials.get("geom_id") or make_geom_id_material()
    scene.view_layers[0].material_override = material


def make_geom_id_material():
    """Flat emission of the object color, with back faces see-through.

    MuJoCo culls back faces, so when the wrist camera gets behind or inside a mesh (a microwave door
    it is pushing shut) its segmentation shows what is behind. The RGB renders deliberately keep
    both sides, which is physically right; only this pass copies MuJoCo, so the alignment check
    measures poses and cameras rather than that rasterizer shortcut.
    """
    material = bpy.data.materials.new("geom_id")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    object_info = nodes.new("ShaderNodeObjectInfo")
    emission = nodes.new("ShaderNodeEmission")
    links.new(object_info.outputs["Color"], emission.inputs["Color"])

    geometry = nodes.new("ShaderNodeNewGeometry")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(geometry.outputs["Backfacing"], mix.inputs["Fac"])
    links.new(emission.outputs["Emission"], mix.inputs[1])
    links.new(transparent.outputs["BSDF"], mix.inputs[2])
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def read_geom_ids(exr_path, id_divisor):
    """Decode an id-pass EXR into an (H, W) int array of geom ids, -1 for background, top row first."""
    image = bpy.data.images.load(str(exr_path))
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    red = pixels.reshape(height, width, 4)[:, :, 0]
    geom_ids = np.rint(red * id_divisor).astype(np.int64) - 1
    return geom_ids[::-1]  # Blender stores the bottom row first
