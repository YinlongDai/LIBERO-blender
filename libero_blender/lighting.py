"""Random lighting per episode: which light is coloured and in what colour, where the key light is, how the
ambient (HDRI) fill light is turned, and overall brightness.

Sampled in the env process and applied by Blender (libero_blender/blender/lighting.py), with ranges from
configs/lighting/*.yaml. Colours are normalised to luminance 1 and the key
light power is scaled with distance, so colour and position do not change how bright the scene is;
only `brightness` does, scaling the key light and the fill light together.
"""
import colorsys

import numpy as np

LUMINANCE = np.array([0.2126, 0.7152, 0.0722])  # Rec. 709, Blender's scene-linear space
WHITE = [1.0, 1.0, 1.0]
COLOURED = ["key", "fill", "both"]


def color(hue, saturation):
    rgb = np.array(colorsys.hsv_to_rgb(hue, saturation, 1.0))
    return (rgb / (rgb @ LUMINANCE)).round(4).tolist()


def key_light_offset(azimuth_deg, elevation_deg, distance_m):
    """Key light position relative to the scene centre (x, y, z), from angles around and above it."""
    azimuth, elevation = np.radians(azimuth_deg), np.radians(elevation_deg)
    offset = distance_m * np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth),
                                    np.sin(elevation)])
    return offset.round(4).tolist()


def sample_lighting(rng, ranges):
    """One lighting sample: {coloured, key_light_color, ambient_color, key_light_offset_m, ambient_rotation_deg,
    brightness}."""
    coloured = COLOURED[rng.integers(len(COLOURED))]
    tint = color(rng.uniform(0.0, 1.0), rng.uniform(*ranges["saturation"]))
    offset = key_light_offset(rng.uniform(0.0, 360.0), rng.uniform(*ranges["key_light_elevation_deg"]),
                              rng.uniform(*ranges["key_light_distance_m"]))
    return {
        "coloured": coloured,
        "key_light_color": tint if coloured in ("key", "both") else WHITE,
        "ambient_color": tint if coloured in ("fill", "both") else WHITE,
        "key_light_offset_m": offset,
        "ambient_rotation_deg": round(rng.uniform(*ranges["ambient_rotation_deg"]), 2),
        "brightness": round(rng.uniform(*ranges["brightness"]), 4),
    }
