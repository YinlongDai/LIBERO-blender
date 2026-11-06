"""Check that Blender and MuJoCo see the same geometry: Blender's geom-id render vs MuJoCo segmentation, per body.

IoU per visible body = |MuJoCo mask & Blender mask| / |MuJoCo mask | Blender mask|. Colors are never
compared, so lighting does not matter. At 128 px a small object is mostly edge, so a half-pixel shift
alone can pull IoU down; a body therefore also passes if none of its mismatched pixels is more than one
pixel away from the same body in the other image.
"""
import mujoco
import numpy as np
from scipy.ndimage import binary_dilation

IOU_TARGET = 0.95
MIN_PIXELS = 50  # smaller masks are not judged
GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
FLIPS = {
    "none": lambda image: image,
    "vertical": lambda image: image[::-1],
    "horizontal": lambda image: image[:, ::-1],
    "both": lambda image: image[::-1, ::-1],
}


def body_rows(mujoco_bodies, blender_bodies, ignore):
    rows = []
    for body in np.union1d(np.unique(mujoco_bodies), np.unique(blender_bodies)):
        in_mujoco = (mujoco_bodies == body) & ~ignore
        in_blender = (blender_bodies == body) & ~ignore
        if body < 0 or not (in_mujoco | in_blender).any():
            continue
        far = (in_mujoco & ~binary_dilation(in_blender)) | (in_blender & ~binary_dilation(in_mujoco))
        iou = (in_mujoco & in_blender).sum() / (in_mujoco | in_blender).sum()
        rows.append({"body": int(body), "iou": float(iou),
                     "judged": max(in_mujoco.sum(), in_blender.sum()) >= MIN_PIXELS,
                     "passed": iou >= IOU_TARGET or not far.any()})
    return rows


def mean_iou(rows):
    judged = [r["iou"] for r in rows if r["judged"]]
    return float(np.mean(judged)) if judged else 0.0


def compare(seg, geom_ids, scene):
    """MuJoCo segmentation (raw robosuite orientation) vs a Blender geom-id image (top row first).

    All four flips of the Blender image are tried and the best is reported, rather than assuming one.
    """
    geom_body = np.array(scene["geom_body"])
    obj_type, obj_id = seg[:, :, 0], seg[:, :, 1]
    is_geom = obj_type == GEOM
    mujoco_bodies = np.full(obj_id.shape, -1)
    mujoco_bodies[is_geom] = geom_body[obj_id[is_geom]]
    ignore = (obj_type != -1) & ~is_geom  # sites and other non-geom objects
    blender_bodies = np.where(geom_ids >= 0, geom_body[np.maximum(geom_ids, 0)], -1)

    scores = {name: mean_iou(body_rows(mujoco_bodies, flip(blender_bodies), ignore)) for name, flip in FLIPS.items()}
    best = max(scores, key=scores.get)
    rows = body_rows(mujoco_bodies, FLIPS[best](blender_bodies), ignore)
    return {"flip": best, "mean_iou": scores[best], "passed": all(r["passed"] for r in rows if r["judged"])}
