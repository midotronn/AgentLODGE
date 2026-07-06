"""Blender headless renderer for precomputed SMPL-X dance meshes.

Run inside Blender:

    blender -b -noaudio -P blender_render_smplx.py -- \
        --meshes meshes.npz --frames-dir out/ --width 960 --height 960 --samples 64 \
        [--texture skin.png]

``meshes.npz`` holds ``verts`` (L, V, 3) in SMPL-X native (Y-up) coordinates, ``faces``
(F, 3) and optional per-loop ``uv`` (F*3, 2). The body is placed on a ground plane,
lit with a sun + fill lights, framed by a camera fitted to the whole motion, and each
frame is rendered to ``frames_dir/frame_%05d.png`` with EEVEE.
"""

import argparse
import math
import os
import sys

import bpy  # type: ignore
import numpy as np
from mathutils import Vector  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_studio as studio  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--meshes", required=True)
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--texture", default="")
    p.add_argument("--color", default="0.32,0.45,0.55",
                   help="Body base colour r,g,b in 0-1 (ignored if --texture set)")
    return p.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def build_mesh(verts0, faces, uv, material):
    mesh = bpy.data.meshes.new("smplx")
    mesh.from_pydata([tuple(v) for v in verts0.tolist()], [], [tuple(f) for f in faces.tolist()])
    mesh.update()
    if uv is not None:
        uv_layer = mesh.uv_layers.new(name="UVMap")
        uv_layer.data.foreach_set("uv", np.asarray(uv, dtype=np.float32).ravel())
    obj = bpy.data.objects.new("SMPLX", mesh)
    obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)
    # Verts arrive already Z-up (oriented upstream), so no object rotation is needed.
    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj, mesh


def world_bounds(verts_all):
    """World-space bounds; verts are already in Blender Z-up world coordinates."""
    flat = verts_all.reshape(-1, 3)
    return flat.min(axis=0), flat.max(axis=0)


def main():
    args = parse_args()
    data = np.load(args.meshes)
    verts = data["verts"].astype(np.float32)
    faces = data["faces"].astype(np.int64)
    uv = data["uv"] if "uv" in data.files else None

    clear_scene()
    try:
        color = tuple(float(c) for c in args.color.split(","))[:3]
    except Exception:
        color = (0.32, 0.45, 0.55)
    material = studio.make_material("SMPLX_Body", args.texture, color)
    obj, mesh = build_mesh(verts[0], faces, uv, material)

    bmin, bmax = world_bounds(verts)
    cx, cy, floor_z = studio.setup_world_and_ground(bmin, bmax)
    spot = studio.setup_lights(cx, cy, float(bmax[2]))

    # Body size from per-frame extents (not the whole trajectory) keeps the dancer large.
    per_frame_ext = verts.max(axis=1) - verts.min(axis=1)  # (L, 3)
    body_size = float(np.median(per_frame_ext.max(axis=1)))
    centroids = verts.mean(axis=1)  # (L, 3) world coords
    cam, target, offset, target_z, follow_xy = studio.setup_follow_camera(
        centroids, body_size, floor_z
    )
    # The key spot tracks the same target and rides high-front of the dancer so the bright
    # floor pool + vignette move with the character (EDGE-style).
    spot_front, spot_high = studio.attach_follow_spot(
        spot, target, floor_z, body_size, offset
    )
    studio.configure_render(args.width, args.height, args.samples)

    scene = bpy.context.scene
    frames_dir = args.frames_dir.rstrip("/")
    for i in range(verts.shape[0]):
        mesh.vertices.foreach_set("co", verts[i].reshape(-1))
        mesh.update()
        tx, ty = float(follow_xy[i, 0]), float(follow_xy[i, 1])
        target.location = (tx, ty, target_z)
        cam.location = (tx + offset[0], ty + offset[1], target_z + offset[2])
        spot.location = (tx, ty + spot_front, spot_high)
        scene.render.filepath = f"{frames_dir}/frame_{i:05d}.png"
        bpy.ops.render.render(write_still=True)
    print(f"BLENDER_RENDERED {verts.shape[0]} frames -> {frames_dir}")


if __name__ == "__main__":
    main()
