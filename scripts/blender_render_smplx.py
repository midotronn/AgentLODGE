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
import sys

import bpy  # type: ignore
import numpy as np
from mathutils import Vector  # type: ignore


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--meshes", required=True)
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--texture", default="")
    return p.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_material(texture_path: str):
    """Clean matte body material (EDGE-style light grey). A texture is used only if
    explicitly supplied; the default look is a uniform soft matte character."""
    mat = bpy.data.materials.new("SMPLX_Body")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.62
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.35
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.10
    if texture_path:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(texture_path)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (0.72, 0.71, 0.70, 1.0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


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


def setup_world_and_ground(bmin, bmax):
    # soft, bright studio backdrop
    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.90, 0.91, 0.93, 1.0)
        bg.inputs[1].default_value = 0.9
    bpy.context.scene.world = world

    cx = 0.5 * (bmin[0] + bmax[0])
    cy = 0.5 * (bmin[1] + bmax[1])
    floor_z = float(bmin[2])
    bpy.ops.mesh.primitive_plane_add(size=60.0, location=(cx, cy, floor_z))
    ground = bpy.context.active_object
    gmat = bpy.data.materials.new("Ground")
    gmat.use_nodes = True
    gbsdf = gmat.node_tree.nodes.get("Principled BSDF")
    if gbsdf:
        gbsdf.inputs["Base Color"].default_value = (0.90, 0.90, 0.92, 1.0)
        gbsdf.inputs["Roughness"].default_value = 0.75
        if "Specular IOR Level" in gbsdf.inputs:
            gbsdf.inputs["Specular IOR Level"].default_value = 0.2
    ground.data.materials.append(gmat)
    return cx, cy, floor_z


def setup_lights(cx, cy, top_z):
    # Soft studio setup: key sun + two fills for even, EDGE-like lighting.
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.0
    sun.data.angle = math.radians(6.0)  # soft shadows
    sun.rotation_euler = (math.radians(48.0), math.radians(12.0), math.radians(35.0))
    bpy.context.collection.objects.link(sun)

    key = bpy.data.objects.new("Key", bpy.data.lights.new("Key", "AREA"))
    key.data.energy = 600.0
    key.data.size = 8.0
    key.location = (cx + 4.0, cy - 5.0, top_z + 3.0)
    bpy.context.collection.objects.link(key)

    fill = bpy.data.objects.new("Fill", bpy.data.lights.new("Fill", "AREA"))
    fill.data.energy = 250.0
    fill.data.size = 10.0
    fill.location = (cx - 5.0, cy - 3.0, top_z + 1.5)
    bpy.context.collection.objects.link(fill)
def _smooth(xy: np.ndarray, k: int = 11) -> np.ndarray:
    """Moving-average smooth an (L, 2) path so the follow camera glides, not jitters."""
    if xy.shape[0] < 3:
        return xy
    k = min(k, xy.shape[0] | 1)
    pad = k // 2
    padded = np.pad(xy, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(k) / k
    out = np.stack([np.convolve(padded[:, d], kernel, mode="valid") for d in range(2)], axis=1)
    return out[: xy.shape[0]]


def setup_follow_camera(centroids, body_size, floor_z):
    """Camera that follows the dancer, keeping them large and centered.

    Framing is sized to the body (not the whole trajectory), and the look-at target
    tracks the smoothed horizontal centroid at a fixed mid-body height, so the dancer
    stays big and steady as it travels across the floor.
    """
    dist = max(body_size, 1.0) * 2.6
    target_z = floor_z + 0.55 * body_size
    offset = np.array([dist * 0.55, -dist, body_size * 0.45])

    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 50.0
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    target = bpy.data.objects.new("Target", None)
    bpy.context.collection.objects.link(target)
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    bpy.context.scene.camera = cam

    follow_xy = _smooth(centroids[:, :2])
    return cam, target, offset, target_z, follow_xy


def configure_render(width, height, samples):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    ee = scene.eevee
    try:
        ee.taa_render_samples = samples
        ee.use_shadows = True
        ee.use_raytracing = True
    except Exception:
        pass


def main():
    args = parse_args()
    data = np.load(args.meshes)
    verts = data["verts"].astype(np.float32)
    faces = data["faces"].astype(np.int64)
    uv = data["uv"] if "uv" in data.files else None

    clear_scene()
    material = make_material(args.texture)
    obj, mesh = build_mesh(verts[0], faces, uv, material)

    bmin, bmax = world_bounds(verts)
    cx, cy, floor_z = setup_world_and_ground(bmin, bmax)
    setup_lights(cx, cy, float(bmax[2]))

    # Body size from per-frame extents (not the whole trajectory) keeps the dancer large.
    per_frame_ext = verts.max(axis=1) - verts.min(axis=1)  # (L, 3)
    body_size = float(np.median(per_frame_ext.max(axis=1)))
    centroids = verts.mean(axis=1)  # (L, 3) world coords
    cam, target, offset, target_z, follow_xy = setup_follow_camera(
        centroids, body_size, floor_z
    )
    configure_render(args.width, args.height, args.samples)

    scene = bpy.context.scene
    frames_dir = args.frames_dir.rstrip("/")
    for i in range(verts.shape[0]):
        mesh.vertices.foreach_set("co", verts[i].reshape(-1))
        mesh.update()
        tx, ty = float(follow_xy[i, 0]), float(follow_xy[i, 1])
        target.location = (tx, ty, target_z)
        cam.location = (tx + offset[0], ty + offset[1], target_z + offset[2])
        scene.render.filepath = f"{frames_dir}/frame_{i:05d}.png"
        bpy.ops.render.render(write_still=True)
    print(f"BLENDER_RENDERED {verts.shape[0]} frames -> {frames_dir}")


if __name__ == "__main__":
    main()
