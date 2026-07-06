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
    p.add_argument("--color", default="0.32,0.45,0.55",
                   help="Body base colour r,g,b in 0-1 (ignored if --texture set)")
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
def make_material(texture_path: str, color):
    """Metallic body material (EDGE-style). ``color`` is an (r,g,b) base colour."""
    mat = bpy.data.materials.new("SMPLX_Body")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Metallic"].default_value = 0.9
    bsdf.inputs["Roughness"].default_value = 0.35
    if texture_path:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(texture_path)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
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
    """Studio cyclorama: a seamless curved backdrop (floor blends up into a back wall) lit
    to fade from a bright pool into darker grey, with a dim world so the spotlight reads."""
    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.045, 0.045, 0.05, 1.0)
        bg.inputs[1].default_value = 1.0
    bpy.context.scene.world = world

    cx = 0.5 * (bmin[0] + bmax[0])
    cy = 0.5 * (bmin[1] + bmax[1])
    floor_z = float(bmin[2])

    width, front, back, height, radius = 60.0, 22.0, 14.0, 22.0, 6.0
    # Profile in (y, z), front -> back along the floor, then a quarter-circle up into a wall.
    prof = [(front, 0.0), (-back + radius, 0.0)]
    ccy = -back + radius
    for a in range(1, 13):
        ang = (math.pi / 2) * (a / 12)
        prof.append((ccy - radius * math.sin(ang), radius * (1 - math.cos(ang))))
    prof.append((-back, height))

    x0, x1 = cx - width / 2, cx + width / 2
    verts, faces = [], []
    n = len(prof)
    for (py, pz) in prof:
        verts.append((x0, cy + py, floor_z + pz))
        verts.append((x1, cy + py, floor_z + pz))
    for i in range(n - 1):
        a0, a1 = 2 * i, 2 * i + 1
        b0, b1 = 2 * (i + 1), 2 * (i + 1) + 1
        faces.append((a0, b0, b1, a1))

    mesh = bpy.data.meshes.new("cyclo")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for poly in mesh.polygons:
        poly.use_smooth = True
    cyc = bpy.data.objects.new("Cyclorama", mesh)
    bpy.context.collection.objects.link(cyc)
    gmat = bpy.data.materials.new("Studio")
    gmat.use_nodes = True
    gb = gmat.node_tree.nodes.get("Principled BSDF")
    if gb:
        gb.inputs["Base Color"].default_value = (0.55, 0.55, 0.57, 1.0)
        gb.inputs["Roughness"].default_value = 0.85
    cyc.data.materials.append(gmat)
    return cx, cy, floor_z


def setup_lights(cx, cy, top_z):
    """A key spot from high-front makes the bright floor pool + vignette; a soft area fill
    keeps the character readable. Returns the spot so it can follow the dancer."""
    spot = bpy.data.objects.new("Spot", bpy.data.lights.new("Spot", "SPOT"))
    spot.data.energy = 6000.0
    spot.data.spot_size = math.radians(60.0)
    spot.data.spot_blend = 0.5
    spot.data.shadow_soft_size = 1.2
    bpy.context.collection.objects.link(spot)

    fill = bpy.data.objects.new("Fill", bpy.data.lights.new("Fill", "AREA"))
    fill.data.energy = 120.0
    fill.data.size = 12.0
    fill.location = (cx - 4.0, cy - 6.0, top_z + 2.0)
    bpy.context.collection.objects.link(fill)
    return spot
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
    offset = np.array([dist * 0.35, -dist, body_size * 0.30])

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
    try:
        color = tuple(float(c) for c in args.color.split(","))[:3]
    except Exception:
        color = (0.32, 0.45, 0.55)
    material = make_material(args.texture, color)
    obj, mesh = build_mesh(verts[0], faces, uv, material)

    bmin, bmax = world_bounds(verts)
    cx, cy, floor_z = setup_world_and_ground(bmin, bmax)
    spot = setup_lights(cx, cy, float(bmax[2]))

    # Body size from per-frame extents (not the whole trajectory) keeps the dancer large.
    per_frame_ext = verts.max(axis=1) - verts.min(axis=1)  # (L, 3)
    body_size = float(np.median(per_frame_ext.max(axis=1)))
    centroids = verts.mean(axis=1)  # (L, 3) world coords
    cam, target, offset, target_z, follow_xy = setup_follow_camera(
        centroids, body_size, floor_z
    )
    # The key spot tracks the same target and rides high-front of the dancer so the bright
    # floor pool + vignette move with the character (EDGE-style).
    tc = spot.constraints.new("TRACK_TO")
    tc.target = target
    tc.track_axis = "TRACK_NEGATIVE_Z"
    tc.up_axis = "UP_Y"
    spot_high = float(floor_z) + body_size * 4.0
    spot_front = body_size * 2.2
    configure_render(args.width, args.height, args.samples)

    scene = bpy.context.scene
    frames_dir = args.frames_dir.rstrip("/")
    for i in range(verts.shape[0]):
        mesh.vertices.foreach_set("co", verts[i].reshape(-1))
        mesh.update()
        tx, ty = float(follow_xy[i, 0]), float(follow_xy[i, 1])
        target.location = (tx, ty, target_z)
        cam.location = (tx + offset[0], ty + offset[1], target_z + offset[2])
        front_sign = -1.0 if offset[1] < 0 else 1.0
        spot.location = (tx, ty + front_sign * spot_front, spot_high)
        scene.render.filepath = f"{frames_dir}/frame_{i:05d}.png"
        bpy.ops.render.render(write_still=True)
    print(f"BLENDER_RENDERED {verts.shape[0]} frames -> {frames_dir}")


if __name__ == "__main__":
    main()
