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
    mat = bpy.data.materials.new("SMPLX_Skin")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.55
    # subsurface for a soft skin look (Blender 4.x uses a weight input)
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.12
    if texture_path:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(texture_path)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (0.62, 0.68, 0.80, 1.0)
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
    # soft sky
    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.92, 0.93, 0.96, 1.0)
        bg.inputs[1].default_value = 0.6
    bpy.context.scene.world = world

    cx = 0.5 * (bmin[0] + bmax[0])
    cy = 0.5 * (bmin[1] + bmax[1])
    floor_z = float(bmin[2])
    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(cx, cy, floor_z))
    ground = bpy.context.active_object
    gmat = bpy.data.materials.new("Ground")
    gmat.use_nodes = True
    gbsdf = gmat.node_tree.nodes.get("Principled BSDF")
    if gbsdf:
        gbsdf.inputs["Base Color"].default_value = (0.80, 0.80, 0.83, 1.0)
        gbsdf.inputs["Roughness"].default_value = 0.9
    ground.data.materials.append(gmat)
    return cx, cy, floor_z


def setup_lights(cx, cy, top_z):
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 4.0
    sun.data.angle = math.radians(3.0)  # soft shadows
    sun.rotation_euler = (math.radians(50.0), math.radians(15.0), math.radians(40.0))
    bpy.context.collection.objects.link(sun)

    fill = bpy.data.objects.new("Fill", bpy.data.lights.new("Fill", "AREA"))
    fill.data.energy = 300.0
    fill.data.size = 6.0
    fill.location = (cx - 4.0, cy - 4.0, top_z + 2.0)
    bpy.context.collection.objects.link(fill)


def setup_camera(bmin, bmax):
    cx = 0.5 * (bmin[0] + bmax[0])
    cz = 0.5 * (bmin[2] + bmax[2])
    depth_c = 0.5 * (bmin[1] + bmax[1])
    span = max(bmax[0] - bmin[0], bmax[2] - bmin[2], bmax[1] - bmin[1], 1.0)
    dist = span * 2.4

    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 50.0
    cam = bpy.data.objects.new("Cam", cam_data)
    cam.location = (cx + dist * 0.55, depth_c - dist, cz + span * 0.35)
    bpy.context.collection.objects.link(cam)

    target = bpy.data.objects.new("Target", None)
    target.location = (cx, depth_c, cz)
    bpy.context.collection.objects.link(target)
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    bpy.context.scene.camera = cam


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
    setup_camera(bmin, bmax)
    configure_render(args.width, args.height, args.samples)

    scene = bpy.context.scene
    n_verts = verts.shape[1]
    frames_dir = args.frames_dir.rstrip("/")
    for i in range(verts.shape[0]):
        mesh.vertices.foreach_set("co", verts[i].reshape(-1))
        mesh.update()
        scene.render.filepath = f"{frames_dir}/frame_{i:05d}.png"
        bpy.ops.render.render(write_still=True)
    print(f"BLENDER_RENDERED {verts.shape[0]} frames -> {frames_dir}")


if __name__ == "__main__":
    main()
