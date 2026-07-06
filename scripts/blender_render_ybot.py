"""Blender headless renderer for the EDGE Mixamo Y-Bot robot character.

EDGE's ``ybot.fbx`` is the segmented Y-Bot robot mesh skinned to an SMPL-named armature
(bones ``m_avg_Pelvis`` ... one per SMPL joint). EDGE drives it by writing SMPL axis-angle
rotations onto those bones via the Autodesk FBX SDK; we reproduce that entirely in Blender
so no FBX SDK is required: import the FBX and pose the ``m_avg_*`` bones from SMPL poses.

Run inside Blender::

    blender -b -noaudio -P blender_render_ybot.py -- \
        --poses poses.npz --ybot ybot.fbx --frames-dir out/ \
        --width 720 --height 720 --samples 32 [--color 0.5,0.5,0.5] [--align-x -90]

``poses.npz`` holds ``poses`` (L, J, 3) SMPL axis-angle (J>=22; hand joints may be zero)
and optional ``trans`` (L, 3). The dancer is centred on the studio floor each frame
(root locked horizontally, feet grounded), lit by the shared EDGE studio.
"""

import argparse
import math
import os
import sys

import bpy  # type: ignore
import numpy as np
from mathutils import Matrix, Quaternion, Vector  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_studio as studio  # noqa: E402

# SMPL joint order (matches EDGE SMPL-to-FBX/SmplObject.py and ax_from_6v output).
JOINT_NAMES = [
    "m_avg_Pelvis", "m_avg_L_Hip", "m_avg_R_Hip", "m_avg_Spine1",
    "m_avg_L_Knee", "m_avg_R_Knee", "m_avg_Spine2", "m_avg_L_Ankle",
    "m_avg_R_Ankle", "m_avg_Spine3", "m_avg_L_Foot", "m_avg_R_Foot",
    "m_avg_Neck", "m_avg_L_Collar", "m_avg_R_Collar", "m_avg_Head",
    "m_avg_L_Shoulder", "m_avg_R_Shoulder", "m_avg_L_Elbow", "m_avg_R_Elbow",
    "m_avg_L_Wrist", "m_avg_R_Wrist", "m_avg_L_Hand", "m_avg_R_Hand",
]
FOOT_BONES = ["m_avg_L_Foot", "m_avg_R_Foot", "m_avg_L_Foot_end", "m_avg_R_Foot_end"]
TARGET_HEIGHT = 1.7  # metres; normalise the robot so the shared studio framing fits.


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True)
    p.add_argument("--ybot", required=True)
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--color", default="0.5,0.5,0.52",
                   help="Robot base colour r,g,b in 0-1")
    p.add_argument("--align-x", type=float, default=0.0,
                   help="Degrees about X aligning SMPL local rotations to the armature frame")
    p.add_argument("--yaw", type=float, default=0.0,
                   help="Extra degrees about Z to face the dancer toward the camera")
    p.add_argument("--stride", type=int, default=1,
                   help="Render every Nth frame (validation only; alignment still uses all)")
    return p.parse_args(argv)


def axis_angle_to_matrix(v):
    ang = float(np.linalg.norm(v))
    if ang < 1e-8:
        return Matrix.Identity(3)
    axis = Vector((float(v[0]) / ang, float(v[1]) / ang, float(v[2]) / ang))
    return Quaternion(axis, ang).to_matrix()


def import_ybot(path):
    bpy.ops.import_scene.fbx(filepath=path)
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError("No armature found in ybot.fbx")
    return arms[0]


def style_robot(arm, color):
    """Grey metallic material on every robot segment; hide non-robot helper meshes.
    Returns the list of visible robot mesh objects (used for accurate grounding)."""
    mat = studio.make_material("Ybot", "", color, metallic=0.85, roughness=0.4)
    robot_meshes = []
    for obj in list(bpy.data.objects):
        if obj.type != "MESH":
            continue
        if obj.name.startswith("Alpha_"):
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            for poly in obj.data.polygons:
                poly.use_smooth = True
            robot_meshes.append(obj)
        else:
            obj.hide_render = True
    return robot_meshes


def lowest_mesh_z(depsgraph, meshes):
    """World-space minimum z over the posed/skinned robot mesh vertices.

    Grounding on the foot *bone* leaves the foot *mesh* (the sole) poking through the
    floor, so we ground on the actual deformed surface instead. ``foreach_get`` + numpy
    keeps this cheap even across the ~100 Y-Bot segments."""
    mz = float("inf")
    for obj in meshes:
        ev = obj.evaluated_get(depsgraph)
        me = ev.data
        n = len(me.vertices)
        if n == 0:
            continue
        co = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        M = np.array(ev.matrix_world)
        world_z = co @ M[:3, :3].T[:, 2] + M[2, 3]
        zmin = float(world_z.min())
        if zmin < mz:
            mz = zmin
    return mz


def rest_rotations(arm):
    """Per-driven-bone armature-space rest rotation quaternion (from bone.matrix_local)."""
    rest = {}
    for name in JOINT_NAMES:
        bone = arm.data.bones.get(name)
        if bone is not None:
            rest[name] = bone.matrix_local.to_3x3().to_quaternion()
    return rest


def normalise_scale(arm):
    """Scale the whole rig so the rest robot is ~TARGET_HEIGHT tall."""
    bpy.context.view_layer.update()
    zs = []
    for bone in arm.data.bones:
        for pt in (bone.head_local, bone.tail_local):
            zs.append((arm.matrix_world @ Vector(pt)).z)
    height = max(zs) - min(zs)
    if height > 1e-6:
        s = TARGET_HEIGHT / height
        arm.scale = (s, s, s)
    bpy.context.view_layer.update()


def main():
    args = parse_args()
    data = np.load(args.poses)
    poses = data["poses"].astype(np.float32)  # (L, J, 3)
    L = poses.shape[0]
    try:
        color = tuple(float(c) for c in args.color.split(","))[:3]
    except Exception:
        color = (0.5, 0.5, 0.52)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm = import_ybot(args.ybot)
    robot_meshes = style_robot(arm, color)
    normalise_scale(arm)

    arm.rotation_mode = "QUATERNION"
    arm.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"

    rest = rest_rotations(arm)
    rest_inv = {k: q.to_matrix().transposed() for k, q in rest.items()}
    rest_mat = {k: q.to_matrix() for k, q in rest.items()}

    def whead(name):
        return arm.matrix_world @ arm.pose.bones[name].head

    def wtail(name):
        return arm.matrix_world @ arm.pose.bones[name].tail

    def apply_pose(i, A, A_inv):
        for j, name in enumerate(JOINT_NAMES):
            if j >= poses.shape[1] or name not in rest:
                continue
            Rj = axis_angle_to_matrix(poses[i, j])
            basis = rest_inv[name] @ A @ Rj @ A_inv @ rest_mat[name]
            arm.pose.bones[name].rotation_quaternion = basis.to_quaternion()

    # The SMPL rotations live in the dance's own world frame (Y-up for FineDance, Z-up for
    # EDGE-derived motion). Rather than guess, pick the fixed alignment A whose posed pelvis
    # stands most consistently upright (bone axis -> +Z) across sampled frames. This keeps
    # the robot vertical every frame while preserving its yaw/facing and limb articulation.
    candidates = [
        ("I", Matrix.Identity(3)),
        ("X+90", Matrix.Rotation(math.radians(90), 3, "X")),
        ("X-90", Matrix.Rotation(math.radians(-90), 3, "X")),
        ("X180", Matrix.Rotation(math.radians(180), 3, "X")),
        ("Y+90", Matrix.Rotation(math.radians(90), 3, "Y")),
        ("Y-90", Matrix.Rotation(math.radians(-90), 3, "Y")),
        ("Z+90", Matrix.Rotation(math.radians(90), 3, "Z")),
        ("Z-90", Matrix.Rotation(math.radians(-90), 3, "Z")),
    ]
    if abs(args.align_x) > 1e-6:  # honour an explicit override first
        candidates.insert(0, ("override", Matrix.Rotation(math.radians(args.align_x), 3, "X")))
    sample = list(range(0, L, max(1, L // 8)))
    best_A, best_score = Matrix.Identity(3), -2.0
    for _label, A in candidates:
        A_inv = A.transposed()
        zsum = 0.0
        for i in sample:
            apply_pose(i, A, A_inv)
            bpy.context.view_layer.update()
            zsum += (wtail("m_avg_Pelvis") - whead("m_avg_Pelvis")).normalized().z
        score = zsum / len(sample)
        if score > best_score + 1e-3:  # strict improvement; ties keep the earlier (identity)
            best_score, best_A = score, A
    A = best_A
    A_inv = A.transposed()

    if abs(args.yaw) > 1e-6:  # optional facing tweak (about the vertical axis)
        arm.rotation_quaternion = Quaternion(Vector((0.0, 0.0, 1.0)), math.radians(args.yaw))
        bpy.context.view_layer.update()

    def pose_frame(i):
        apply_pose(i, A, A_inv)

    # Studio: dancer centred at origin, floor at z=0.
    body_size = TARGET_HEIGHT
    studio.setup_world_and_ground(
        np.array([-1.5, -1.5, 0.0]), np.array([1.5, 1.5, body_size])
    )
    spot = studio.setup_lights(0.0, 0.0, body_size)
    centroids = np.tile(np.array([[0.0, 0.0, 0.55 * body_size]]), (L, 1))
    cam, target, offset, target_z, follow_xy = studio.setup_follow_camera(
        centroids, body_size, 0.0
    )
    spot_front, spot_high = studio.attach_follow_spot(spot, target, 0.0, body_size, offset)
    target.location = (0.0, 0.0, target_z)
    cam.location = (offset[0], offset[1], target_z + offset[2])
    spot.location = (0.0, spot_front, spot_high)
    studio.configure_render(args.width, args.height, args.samples)

    scene = bpy.context.scene
    frames_dir = args.frames_dir.rstrip("/")
    stride = max(1, args.stride)
    for i in range(0, L, stride):
        pose_frame(i)
        # Centre the dancer horizontally on the pelvis...
        arm.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        pelvis = whead("m_avg_Pelvis")
        arm.location = (-pelvis.x, -pelvis.y, 0.0)
        bpy.context.view_layer.update()
        # ...then ground the lowest posed *mesh* vertex (foot sole) at z=0 so the robot
        # rests on the floor instead of sinking its feet through it.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        mz = lowest_mesh_z(depsgraph, robot_meshes)
        if mz != float("inf"):
            arm.location = (arm.location.x, arm.location.y, -mz)
            bpy.context.view_layer.update()
        scene.render.filepath = f"{frames_dir}/frame_{i:05d}.png"
        bpy.ops.render.render(write_still=True)
    print(f"BLENDER_RENDERED {L} frames -> {frames_dir}")


if __name__ == "__main__":
    main()
