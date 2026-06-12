"""
blender_roundtrip.py — run INSIDE Blender (headless) to reproduce the plugin's
BVH export base-change.

For each input BVH: import it, then re-export with the EXACT flags the plugin
uses in neural_nla_blend.export_strip_as_bvh, writing <stem>.rt.bvh next to a
chosen output dir.  This lets us measure the real import->export distortion.

Run:
  blender --background --python devtools/blender_roundtrip.py -- <in_bvh> [<in_bvh> ...] <out_dir>
"""
import bpy
import importlib.util
import sys
import os
from pathlib import Path


_ADDON_PATH = Path(__file__).resolve().parent.parent / "blendanything_client" / "addon.py"
_spec = importlib.util.spec_from_file_location("neural_nla_blend_rt", str(_ADDON_PATH))
_addon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_addon)

argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []
*in_bvhs, out_dir = argv
os.makedirs(out_dir, exist_ok=True)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for blk in (bpy.data.armatures, bpy.data.actions, bpy.data.objects):
        for d in list(blk):
            try:
                blk.remove(d)
            except Exception:
                pass


for in_bvh in in_bvhs:
    clear_scene()
    stem = os.path.splitext(os.path.basename(in_bvh))[0]
    # Import with matching axis convention (plugin imports default; BVH addon
    # default import is Y-up). Use import defaults so the round-trip reflects
    # exactly what a user sees: import then plugin-export.
    bpy.ops.import_anim.bvh(filepath=in_bvh, axis_forward="-Z", axis_up="Y")
    armature = bpy.context.selected_objects[0]
    action = armature.animation_data.action
    frame_start = int(action.frame_range[0])
    frame_end = int(action.frame_range[1])
    out = os.path.join(out_dir, stem + ".rt.bvh")
    # Match neural_nla_blend.export_strip_as_bvh exactly.
    bpy.ops.export_anim.bvh(
        filepath=out,
        frame_start=frame_start,
        frame_end=frame_end,
        root_transform_only=True,
        rotate_mode="NATIVE",
    )
    source_out = os.path.join(out_dir, stem + ".source.bvh")
    _addon._write_source_bvh_clip(
        in_bvh, source_out, frame_start, frame_end, frame_start
    )
    print(f"[roundtrip] {in_bvh} -> {out}")
    print(f"[source]    {in_bvh} -> {source_out}")

print("[roundtrip] done")
