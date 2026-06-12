"""
diag_bvh_npy.py — Diagnose the BVH->npy conversion used by the Blender plugin.

We have, for every clip in the dataset:
  - motions/<name>.npy   : ground-truth (T-1, J, 13) features the model trained on
  - bvhs/<name>.bvh      : the *processed* (HML-aligned) BVH saved at build time

Experiment A (no Blender needed):
  Feed the processed BVH through the EVAL path (bvh_to_features: NO process_anim)
  and through the PLUGIN path (blendanything_server.bvh_pipeline.bvh_to_npy)
  None alignment params).  Compare both against the ground-truth npy.

  Expectation:
    eval path   ~= npy          (small numerical error)   -> extraction is sound
    plugin path != npy          (double-transform error)  -> the bug

Per-channel L1 error is reported so we can localise the damage:
  [0:3]  rifke position   [3:9] cont6d rotation   [9:12] local vel   [12] foot
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PKG  = os.path.join(_ROOT, "neural_motion_blending")
for p in (_PKG, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import BVH
from data_loaders.truebones.truebones_utils.motion_process import (
    get_bvh_cont6d_params, get_foot_contact, get_rifke, get_motion_features,
)
from data_loaders.truebones.truebones_utils.param_utils import FOOT_CONTACT_VEL_THRESH
from data_loaders.truebones.truebones_utils.motion_process import get_motion
from utils.bvh_io import bvh_load_safe
from blendanything_server.bvh_pipeline import _reorder_anim, bvh_to_npy

DATA = os.path.join(_PKG, "dataset/truebones/zoo/truebones_processed")
COND = np.load(os.path.join(_ROOT, "data", "truebones_cond.npy"), allow_pickle=True).item()


def eval_path_features(bvh_path, object_type):
    """Replica of fid_truebones_blend.bvh_to_features — NO process_anim."""
    ocd = COND[object_type]
    anim, bvh_names, _ = bvh_load_safe(bvh_path)
    anim = _reorder_anim(anim, list(bvh_names), list(ocd["joints_names"]), ocd=ocd)
    c6d, _, _, r_rot, gpos = get_bvh_cont6d_params(anim, object_type, face_joints=ocd["face_joints"])
    foot = get_foot_contact(gpos, ocd["foot_indices"], FOOT_CONTACT_VEL_THRESH)
    pos  = get_rifke(gpos, r_rot)
    lvel = np.repeat(r_rot[1:, None], gpos.shape[1], axis=1) * (gpos[1:] - gpos[:-1])
    feats, _ = get_motion_features(pos, c6d, foot, lvel, len(ocd["offsets"]))
    return feats


def plugin_path_stored_params(bvh_bytes, object_cond):
    """Same as bvh_to_npy but passing the STORED cond_dict alignment params
    instead of None — the proposed fix."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
        f.write(bvh_bytes); tmp = f.name
    try:
        anim, bvh_names, _ = bvh_load_safe(tmp)
    finally:
        os.unlink(tmp)
    anim = _reorder_anim(anim, list(bvh_names), list(object_cond["joints_names"]), ocd=object_cond)
    feats, _, _, _ = get_motion(
        bvh_path=anim,
        foot_contact_vel_thresh=FOOT_CONTACT_VEL_THRESH,
        object_type=object_cond["object_type"],
        max_joints=len(object_cond["offsets"]),
        root_pose_init_xz=object_cond["root_pose_init_xz"],
        scale_factor=object_cond["scale_factor"],
        ground_height=object_cond["ground_height"],
        offsets=object_cond["offsets"],
        foot_indices=object_cond["foot_indices"],
        tpos_rots=object_cond["tpos_rots"],
        squared_positions_error={},
        face_joints=object_cond.get("face_joints"),
    )
    return feats


def chan_l1(a, b):
    d = np.abs(a - b)
    segs = {"pos[0:3]": (0, 3), "rot[3:9]": (3, 9), "vel[9:12]": (9, 12), "foot[12]": (12, 13)}
    return {k: float(d[..., s:e].mean()) for k, (s, e) in segs.items()}, float(d.mean())


def object_type_from_name(name):
    # dataset npy stem looks like "Camel___ScrapeHoof_180" ; char is before first "_"
    cand = name.split("___")[0].split("_")[0]
    if cand in COND:
        return cand
    for k in COND:
        if name.startswith(k):
            return k
    return None


def main():
    motions_dir = os.path.join(DATA, "motions")
    bvhs_dir    = os.path.join(DATA, "bvhs")
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(motions_dir) if f.endswith(".npy"))

    # pick a spread of distinct characters
    picked, seen = [], set()
    for n in names:
        ot = object_type_from_name(n)
        if ot and ot not in seen:
            picked.append((n, ot)); seen.add(ot)
        if len(picked) >= 8:
            break

    hdr = f"{'clip':<30}{'type':<12}{'eval':>9}{'plugin(None)':>14}{'plugin(stored)':>16}"
    print(hdr)
    print("-" * len(hdr))
    eval_rows, plug_rows, stored_rows = [], [], []
    for name, ot in picked:
        npy_path = os.path.join(motions_dir, name + ".npy")
        bvh_path = os.path.join(bvhs_dir, name + ".bvh")
        if not os.path.exists(bvh_path):
            continue
        gt = np.load(npy_path)                       # (T-1, J, 13)

        try:
            ev = eval_path_features(bvh_path, ot)
        except Exception as e:
            ev = None; print(f"  eval fail {name}: {e}")

        with open(bvh_path, "rb") as f:
            bvh_bytes = f.read()
        try:
            pl = bvh_to_npy(bvh_bytes, COND[ot])     # plugin path (None params)
        except Exception as e:
            pl = None; print(f"  plugin fail {name}: {e}")

        try:
            st = plugin_path_stored_params(bvh_bytes, COND[ot])  # proposed fix
        except Exception as e:
            st = None; print(f"  stored fail {name}: {e}")

        def align(x):
            if x is None:
                return None
            T = min(gt.shape[0], x.shape[0])
            J = min(gt.shape[1], x.shape[1])
            return gt[:T, :J], x[:T, :J]

        et = pt = stt = float("nan")
        if ev is not None:
            g, e = align(ev); ech, et = chan_l1(g, e); eval_rows.append(ech)
        if pl is not None:
            g, p = align(pl); pch, pt = chan_l1(g, p); plug_rows.append(pch)
        if st is not None:
            g, s = align(st); sch, stt = chan_l1(g, s); stored_rows.append(sch)
        print(f"{name:<30}{ot:<12}{et:>9.4f}{pt:>14.4f}{stt:>16.4f}")

    def avg(rows):
        if not rows:
            return {}
        return {k: np.mean([r[k] for r in rows]) for k in rows[0]}

    print("\nMean per-channel L1 vs ground-truth npy:")
    print(f"  {'channel':<12}{'EVAL':>10}{'PLUGIN(None)':>14}{'PLUGIN(stored)':>16}")
    ea, pa, sa = avg(eval_rows), avg(plug_rows), avg(stored_rows)
    for k in ["pos[0:3]", "rot[3:9]", "vel[9:12]", "foot[12]"]:
        print(f"  {k:<12}{ea.get(k, float('nan')):>10.4f}"
              f"{pa.get(k, float('nan')):>14.4f}{sa.get(k, float('nan')):>16.4f}")


if __name__ == "__main__":
    main()
