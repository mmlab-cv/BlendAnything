"""
make_user_space_bvh.py — synthesize a 'user-authored' BVH from a dataset
(model-space) processed BVH, for end-to-end Blender validation.

Model space is Y-up / Z-forward; the user authoring contract is Z-up / Y-forward.
We apply model→user = -90° about X to the root channel so the saved BVH is what
a user would have in their Z-up Blender scene.

NOTE: the dataset BVH's rotations are tpos-relative (rest != zero), which differs
from the user contract (zero == T-pose).  This test therefore validates the
Blender round-trip survival + axis handling + rotation extraction, not the
absolute rest-pose offset.

Usage:
  python devtools/make_user_space_bvh.py <in_model_bvh> <out_user_bvh>
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(os.path.dirname(_HERE), "neural_motion_blending")
for p in (_PKG, os.path.dirname(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import BVH
from Animation import Animation
from Quaternions import Quaternions
from utils.bvh_io import bvh_load_safe

MODEL_TO_USER = Quaternions.from_euler(np.array([-np.pi / 2, 0.0, 0.0]), order="xyz")


def main():
    inp, out = sys.argv[1], sys.argv[2]
    a, names, ft = bvh_load_safe(inp)
    T = a.rotations.shape[0]
    q = MODEL_TO_USER.repeat(T, axis=0)
    nr = a.rotations.copy()
    nr[:, 0] = q * nr[:, 0]
    npos = a.positions.copy()
    npos[:, 0] = q * npos[:, 0]
    a2 = Animation(nr, npos, a.orients.copy(), a.offsets.copy(), a.parents.copy())
    BVH.save(out, a2, names=list(names), frametime=ft)
    print(f"[make_user_space] {inp} -> {out}")


if __name__ == "__main__":
    main()
