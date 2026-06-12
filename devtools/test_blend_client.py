"""
test_blend_client.py — exercise the /blend endpoint without Blender.

Posts a reference + target BVH plus the same metadata shape the Blender plugin
sends, then writes the blended BVH result to disk.

Usage:
  python devtools/test_blend_client.py [REF_BVH] [TGT_BVH] [OUT_BVH] [--url URL]

Defaults to the bundled Elephant/Skunk samples and writes /tmp/blend_result.bvh.
"""
import json
import sys
import os

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "samples")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
url = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--url=")),
           "http://localhost:8000")

ref = args[0] if len(args) > 0 else os.path.join(D, "Elephant___walk_327.bvh")
tgt = args[1] if len(args) > 1 else os.path.join(D, "Skunk___Spray_891.bvh")
out = args[2] if len(args) > 2 else "/tmp/blend_result.bvh"


def strip_meta(name, char):
    # Mirrors neural_nla_blend.py strip_meta + strength_to_metadata.
    return {
        "name": name,                # char prefix is how the server infers skeleton
        "action": name,
        "frame_start": 0, "frame_end": 100,
        "action_frame_start": 0, "action_frame_end": 100,
        "repeat": 1.0, "scale": 1.0, "use_reverse": False,
        "extrapolation": "HOLD",
        "strength": {
            "profile": "CONSTANT",
            "frame_start": 0, "frame_end": 100,
            "samples": [0.5] * 101,
            "value": 0.5,
        },
    }


# server infers object_type from the strip 'name' prefix; use the char name.
ref_char = os.path.basename(ref).split("___")[0].split("_")[0]
tgt_char = os.path.basename(tgt).split("___")[0].split("_")[0]

metadata = {
    "reference": strip_meta(ref_char, ref_char),
    "targets":   [strip_meta(tgt_char, tgt_char)],
    "blend_mode": "BLEND",
    "control_mode": "both",
    "output_mode": "POSITIONS_IK",
    "ik_iterations": 150,
}

print(f"POST {url}/blend\n  ref={ref}\n  tgt={tgt}\n  ref_char={ref_char} tgt_char={tgt_char}")

with open(ref, "rb") as fr, open(tgt, "rb") as ft:
    files = [
        ("reference_bvh", ("reference.bvh", fr, "text/plain")),
        ("target_bvh",    ("target_0.bvh", ft, "text/plain")),
    ]
    resp = requests.post(
        url.rstrip("/") + "/blend",
        files=files,
        data={"metadata": json.dumps(metadata)},
        timeout=600,
    )

print("HTTP", resp.status_code)
if resp.status_code != 200:
    print(resp.text[:2000])
    sys.exit(1)

with open(out, "wb") as f:
    f.write(resp.content)
print(f"Wrote {len(resp.content)} bytes -> {out}")
print("Also check BlendAnything/.bvh_cache/ for *_ref_processed / *_blended intermediates.")
