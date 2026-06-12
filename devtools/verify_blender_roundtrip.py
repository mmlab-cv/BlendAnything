"""Run the real Blender round trip and verify model-space features.

Defaults to the Elephant reference and Skunk target used by the plugin test.

Usage:
    python devtools/verify_blender_roundtrip.py /path/to/blender
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PKG = ROOT / "neural_motion_blending"
sys.path[:0] = [str(PKG), str(ROOT)]

import BVH
from blendanything_server.bvh_pipeline import bvh_to_npy

CASES = [
    ("Elephant", ROOT / "samples/Elephant___walk_327.bvh"),
    ("Skunk", ROOT / "samples/Skunk___Spray_891.bvh"),
]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_blender_roundtrip.py /path/to/blender")
    blender = Path(sys.argv[1]).expanduser().resolve()
    if not blender.is_file():
        raise SystemExit(f"Blender executable not found: {blender}")

    cond = np.load(ROOT / "data/truebones_cond.npy", allow_pickle=True).item()

    with tempfile.TemporaryDirectory(prefix="blendanything_blender_rt_") as tmp:
        command = [
            str(blender), "--background",
            "--python", str(HERE / "blender_roundtrip.py"), "--",
            *(str(path) for _, path in CASES), tmp,
        ]
        subprocess.run(command, check=True)

        failed = False
        for object_type, source in CASES:
            blender_output = Path(tmp) / f"{source.stem}.rt.bvh"
            source_output = Path(tmp) / f"{source.stem}.source.bvh"
            _, blender_names, _ = BVH.load(str(blender_output))
            expected_names = list(cond[object_type]["joints_names"])
            missing = [name for name in expected_names if name not in blender_names]
            extra = [name for name in blender_names if name not in expected_names]

            truth = np.load(
                PKG / "dataset/truebones/zoo/truebones_processed/motions"
                / f"{source.stem}.npy"
            )
            blender_features = bvh_to_npy(
                blender_output.read_bytes(),
                cond[object_type],
                coordinate_space="BLENDER_Z_UP",
            )
            source_features = bvh_to_npy(
                source_output.read_bytes(),
                cond[object_type],
                coordinate_space="MODEL_Y_UP",
            )
            if blender_features is None or source_features is None:
                raise RuntimeError(f"Feature conversion failed for {object_type}")

            frames = min(len(source_features), len(truth))
            source_error = np.abs(source_features[:frames] - truth[:frames])
            source_mean = float(source_error.mean())
            source_max = float(source_error.max())
            ok = source_mean < 1e-4
            failed |= not ok

            blender_frames = min(len(blender_features), len(truth))
            blender_error = np.abs(
                blender_features[:blender_frames] - truth[:blender_frames]
            )
            print(
                f"{object_type}: Blender joints={len(blender_names)}, "
                f"cond joints={len(expected_names)}, missing={missing}, extra={extra}"
            )
            print(
                f"  lossy Blender export mean={float(blender_error.mean()):.8g}, "
                f"max={float(blender_error.max()):.8g}"
            )
            print(
                f"  source payload mean={source_mean:.8g}, max={source_max:.8g} "
                f"({'PASS' if ok else 'FAIL'})"
            )

        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
