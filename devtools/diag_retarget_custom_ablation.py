"""Compare known and custom-source retargeting with controlled statistics."""

import argparse
import copy
import json
import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blendanything_server.bvh_pipeline import build_user_conditioning, bvh_to_npy
from blendanything_server.model_bridge import load_pipeline, retarget_single


def _metrics(a: np.ndarray, b: np.ndarray) -> dict:
    frames = min(len(a), len(b))
    diff = np.asarray(a[:frames], dtype=np.float64) - np.asarray(
        b[:frames], dtype=np.float64
    )
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def _print_metrics(label: str, a: np.ndarray, b: np.ndarray) -> None:
    values = _metrics(a, b)
    print(
        f"{label:<34} MAE={values['mae']:.6f} "
        f"RMSE={values['rmse']:.6f} MAX={values['max_abs']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_bvh")
    parser.add_argument("--source", default="Tyranno")
    parser.add_argument("--destination", default="Ostrich")
    parser.add_argument(
        "--model",
        default=os.path.join(
            REPO_ROOT,
            "neural_motion_blending",
            "save",
            "modiffae_truebone_all_globpool",
            "model000449998.pt",
        ),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm")
    parser.add_argument(
        "--output-mode",
        choices=("POSITIONS_IK", "ROTATIONS"),
        default="POSITIONS_IK",
    )
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, ".diagnostics"))
    args = parser.parse_args()

    catalog = np.load(
        os.path.join(REPO_ROOT, "data", "truebones_cond.npy"),
        allow_pickle=True,
    ).item()
    known_source = catalog[args.source]
    destination = catalog[args.destination]
    training_catalog = dict(catalog)
    training_catalog.pop(args.source)

    with open(
        os.path.join(
            REPO_ROOT,
            "blendanything_client",
            "data",
            "truebones_skeletons.json",
        ),
        "r",
    ) as handle:
        face_joints = json.load(handle)["profiles"][args.source]["face_joints"]
    with open(args.source_bvh, "rb") as handle:
        source_bytes = handle.read()

    custom_estimated = build_user_conditioning(
        source_bytes,
        object_type=f"user::{args.source}_estimated",
        coordinate_space="MODEL_Y_UP",
        face_joint_names=face_joints,
        training_catalog=training_catalog,
    )
    custom_real_stats = copy.deepcopy(custom_estimated)
    custom_real_stats["object_type"] = f"user::{args.source}_real_stats"
    custom_real_stats["mean"] = np.asarray(known_source["mean"]).copy()
    custom_real_stats["std"] = np.asarray(known_source["std"]).copy()

    print("Loading model...")
    pipeline = load_pipeline(
        model_path=args.model,
        device=args.device,
        dataset="truebones",
        cond_path=os.path.join(REPO_ROOT, "data", "truebones_cond.npy"),
        sampler=args.sampler,
        seed=args.seed,
    )
    estimated_key = custom_estimated["object_type"]
    real_key = custom_real_stats["object_type"]
    pipeline.cond_dict[estimated_key] = custom_estimated
    pipeline.cond_dict[real_key] = custom_real_stats

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "cache")

    def run(label: str, source_type: str) -> bytes:
        print(f"\nRunning {label}...")
        result = retarget_single(
            pipeline,
            source_bytes,
            {
                "name": os.path.basename(args.source_bvh),
                "object_type": source_type,
                "coordinate_space": "MODEL_Y_UP",
                "face_joints": face_joints,
            },
            args.destination,
            npy_cache_dir=cache_dir,
            output_mode=args.output_mode,
            ddim_inversion=False,
            transition_slerp=False,
            seed=args.seed,
        )
        path = os.path.join(args.output_dir, f"{label}.bvh")
        with open(path, "wb") as handle:
            handle.write(result)
        return result

    known_result = run("known", args.source)
    estimated_result = run("custom_estimated", estimated_key)
    real_stats_result = run("custom_real_stats", real_key)

    def features(data: bytes) -> np.ndarray:
        result = bvh_to_npy(
            data,
            destination,
            coordinate_space="MODEL_Y_UP",
        )
        if result is None:
            raise RuntimeError("Could not convert retarget result to features.")
        return result

    known_features = features(known_result)
    estimated_features = features(estimated_result)
    real_stats_features = features(real_stats_result)

    print("\nRetarget output comparison against known-source ground truth")
    _print_metrics("custom estimated statistics", estimated_features, known_features)
    _print_metrics("custom copied real statistics", real_stats_features, known_features)
    _print_metrics("estimated vs copied-real", estimated_features, real_stats_features)

    print("\nSource conditioning comparison")
    for field in (
        "parents",
        "offsets",
        "tpos_first_frame",
        "joint_relations",
        "joints_graph_dist",
        "mean",
        "std",
    ):
        _print_metrics(
            f"estimated {field}",
            np.asarray(custom_estimated[field]),
            np.asarray(known_source[field]),
        )
        _print_metrics(
            f"real-stats {field}",
            np.asarray(custom_real_stats[field]),
            np.asarray(known_source[field]),
        )

    print(f"\nSaved BVHs under {args.output_dir}")


if __name__ == "__main__":
    main()
