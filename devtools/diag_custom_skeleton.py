"""Diagnose known-vs-custom skeleton conditioning for one BVH.

Example:
    python devtools/diag_custom_skeleton.py Tyranno \
      neural_motion_blending/dataset/truebones/zoo/truebones_processed/bvhs/Tyranno___Run_1068.bvh
"""

import argparse
import json
import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blendanything_server.bvh_pipeline import build_user_conditioning, bvh_to_npy


def _load_catalog(dataset: str) -> dict:
    return np.load(
        os.path.join(REPO_ROOT, "data", f"{dataset}_cond.npy"),
        allow_pickle=True,
    ).item()


def _load_face_joints(dataset: str, character: str, known_cond: dict) -> list:
    path = os.path.join(
        REPO_ROOT,
        "blendanything_client",
        "data",
        f"{dataset}_skeletons.json",
    )
    if os.path.exists(path):
        with open(path, "r") as handle:
            profile = json.load(handle)["profiles"].get(character)
        if profile:
            return list(profile.get("face_joints") or [])

    face = known_cond.get("face_joints")
    names = list(known_cond["joints_names"])
    if face and isinstance(face[0], (int, np.integer)):
        return [names[int(index)] for index in face]
    return list(face or [])


def _metrics(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def _print_metrics(label: str, a: np.ndarray, b: np.ndarray) -> None:
    values = _metrics(a, b)
    print(
        f"{label:<34} "
        f"MAE={values['mae']:.6f}  RMSE={values['rmse']:.6f}  "
        f"MAX={values['max_abs']:.6f}"
    )


def _normalise(motion: np.ndarray, cond: dict) -> np.ndarray:
    return np.nan_to_num(
        (motion - np.asarray(cond["mean"])[None])
        / (np.asarray(cond["std"])[None] + 1e-6)
    )


def _top_joints(label: str, names: list, a: np.ndarray, b: np.ndarray, count: int) -> None:
    per_joint = np.mean(np.abs(a - b), axis=tuple(range(1, a.ndim))[1:])
    if a.ndim == 2:
        per_joint = np.mean(np.abs(a - b), axis=1)
    elif a.ndim == 3:
        per_joint = np.mean(np.abs(a - b), axis=(0, 2))
    print(f"\nWorst {label}:")
    for index in np.argsort(per_joint)[-count:][::-1]:
        print(f"  {index:>3} {names[index]:<32} {per_joint[index]:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("character", help="Known conditioning key, e.g. Tyranno")
    parser.add_argument("bvh", help="BVH motion to diagnose")
    parser.add_argument("--dataset", default="truebones", choices=("truebones", "mixamo"))
    parser.add_argument(
        "--coordinate-space",
        default="MODEL_Y_UP",
        choices=("MODEL_Y_UP", "RAW_TPOSE_RELATIVE", "BLENDER_Z_UP"),
    )
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    catalog = _load_catalog(args.dataset)
    if args.character not in catalog:
        raise SystemExit(f"{args.character!r} not found in {args.dataset}_cond.npy")

    known_cond = catalog[args.character]
    training_catalog = dict(catalog)
    training_catalog.pop(args.character, None)
    face_joints = _load_face_joints(args.dataset, args.character, known_cond)
    if len(face_joints) != 4:
        raise SystemExit(
            f"Need four face joints for {args.character}; got {face_joints!r}"
        )

    with open(args.bvh, "rb") as handle:
        bvh_bytes = handle.read()

    custom_cond = build_user_conditioning(
        bvh_bytes,
        object_type=f"user::{args.character}_heldout",
        coordinate_space=args.coordinate_space,
        face_joint_names=face_joints,
        training_catalog=training_catalog,
    )

    known_raw = bvh_to_npy(
        bvh_bytes,
        known_cond,
        coordinate_space=args.coordinate_space,
        face_joint_names=face_joints,
    )
    custom_raw = bvh_to_npy(
        bvh_bytes,
        custom_cond,
        coordinate_space=args.coordinate_space,
        face_joint_names=face_joints,
    )
    if known_raw is None or custom_raw is None:
        raise SystemExit("BVH conversion failed for known or custom conditioning.")

    print(f"Character: {args.character}")
    print(f"BVH:       {os.path.relpath(args.bvh, REPO_ROOT)}")
    print(f"Space:     {args.coordinate_space}")
    print(f"Joints:    known={len(known_cond['parents'])} custom={len(custom_cond['parents'])}")
    print(f"Frames:    known={known_raw.shape[0]} custom={custom_raw.shape[0]}")
    print(f"Names eq:  {list(known_cond['joints_names']) == list(custom_cond['joints_names'])}")
    print(
        "Parents eq:",
        np.array_equal(known_cond["parents"], custom_cond["parents"]),
    )
    print("Face joints:", face_joints)
    print("Known feet: ", list(known_cond.get("foot_indices", [])))
    print("Custom feet:", list(custom_cond.get("foot_indices", [])))
    print(
        "Statistics:",
        custom_cond.get("statistics_source"),
        f"(prior weight={custom_cond.get('statistics_prior_weight', 0.0):.3f})",
    )

    print("\nConditioning fields")
    _print_metrics("offsets", custom_cond["offsets"], known_cond["offsets"])
    _print_metrics("tpos_first_frame", custom_cond["tpos_first_frame"], known_cond["tpos_first_frame"])
    known_tpos_norm = (
        np.asarray(known_cond["tpos_first_frame"]) - np.asarray(known_cond["mean"])
    ) / (np.asarray(known_cond["std"]) + 1e-6)
    custom_tpos_norm = (
        np.asarray(custom_cond["tpos_first_frame"]) - np.asarray(custom_cond["mean"])
    ) / (np.asarray(custom_cond["std"]) + 1e-6)
    _print_metrics("normalised tpos", custom_tpos_norm, known_tpos_norm)
    _print_metrics("mean", custom_cond["mean"], known_cond["mean"])
    _print_metrics("std", custom_cond["std"], known_cond["std"])
    _print_metrics("joint_relations", custom_cond["joint_relations"], known_cond["joint_relations"])
    _print_metrics("joints_graph_dist", custom_cond["joints_graph_dist"], known_cond["joints_graph_dist"])

    frames = min(len(known_raw), len(custom_raw))
    known_raw = known_raw[:frames]
    custom_raw = custom_raw[:frames]

    print("\nMotion features")
    _print_metrics("raw features", custom_raw, known_raw)
    known_norm = _normalise(known_raw, known_cond)
    custom_norm = _normalise(custom_raw, custom_cond)
    custom_raw_known_stats = _normalise(custom_raw, known_cond)
    known_raw_custom_stats = _normalise(known_raw, custom_cond)
    _print_metrics("normalised full custom", custom_norm, known_norm)
    _print_metrics("custom raw + known stats", custom_raw_known_stats, known_norm)
    _print_metrics("known raw + custom stats", known_raw_custom_stats, known_norm)

    print("\nFeature-wise MAE")
    print("raw:       ", np.mean(np.abs(custom_raw - known_raw), axis=(0, 1)).round(6).tolist())
    print("norm full: ", np.mean(np.abs(custom_norm - known_norm), axis=(0, 1)).round(6).tolist())
    print("stats only:", np.mean(np.abs(known_raw_custom_stats - known_norm), axis=(0, 1)).round(6).tolist())

    _top_joints("raw-feature joints", known_cond["joints_names"], custom_raw, known_raw, args.top)
    _top_joints("mean-stat joints", known_cond["joints_names"], custom_cond["mean"], known_cond["mean"], args.top)
    _top_joints("tpos joints", known_cond["joints_names"], custom_cond["tpos_first_frame"], known_cond["tpos_first_frame"], args.top)


if __name__ == "__main__":
    main()
