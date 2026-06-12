"""Verify request-scoped custom skeleton conditioning."""

import json
import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blendanything_server.bvh_pipeline import (
    build_user_conditioning,
    bvh_to_npy,
    estimate_joint_statistics,
)


def main() -> None:
    bvh_path = os.path.join(
        REPO_ROOT,
        "neural_motion_blending",
        "dataset",
        "truebones",
        "zoo",
        "truebones_processed",
        "bvhs",
        "Coyote___Howling_226.bvh",
    )
    with open(bvh_path, "rb") as handle:
        bvh_bytes = handle.read()

    cond = np.load(
        os.path.join(REPO_ROOT, "data", "truebones_cond.npy"),
        allow_pickle=True,
    ).item()
    with open(
        os.path.join(
            REPO_ROOT,
            "blendanything_client",
            "data",
            "truebones_skeletons.json",
        ),
        "r",
    ) as handle:
        face_joints = json.load(handle)["profiles"]["Coyote"]["face_joints"]

    user_cond = build_user_conditioning(
        bvh_bytes,
        object_type="user::CoyoteCustom",
        coordinate_space="MODEL_Y_UP",
        face_joint_names=face_joints,
        training_catalog=cond,
    )
    assert user_cond["object_type"] == "user::CoyoteCustom"
    assert user_cond["mean"].shape == user_cond["tpos_first_frame"].shape
    assert np.allclose(
        user_cond["tpos_first_frame"],
        cond["Coyote"]["tpos_first_frame"],
        atol=2e-6,
    )
    assert user_cond["foot_indices"] == cond["Coyote"]["foot_indices"]
    assert user_cond["statistics_source"] == "clip_calibrated_semantic_prior"

    motion = bvh_to_npy(
        bvh_bytes,
        user_cond,
        coordinate_space="MODEL_Y_UP",
        face_joint_names=face_joints,
    )
    assert motion is not None
    assert motion.shape[1:] == user_cond["mean"].shape

    held_out = dict(cond)
    coyote = held_out.pop("Coyote")
    matched_mean, matched_std, diagnostics = estimate_joint_statistics(
        coyote["joints_names"],
        coyote["parents"],
        held_out,
    )
    global_mean = np.tile(
        np.concatenate([profile["mean"] for profile in held_out.values()]).mean(axis=0),
        (len(coyote["parents"]), 1),
    )
    global_std = np.tile(
        np.concatenate([profile["std"] for profile in held_out.values()]).mean(axis=0),
        (len(coyote["parents"]), 1),
    )
    matched_error = np.mean((matched_mean - coyote["mean"]) ** 2)
    global_error = np.mean((global_mean - coyote["mean"]) ** 2)
    matched_std_error = np.mean((matched_std - coyote["std"]) ** 2)
    global_std_error = np.mean((global_std - coyote["std"]) ** 2)
    assert matched_error < global_error
    assert matched_std_error < global_std_error
    assert diagnostics[0]["role"] == "root"
    print("Custom skeleton conditioning smoke test passed")


if __name__ == "__main__":
    main()
