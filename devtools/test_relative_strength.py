"""Smoke-test relative-strength normalization outside the neural model."""

import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blendanything_server.model_bridge import (
    _meta_to_blend_kwargs,
    _normalized_relative_alpha,
)
from blendanything_server.app import _should_use_ddim_inversion


def main() -> None:
    control = SimpleNamespace(
        y={
            "crop_start_ind": [0, 2],
            "lengths": [6, 6],
        }
    )
    fallback = torch.linspace(0.0, 1.0, 8)
    alpha = _normalized_relative_alpha(
        fallback,
        control,
        ref_samples=[1.0, 1.0, 0.0],
        tgt_samples=[0.0, 1.0, 1.0],
    )
    assert torch.allclose(alpha[:2], fallback[:2])
    assert alpha[2] < alpha[3] < alpha[4] < alpha[5]
    assert 0.0 <= float(alpha.min()) <= float(alpha.max()) <= 1.0

    zeros = _normalized_relative_alpha(
        fallback,
        control,
        ref_samples=[0.0, 0.0],
        tgt_samples=[0.0, 0.0],
    )
    assert torch.allclose(zeros[2:6], fallback[2:6])

    kwargs = _meta_to_blend_kwargs(
        {
            "frame_start": 20,
            "strength": {"profile": "SMOOTH", "samples": [0.0, 1.0]},
        },
        {
            "frame_start": 10,
            "strength": {"profile": "SMOOTH", "samples": [1.0, 0.0]},
        },
    )
    assert kwargs["ref_frame_start"] == 10
    assert kwargs["tgt_frame_start"] == 0
    assert kwargs["overlap_length"] == -1
    assert kwargs["ref_strength_samples"] == [0.0, 1.0]
    assert kwargs["tgt_strength_samples"] == [1.0, 0.0]

    elephant = {"skeleton_profile": "Elephant"}
    skunk = {"skeleton_profile": "Skunk"}
    assert not _should_use_ddim_inversion(
        {"ddim_inversion_policy": "NEVER"}, elephant, elephant
    )
    assert _should_use_ddim_inversion(
        {"ddim_inversion_policy": "SAME_SKELETON"}, elephant, elephant
    )
    assert _should_use_ddim_inversion(
        {"ddim_inversion_policy": "SAME_SKELETON"}, elephant, skunk
    )
    assert not _should_use_ddim_inversion(
        {
            "ddim_inversion_policy": "SAME_SKELETON",
            "control_mode": "tgt",
        },
        elephant,
        skunk,
    )
    assert _should_use_ddim_inversion(
        {
            "ddim_inversion_policy": "SAME_SKELETON",
            "control_mode": "both",
        },
        {
            "skeleton_profile": "truebones::HermitCrab",
            "skeleton_dataset": "truebones",
        },
        {
            "skeleton_profile": "truebones::BrownBear",
            "skeleton_dataset": "truebones",
        },
    )
    assert _should_use_ddim_inversion(
        {"ddim_inversion_policy": "ALWAYS"}, elephant, skunk
    )
    assert _should_use_ddim_inversion(
        {
            "ddim_inversion_policy": "SAME_SKELETON",
            "control_mode": "tgt",
        },
        {
            "skeleton_profile": "truebones::Elephant",
            "skeleton_dataset": "truebones",
        },
        {"skeleton_profile": "Elephant", "object_type": "Elephant"},
    )
    assert _should_use_ddim_inversion(
        {"ddim_inversion_policy": "ON_SAME_SKELETON"}, elephant, elephant
    )
    assert _should_use_ddim_inversion(
        {"ddim_inversion_policy": "WHEN_SAME_SKELETON"}, elephant, elephant
    )
    assert not _should_use_ddim_inversion(
        {
            "ddim_inversion_policy": "SAME_SKELETON",
            "control_mode": "tgt",
        },
        {
            "skeleton_profile": "truebones::Elephant",
            "skeleton_dataset": "truebones",
        },
        {
            "skeleton_profile": "mixamo::Elephant",
            "skeleton_dataset": "mixamo",
        },
    )
    print("Relative-strength normalization smoke test passed")


if __name__ == "__main__":
    main()
