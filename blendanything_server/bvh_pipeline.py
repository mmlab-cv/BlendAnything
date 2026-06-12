"""
blend.py — Neural NLA blending inference entry point
=====================================================
Focused inference script designed for the Blender plugin server pipeline.
Takes two raw BVH files (already exported by the plugin), applies NLA clip-editing
instructions from the metadata dict, converts each to the model's feature
representation, runs the diffusion blend, and returns the result as a BVH bytes object.

Usage (programmatic — called from blendanything_server.app):
    from sample.blend import load_model_once, run_blend
    model_bundle = load_model_once(model_path, device)
    result_bvh_bytes = run_blend(model_bundle, ref_bvh_bytes, tgt_bvh_bytes, meta)

The metadata dict mirrors what the Blender plugin sends in its /blend payload:
    {
        "reference": { "name": ..., "repeat": ..., "use_reverse": ...,
                       "strength": {"samples": [...], ...}, ... },
        "targets":   [{ same fields }, ...]
    }

Blend parameters are drawn from the strength samples array in meta:
    - overlap_length  how many frames of crossfade (derived from strength profile)
    - blend_schedule  "ease" (default) — always uses the model's natural latent lerp
    - alpha           mid-blend alpha value (0 = pure ref, 1 = pure tgt, default 0.5)
    - sampler         "ddpm" or "ddim" (default "ddpm" — slower but higher quality)
"""

import io
import json
import math
import os
import re
import sys
import tempfile
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from einops import rearrange

# ── Path setup ───────────────────────────────────────────────────────────────
# blendanything_server/ lives inside BlendAnything/; the neural package is a sibling dir.

_SERVER_DIR   = os.path.dirname(os.path.abspath(__file__))           # BlendAnything/blendanything_server/
_REPO_ROOT    = os.path.dirname(_SERVER_DIR)                         # BlendAnything/
_ROOT         = os.path.join(_REPO_ROOT, "neural_motion_blending")  # BlendAnything/neural_motion_blending/
for _p in (_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Project imports ──────────────────────────────────────────────────────────

import BVH
from Animation import Animation, positions_global
from InverseKinematics import animation_from_positions
from Quaternions import Quaternions

from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
from utils import dist_util
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import (
    recover_from_bvh_ric_np, get_foot_contact, get_rifke,
    get_motion_features, get_bvh_cont6d_params, rotate_to_hml_orientation,
    move_xz_to_origin, scale as scale_animation, put_on_ground,
    create_topology_edge_relations, parents2kinchains, process_anim,
    get_mean_std,
)
from data_loaders.truebones.truebones_utils.param_utils import FOOT_CONTACT_VEL_THRESH
from utils.bvh_io import bvh_load_safe as _bvh_load_safe
from model.modules.conditioners import T5Conditioner
from model.motion_diffusion_ae import ControlConfig, MoDiffAE
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window


# ── Helpers: BVH bytes ↔ temp file ──────────────────────────────────────────

def _bytes_to_tmp_bvh(data: bytes) -> str:
    """Write bytes to a temp .bvh file; caller must os.unlink when done."""
    with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False, prefix="blend_in_") as f:
        f.write(data)
        return f.name


def _save_bvh_bytes(anim, names: list, frametime: float) -> bytes:
    """Serialise an Animation object to BVH bytes."""
    with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False, prefix="blend_out_") as f:
        tmp = f.name
    try:
        BVH.save(tmp, anim, names=names, frametime=frametime)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp)


# ── BVH → model feature representation ──────────────────────────────────────

def _reorder_anim(
    anim,
    bvh_names,
    ref_names,
    ocd=None,
    *,
    synthesize_missing_leaves=True,
):
    """
    Permute joints of an Animation to match ref_names ordering.
    Handles Blender reordering and inserts missing end-site joints
    with identity rotations. Copied from eval/fid_truebones_blend.py.
    """
    if list(bvh_names) == list(ref_names):
        return anim

    name_to_bvh = {n: i for i, n in enumerate(bvh_names)}
    missing = [n for n in ref_names if n not in name_to_bvh]

    # Custom BVHs may use different names while preserving the exact known
    # hierarchy. BVH.load is hierarchy-ordered, so an identical parent array is
    # a deterministic positional mapping.
    if (
        missing
        and ocd is not None
        and len(bvh_names) == len(ref_names)
        and np.array_equal(np.asarray(anim.parents), np.asarray(ocd["parents"]))
    ):
        name_to_bvh = {name: i for i, name in enumerate(ref_names)}
        missing = []

    # Pass 1: Blender '__suffix' dedup stripping (original heuristic).
    if missing:
        def _strip(name):
            parts = name.rsplit('__', 1)
            return parts[0] if len(parts) == 2 and parts[0] else name
        stripped = [_strip(n) for n in bvh_names]
        name_to_bvh = {s: i for i, s in enumerate(stripped)}
        missing = [n for n in ref_names if n not in name_to_bvh]

    # Generic Blender end-site names are not stable identifiers. A parent can
    # have several named endpoint joints, so guessing by suffix can attach the
    # wrong offset. Keep exact/deduplicated names only; missing cond_dict leaves
    # are synthesized below from authoritative topology and offsets.

    if missing:
        if ocd is None:
            raise ValueError(f'BVH missing joints not in cond_dict: {missing}')
        if not synthesize_missing_leaves:
            raise ValueError(
                "Raw T-pose-relative BVH must retain the complete named hierarchy; "
                f"missing joints: {missing}"
            )
        ref_parents = list(ocd['parents'])
        ref_offsets = np.array(ocd['offsets'])
        T, J_bvh = anim.rotations.shape
        ref_name_to_idx = {n: i for i, n in enumerate(ref_names)}
        for m in missing:
            m_idx = ref_name_to_idx[m]
            children = [i for i, p in enumerate(ref_parents) if p == m_idx]
            if children:
                raise ValueError(f'BVH missing non-leaf joint "{m}"')

        bvh_names = list(bvh_names)
        n_missing = len(missing)
        new_rots_ext = np.zeros((T, n_missing, 4)); new_rots_ext[..., 0] = 1.0
        new_pos_ext  = np.zeros((T, n_missing, 3))
        new_off_ext  = np.zeros((n_missing, 3))
        new_ori_ext  = np.zeros((n_missing, 4)); new_ori_ext[..., 0] = 1.0
        new_par_ext  = []

        for m in missing:
            m_idx   = ref_name_to_idx[m]
            ref_par = ref_parents[m_idx]
            par_name = ref_names[ref_par] if ref_par != -1 else None
            par_bvh  = name_to_bvh.get(par_name, -1) if par_name else -1
            missing_idx = missing.index(m)
            new_par_ext.append(par_bvh)
            new_off_ext[missing_idx] = ref_offsets[m_idx]
            new_pos_ext[:, missing_idx] = ref_offsets[m_idx]
            bvh_names.append(m)

        anim = Animation(
            Quaternions(np.concatenate([anim.rotations.qs, new_rots_ext], axis=1)),
            np.concatenate([anim.positions, new_pos_ext], axis=1),
            Quaternions(np.concatenate([anim.orients.qs, new_ori_ext], axis=0)),
            np.concatenate([anim.offsets, new_off_ext], axis=0),
            np.concatenate([anim.parents, new_par_ext]),
        )
        for k, m in enumerate(missing):
            name_to_bvh[m] = J_bvh + k

    perm = np.array([name_to_bvh[n] for n in ref_names])
    bvh_to_new = {old: new for new, old in enumerate(perm)}
    new_parents = np.array([
        bvh_to_new[anim.parents[perm[i]]] if anim.parents[perm[i]] != -1 else -1
        for i in range(len(ref_names))
    ])
    if ocd is not None:
        expected_parents = np.asarray(ocd["parents"])
        if not np.array_equal(new_parents, expected_parents):
            mismatches = [
                f"{ref_names[i]}: got parent {new_parents[i]}, expected {expected_parents[i]}"
                for i in range(len(ref_names))
                if new_parents[i] != expected_parents[i]
            ]
            raise ValueError(
                "BVH hierarchy does not match cond_dict after joint reconciliation: "
                + "; ".join(mismatches[:8])
            )
    return Animation(
        anim.rotations[:, perm],
        anim.positions[:, perm],
        anim.orients[perm],
        anim.offsets[perm],
        new_parents,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Blender-export fallback basis
# ─────────────────────────────────────────────────────────────────────────────
# The preferred plugin path uploads the original processed BVH in model space.
# If no source BVH is assigned, Blender exports (x, -z, y), a +90 degree X
# rotation from model space. Apply the inverse once before feature extraction.
_BLENDER_TO_MODEL = Quaternions.from_euler(
    np.array([-np.pi / 2.0, 0.0, 0.0]), order="xyz"
)


def _apply_world_rotation(anim: Animation, q: Quaternions) -> Animation:
    """Rotate the whole animation's root channel by the fixed world quaternion q.

    Only the root joint carries world orientation (children are relative), so we
    pre-multiply q onto the root rotation and rotate the root translation.
    Follows the broadcast pattern of rotate_to_hml_orientation: repeat the single
    quaternion across the time axis before multiplying.
    """
    n_frames = anim.rotations.shape[0]
    q_t = q.repeat(n_frames, axis=0)            # (T,) to match per-frame root
    new_rots = anim.rotations.copy()
    new_rots[:, 0] = q_t * new_rots[:, 0]
    new_pos = anim.positions.copy()
    new_pos[:, 0] = q_t * new_pos[:, 0]
    return Animation(new_rots, new_pos, anim.orients.copy(),
                     anim.offsets.copy(), anim.parents.copy())


def _infer_up_rotation(anim: Animation, foot_indices: list) -> Quaternions:
    """Infer a signed cardinal up axis from root/foot geometry."""
    global_positions = positions_global(anim)
    candidates = [
        ("+Y", Quaternions.id(1)),
        ("-Y", Quaternions.from_euler(np.array([np.pi, 0.0, 0.0]), order="xyz")),
        ("+Z", Quaternions.from_euler(np.array([-np.pi / 2.0, 0.0, 0.0]), order="xyz")),
        ("-Z", Quaternions.from_euler(np.array([np.pi / 2.0, 0.0, 0.0]), order="xyz")),
        ("+X", Quaternions.from_euler(np.array([0.0, 0.0, np.pi / 2.0]), order="xyz")),
        ("-X", Quaternions.from_euler(np.array([0.0, 0.0, -np.pi / 2.0]), order="xyz")),
    ]
    feet = [i for i in foot_indices if 0 <= int(i) < global_positions.shape[1]]
    if not feet:
        feet = list(range(1, global_positions.shape[1]))

    ranked = []
    flat = global_positions.reshape(-1, 3)
    for label, rotation in candidates:
        rotated = rotation * flat
        rotated = rotated.reshape(global_positions.shape)
        y = rotated[..., 1]
        span = max(float(np.percentile(y, 95) - np.percentile(y, 5)), 1e-6)
        root_height = np.median(y[:, 0])
        foot_height = np.median(y[:, feet])
        low_height = np.percentile(y, 5)
        # Correct up puts the root above feet and feet near the lower envelope.
        score = (root_height - foot_height) / span
        score -= abs(foot_height - low_height) / span
        ranked.append((score, label, rotation))

    ranked.sort(key=lambda item: item[0], reverse=True)
    best, second = ranked[0], ranked[1]
    if best[0] - second[0] < 0.05:
        print(
            f"[blend.py] Up-axis inference is weak: {best[1]}={best[0]:.3f}, "
            f"{second[1]}={second[0]:.3f}; using {best[1]}."
        )
    return best[2]


def _extract_features(anim: Animation, object_cond: dict, face_joints: list) -> np.ndarray:
    cont6d, _, _, root_rot, global_positions = get_bvh_cont6d_params(
        anim,
        object_cond["object_type"],
        face_joints=face_joints,
    )
    foot = get_foot_contact(
        global_positions,
        object_cond["foot_indices"],
        FOOT_CONTACT_VEL_THRESH,
    )
    positions = get_rifke(global_positions, root_rot)
    local_vel = (
        np.repeat(root_rot[1:, None], global_positions.shape[1], axis=1)
        * (global_positions[1:] - global_positions[:-1])
    )
    features, _ = get_motion_features(
        positions,
        cont6d,
        foot,
        local_vel,
        len(object_cond["offsets"]),
    )
    return features


def _infer_foot_indices(anim: Animation, names: list = None) -> list:
    """Choose likely contact joints using names first, then low leaf joints."""
    parents = list(anim.parents)
    if names:
        feet = [
            index for index, name in enumerate(names)
            if any(
                token in str(name).lower()
                for token in ("toe", "foot", "phalanx", "hoof", "ashi")
            )
        ]
        for index in list(feet):
            if index in parents:
                for child, parent in enumerate(parents):
                    if parent == index and child not in feet:
                        feet.append(child)
        if feet:
            return feet

    children = {int(parent) for parent in parents if int(parent) >= 0}
    leaves = [index for index in range(1, len(parents)) if index not in children]
    if not leaves:
        leaves = list(range(1, len(parents)))
    positions = positions_global(anim)[0]
    leaves.sort(key=lambda index: float(positions[index, 1]))
    return leaves[: min(8, len(leaves))]


_NAME_NOISE = {
    "bip", "bn", "joint", "jnt", "jt", "bone", "mixamorig", "npc",
    "end", "site", "nub", "base", "top", "01", "02", "03", "04",
}


def _joint_tokens(name: str) -> set:
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    return {
        token
        for token in re.findall(r"[a-z]+|\d+", expanded.lower())
        if token not in _NAME_NOISE and not token.isdigit()
    }


def _joint_side(name: str, tokens: set) -> str:
    lowered = str(name).lower()
    if "left" in tokens or re.search(r"(^|[_\-.])l([_\-.]|$)", lowered):
        return "left"
    if "right" in tokens or re.search(r"(^|[_\-.])r([_\-.]|$)", lowered):
        return "right"
    return "center"


def _joint_role(name: str, tokens: set, is_root: bool) -> str:
    if is_root:
        return "root"
    text = "".join(tokens)
    role_groups = (
        ("pelvis", ("pelvis", "hips", "hip")),
        ("spine", ("spine", "chest", "torso", "abdomen")),
        ("neck", ("neck",)),
        ("head", ("head", "skull")),
        ("shoulder", ("shoulder", "clavicle", "collar")),
        ("upper_arm", ("upperarm", "uparm", "humerus")),
        ("forearm", ("forearm", "lowerarm", "elbow")),
        ("hand", ("hand", "wrist", "palm")),
        ("finger", ("finger", "thumb", "index", "middle", "ring", "pinky")),
        ("thigh", ("thigh", "upleg", "upperleg", "femur")),
        ("calf", ("calf", "leg", "lowerleg", "shin", "knee")),
        ("foot", ("foot", "ankle", "hoof")),
        ("toe", ("toe", "phalanx")),
        ("tail", ("tail",)),
        ("wing", ("wing",)),
        ("ear", ("ear",)),
        ("eye", ("eye",)),
        ("face", ("jaw", "mouth", "nose", "beak", "tongue")),
        ("hair_cloth", ("hair", "cape", "skirt", "cloth", "sleeve")),
    )
    for role, aliases in role_groups:
        if any(alias in text for alias in aliases):
            return role
    return "generic"


def _joint_descriptors(names: list, parents: np.ndarray) -> list:
    parents = np.asarray(parents, dtype=np.int64)
    children = np.bincount(parents[parents >= 0], minlength=len(parents))
    depths = np.zeros(len(parents), dtype=np.int64)
    for index in range(1, len(parents)):
        parent = int(parents[index])
        depths[index] = depths[parent] + 1 if parent >= 0 else 0
    max_depth = max(int(depths.max()), 1)
    descriptors = []
    for index, name in enumerate(names):
        tokens = _joint_tokens(name)
        descriptors.append({
            "tokens": tokens,
            "side": _joint_side(name, tokens),
            "role": _joint_role(name, tokens, index == 0),
            "root": index == 0,
            "leaf": int(children[index]) == 0,
            "children": min(int(children[index]), 4),
            "depth": float(depths[index]) / max_depth,
        })
    return descriptors


def estimate_joint_statistics(
    joint_names: list,
    parents: np.ndarray,
    training_catalog: dict,
    *,
    neighbors: int = 24,
) -> tuple:
    """Estimate per-joint normalization from semantically similar train joints."""
    target_descriptors = _joint_descriptors(joint_names, parents)
    donors = []
    for profile in training_catalog.values():
        donor_names = list(profile["joints_names"])
        donor_descriptors = _joint_descriptors(donor_names, profile["parents"])
        for index, descriptor in enumerate(donor_descriptors):
            donors.append((
                descriptor,
                np.asarray(profile["mean"][index], dtype=np.float64),
                np.asarray(profile["std"][index], dtype=np.float64),
            ))

    means = []
    stds = []
    diagnostics = []
    for target in target_descriptors:
        ranked = []
        for donor, mean, std in donors:
            if target["root"] != donor["root"]:
                continue
            union = target["tokens"] | donor["tokens"]
            token_score = (
                len(target["tokens"] & donor["tokens"]) / len(union)
                if union else 0.0
            )
            score = 0.0
            score += 5.0 if target["role"] == donor["role"] else 0.0
            score += 3.0 * token_score
            score += 1.25 if target["side"] == donor["side"] else 0.0
            score += 1.0 if target["leaf"] == donor["leaf"] else 0.0
            score += 1.0 - min(abs(target["depth"] - donor["depth"]), 1.0)
            score += 0.5 * (
                1.0 - min(abs(target["children"] - donor["children"]) / 4.0, 1.0)
            )
            ranked.append((score, mean, std, donor["role"]))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[: max(1, min(neighbors, len(ranked)))]
        scores = np.asarray([item[0] for item in selected], dtype=np.float64)
        weights = np.exp((scores - scores.max()) / 1.5)
        weights /= weights.sum()
        mean = np.sum(
            np.stack([item[1] for item in selected]) * weights[:, None],
            axis=0,
        )
        # Dataset preprocessing already pools several std channels across
        # joints. Preserve that normalization scale instead of treating donor
        # disagreement as additional motion variance.
        std = np.sum(
            np.stack([item[2] for item in selected]) * weights[:, None],
            axis=0,
        )
        means.append(mean)
        stds.append(np.maximum(std, 1e-6))
        diagnostics.append({
            "role": target["role"],
            "best_score": float(scores[0]),
            "donor_roles": sorted({item[3] for item in selected[:5]}),
        })
    return np.stack(means), np.stack(stds), diagnostics


def build_user_conditioning(
    bvh_bytes: bytes,
    *,
    object_type: str,
    coordinate_space: str,
    face_joint_names: list,
    training_catalog: dict,
) -> dict:
    """Derive an approximate cond_dict entry for one uploaded user BVH."""
    if not face_joint_names or len(face_joint_names) != 4:
        raise ValueError("Custom skeletons require four face-joint names.")

    tmp = _bytes_to_tmp_bvh(bvh_bytes)
    try:
        anim, names, _ = _bvh_load_safe(tmp)
    finally:
        os.unlink(tmp)

    names = list(names)
    missing_face = [name for name in face_joint_names if name not in names]
    if missing_face:
        raise ValueError(
            f"Custom skeleton face joints are missing from BVH: {missing_face}"
        )
    face_joints = [names.index(name) for name in face_joint_names]

    if coordinate_space == "BLENDER_Z_UP":
        anim = _apply_world_rotation(anim, _BLENDER_TO_MODEL)
        root_pose_init_xz = np.zeros(3)
        ground_height = 0.0
        scale_factor = 1.0
    elif coordinate_space == "RAW_TPOSE_RELATIVE":
        foot_guess = _infer_foot_indices(anim, names)
        anim = _apply_world_rotation(anim, _infer_up_rotation(anim, foot_guess))
        anim, root_pose_init_xz, ground_height, scale_factor = process_anim(
            anim,
            object_type,
            face_joints=face_joints,
        )
    elif coordinate_space == "MODEL_Y_UP":
        root_pose_init_xz = np.zeros(3)
        ground_height = 0.0
        scale_factor = 1.0
    else:
        raise ValueError(f"Unsupported coordinate_space: {coordinate_space}")

    parents = np.asarray(anim.parents, dtype=np.int64)
    offsets = np.asarray(anim.offsets, dtype=np.float64)
    foot_indices = _infer_foot_indices(anim, names)
    joint_relations, joints_graph_dist = create_topology_edge_relations(parents)
    rest_positions = np.tile(offsets[None], (2, 1, 1))
    rest_anim = Animation(
        Quaternions.id((2, len(parents))),
        rest_positions,
        anim.orients.copy(),
        offsets.copy(),
        parents.copy(),
    )
    tpos_first_frame = _extract_features(
        rest_anim,
        {
            "object_type": object_type,
            "foot_indices": foot_indices,
            "offsets": offsets,
        },
        face_joints,
    )[0]
    prior_mean, prior_std, statistics_matches = estimate_joint_statistics(
        names,
        parents,
        training_catalog,
    )
    motion_features = _extract_features(
        anim,
        {
            "object_type": object_type,
            "foot_indices": foot_indices,
            "offsets": offsets,
        },
        face_joints,
    )
    if len(motion_features) == 0:
        raise ValueError(
            "Custom skeleton BVH must contain at least two motion frames."
        )
    clip_mean, clip_std = get_mean_std(motion_features.copy())
    prior_weight = 10.0 / (len(motion_features) + 10.0)
    mean = prior_weight * prior_mean + (1.0 - prior_weight) * clip_mean
    variance = (
        prior_weight
        * (prior_std ** 2 + (prior_mean - mean) ** 2)
        + (1.0 - prior_weight)
        * (clip_std ** 2 + (clip_mean - mean) ** 2)
    )
    std = np.sqrt(np.maximum(variance, 1e-6))

    return {
        "tpos_first_frame": tpos_first_frame,
        "joint_relations": joint_relations,
        "joints_graph_dist": joints_graph_dist,
        "object_type": object_type,
        "parents": parents,
        "offsets": offsets,
        "joints_names": names,
        "kinematic_chains": parents2kinchains(parents),
        "mean": mean,
        "std": std,
        "root_pose_init_xz": np.asarray(root_pose_init_xz, dtype=np.float64),
        "scale_factor": np.asarray(scale_factor, dtype=np.float64),
        "ground_height": np.asarray(ground_height, dtype=np.float64),
        "tpos_rots": anim.rotations[:1],
        "foot_indices": foot_indices,
        "face_joints": face_joints,
        "statistics_matches": statistics_matches,
        "statistics_prior_weight": prior_weight,
        "statistics_source": "clip_calibrated_semantic_prior",
    }


def bvh_to_npy(
    bvh_bytes: bytes,
    object_cond: dict,
    *,
    coordinate_space: str = "MODEL_Y_UP",
    face_joint_names: Optional[list] = None,
) -> Optional[np.ndarray]:
    """Convert a processed BVH to the model's ``(T-1, J, 13)`` features.

    The Truebones BVHs used by the plugin are already HML-aligned and
    T-pose-relative. Calling ``get_motion`` here rebases them against the
    T-pose a second time. Instead, this mirrors the evaluation converter:
    undo Blender's Z-up export basis, reconcile the skeleton against cond.npy,
    and extract features directly.
    """
    tmp = _bytes_to_tmp_bvh(bvh_bytes)
    try:
        anim, bvh_names, _ = _bvh_load_safe(tmp)
    finally:
        os.unlink(tmp)

    try:
        raw_tpose_relative = coordinate_space == "RAW_TPOSE_RELATIVE"
        if coordinate_space == "BLENDER_Z_UP":
            anim = _apply_world_rotation(anim, _BLENDER_TO_MODEL)
        elif coordinate_space not in {"MODEL_Y_UP", "RAW_TPOSE_RELATIVE"}:
            raise ValueError(f"Unsupported coordinate_space: {coordinate_space}")

        ref_names = list(object_cond["joints_names"])
        anim = _reorder_anim(
            anim,
            list(bvh_names),
            ref_names,
            ocd=object_cond,
            synthesize_missing_leaves=not raw_tpose_relative,
        )

        if face_joint_names:
            missing_face = [name for name in face_joint_names if name not in ref_names]
            if missing_face:
                raise ValueError(f"Unknown face joints for profile: {missing_face}")
            face_joints = [ref_names.index(name) for name in face_joint_names]
        else:
            face_joints = object_cond.get("face_joints")

        if raw_tpose_relative:
            if not face_joints or len(face_joints) != 4:
                raise ValueError(
                    "RAW_TPOSE_RELATIVE requires four face-joint names "
                    "(right, left, upper-right, upper-left)."
                )
            up_rotation = _infer_up_rotation(anim, object_cond["foot_indices"])
            anim = _apply_world_rotation(anim, up_rotation)
            anim = rotate_to_hml_orientation(
                anim, object_cond["object_type"], face_joints=face_joints
            )
            anim, _ = move_xz_to_origin(anim, None)
            anim, _ = scale_animation(anim, None)

            # T-pose-relative input lets us reconstruct a zero-rotation rest
            # pose from offsets. Ground against that pose, not against the
            # motion's lowest frame (which changes root height per action).
            rest_positions = anim.offsets[None].copy()
            rest = Animation(
                Quaternions.id((1, len(anim.parents))),
                rest_positions,
                anim.orients.copy(),
                anim.offsets.copy(),
                anim.parents.copy(),
            )
            rest_ground = float(positions_global(rest)[..., 1].min())
            anim, _ = put_on_ground(anim, rest_ground)

        return _extract_features(anim, object_cond, face_joints)
    except Exception as exc:
        print(f"[blend.py] bvh_to_npy failed: {exc}")
        import traceback
        traceback.print_exc()
        return None


# ── Strength → overlap parameters ───────────────────────────────────────────

def _parse_overlap_from_strength(ref_meta: dict, tgt_meta: dict) -> int:
    """
    Derive the crossfade overlap length (frames) from the strength profile
    sent by the Blender plugin.

    Strategy: the samples array in ref_meta["strength"] encodes the plugin's
    blend envelope as one value per frame.  We count the number of frames
    where the envelope is neither at its minimum nor maximum as the
    "transition zone" and use that as the overlap.

    Falls back to a default of 20 frames if the profile is constant or missing.
    """
    ref_strength = ref_meta.get("strength", {})
    samples = ref_strength.get("samples", [])
    if not samples:
        return 20

    profile = ref_strength.get("profile", "CONSTANT")
    if profile == "CONSTANT":
        return 20

    # count frames where envelope is strictly between min and max
    arr = np.array(samples, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-6:
        return 20
    transition = np.sum((arr > lo + 1e-6) & (arr < hi - 1e-6))
    return max(1, int(transition))


# ── Batch preparation (mirrors mix.py's get_control_batches / create_sample_in_batch) ──

def _encode_joints_names(joints_names: list, t5_conditioner) -> np.ndarray:
    tokens = t5_conditioner.tokenize(joints_names)
    embs = t5_conditioner(tokens)
    return embs.detach().cpu().numpy()


def _make_batch_element(
    object_type: str,
    object_cond: dict,
    motion: np.ndarray,
    opt,
    t5_conditioner,
    frame_start: int = 0,
    loop_times: int = 1,
) -> list:
    """
    Build a single-sample list in the format expected by truebones_batch_collate.

    Mirrors create_sample_in_batch from mix.py, but takes the feature array
    directly rather than loading from disk.
    """
    parents = object_cond["parents"]
    n_joints = len(parents)
    mean = object_cond["mean"]
    std = object_cond["std"]
    n_frames = motion.shape[0]

    if loop_times > 1:
        n_frames *= loop_times
        reps = [loop_times] + [1] * (motion.ndim - 1)
        motion = np.tile(motion, reps)

    # Normalise
    motion_norm = (motion - mean[None]) / (std[None] + 1e-6)
    motion_norm = np.nan_to_num(motion_norm)

    tpos_ff = object_cond["tpos_first_frame"]
    tpos_ff_norm = (tpos_ff - mean) / (std + 1e-6)
    tpos_ff_norm = np.nan_to_num(tpos_ff_norm)

    joint_relations   = object_cond["joint_relations"]
    joints_graph_dist = object_cond["joints_graph_dist"]
    offsets           = object_cond["offsets"]
    kin_chain         = object_cond["kinematic_chains"]
    joints_names_embs = _encode_joints_names(object_cond["joints_names"], t5_conditioner)
    temporal_window   = getattr(opt, "temporal_window", 31)
    temporal_mask     = create_temporal_mask_for_window(temporal_window, n_frames)

    return [
        motion_norm, n_frames, parents, tpos_ff_norm,
        offsets, temporal_mask, joints_graph_dist,
        joint_relations, object_type, joints_names_embs,
        frame_start, mean, std, kin_chain, opt.max_joints,
    ]


# ── Alpha tensor + blend mask (mirrors mix.py's format_alpha_tensor) ────────

def _make_alpha_and_mask(batch, control, alpha: float, overlap_length: int, schedule: str):
    """
    Build per-frame alpha tensor and blend mask for a temporal blend.

    Unlike mix.py we always use a fixed overlap_length derived from the
    strength profile rather than relying on ref/tgt frame_starts.
    """
    T = batch.shape[-1]
    device = batch.device

    ref_s = int(control.y["crop_start_ind"][0])
    ref_e = ref_s + int(control.y["lengths"][0])
    tgt_s = int(control.y["crop_start_ind"][1])
    tgt_e = tgt_s + int(control.y["lengths"][1])

    alpha_tensor = torch.zeros(T, device=device)
    if ref_s <= tgt_s:
        alpha_tensor[ref_e:] = 1.0
    else:
        alpha_tensor[:ref_s] = 1.0

    blend_mask = torch.zeros(T, dtype=torch.bool, device=device)
    blend_mask[ref_s:ref_e] = True
    blend_mask[tgt_s:tgt_e] = True

    inter_s = max(ref_s, tgt_s)
    inter_e = min(ref_e, tgt_e)

    if schedule == "static":
        alpha_tensor[:] = alpha
    elif inter_e > inter_s:
        dist = inter_e - inter_s
        t = torch.linspace(0, 1, dist, device=device)
        if schedule == "linear":
            ramp = t
        else:  # ease — symmetric sine curve
            s = 0.5 * (1 + torch.sin(math.pi * (t - 0.5)))
            ramp = s
        if ref_s <= tgt_s:
            alpha_tensor[inter_s:inter_e] = ramp
        else:
            alpha_tensor[inter_s:inter_e] = 1.0 - ramp

    return alpha_tensor, blend_mask


# ── Core format helpers (verbatim from mix.py) ───────────────────────────────

def _format_control_batch(ref_batch, tgt_batch, interpolation_mode: str = "lerp"):
    """Merge ref + tgt into a ControlConfig (mirrors mix.py's format_control_batch)."""
    merged_batch, merged_kwargs = truebones_batch_collate(ref_batch + tgt_batch)
    bs = len(ref_batch)
    control = ControlConfig(
        x=torch.zeros_like(merged_batch),
        y={},
        alpha=torch.zeros(bs),
        interpolation_mode=interpolation_mode,
    )

    def _fmt(v):
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return rearrange(v, "(s b) ... -> (b s) ...", s=2)
        return [item for pair in zip(v[:bs], v[bs:]) for item in pair]

    control.x = rearrange(merged_batch, "(s b) ... -> b s ...", s=2)
    for k, v in merged_kwargs["y"].items():
        control.y[k] = _fmt(v)
    control.x = control.x.to(dist_util.dev())
    return control


def _format_gen_batch(control, cond_dict: dict, t5_conditioner, opt):
    """Build the generation batch (mirrors mix.py's format_gen_batch)."""
    gen_object_type = control.y["object_type"][0]
    gen_n_frames = max(
        a + b for a, b in zip(control.y["lengths"], control.y["crop_start_ind"])
    )
    gen_batch_elem = _make_batch_element(
        gen_object_type,
        cond_dict[gen_object_type],
        motion=np.zeros((gen_n_frames, len(cond_dict[gen_object_type]["parents"]), 13)),
        opt=opt,
        t5_conditioner=t5_conditioner,
        frame_start=0,
        loop_times=1,
    )
    batch, model_kwargs = truebones_batch_collate([gen_batch_elem])
    return batch, model_kwargs


# ── Post-process: model output → BVH bytes ──────────────────────────────────

def _motion_to_bvh_bytes(
    motion_tensor: torch.Tensor,
    model_kwargs: dict,
    cond_dict: dict,
    frametime: float,
) -> bytes:
    """
    Denormalise and convert a single-sample model output back to BVH bytes.

    motion_tensor: (max_joints, feats, T) — one element from sampling output.
    """
    i = 0
    n_joints = model_kwargs["y"]["n_joints"][i].item()
    object_type = model_kwargs["y"]["object_type"][i]
    parents = model_kwargs["y"]["parents"][i]
    mean = cond_dict[object_type]["mean"][None, :]
    std  = cond_dict[object_type]["std"][None, :]
    offsets = cond_dict[object_type]["offsets"]
    joints_names = cond_dict[object_type]["joints_names"]

    motion = motion_tensor[:n_joints]
    motion = motion.cpu().permute(2, 0, 1).numpy() * std + mean  # (T, joints, 13)

    global_positions = recover_from_bvh_ric_np(motion)  # (T, joints, 3)

    out_anim, _, _ = animation_from_positions(
        positions=global_positions,
        parents=parents,
        offsets=offsets,
        iterations=150,
    )

    return _save_bvh_bytes(out_anim, joints_names, frametime)


# ── Model bundle — loaded once and reused across requests ───────────────────

class ModelBundle:
    """Holds all expensive-to-load objects so they survive across HTTP requests."""

    def __init__(self, model, diffusion, t5_conditioner, cond_dict, opt, args):
        self.model         = model
        self.diffusion     = diffusion
        self.t5_conditioner = t5_conditioner
        self.cond_dict     = cond_dict
        self.opt           = opt
        self.args          = args


def load_model_once(model_path: str, device: int = 0, dataset: str = "truebones") -> ModelBundle:
    """
    Load and return all inference components.  Call this once at server startup
    and pass the returned ModelBundle to run_blend on every request.

    model_path must point to a model####.pt file whose directory contains args.json.
    """
    # Build a minimal args namespace that parse_and_load_from_model would produce,
    # then overwrite the model-architecture fields from args.json.
    import argparse, json, copy
    from utils.parser_util import add_base_options, add_model_options, add_data_options

    parser = argparse.ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)

    # Parse only the args we control; ignore everything else (uvicorn's sys.argv).
    args, _ = parser.parse_known_args([
        "--device",  str(device),
        "--dataset", dataset,
    ])
    args.model_path = model_path

    # Overwrite architecture-specific args from the saved args.json
    args_json_path = os.path.join(os.path.dirname(model_path), "args.json")
    if os.path.exists(args_json_path):
        with open(args_json_path) as f:
            saved = json.load(f)
        # Fields that must come from the checkpoint
        arch_fields = [
            "arch", "layers", "latent_dim", "t5_name", "temporal_window",
            "skip_t5", "value_emb", "model_name", "num_layers_semantic",
            "num_layers_stochastic", "num_virtual_joints", "compress_virtual_joints",
            "projection_head_depth", "kl_bottleneck", "second_temporal_attn",
            "semantic_feat_mode", "condition_strategy", "emb_trans_dec",
            "noise_schedule", "diffusion_steps", "sigma_small",
        ]
        for field in arch_fields:
            if field in saved:
                setattr(args, field, saved[field])
        # backward compat (same as extract_args in parser_util)
        if isinstance(args.emb_trans_dec, bool):
            if args.emb_trans_dec:
                args.emb_trans_dec = "cls_tcond_cross_tcond"
            else:
                args.emb_trans_dec = "cls_none_cross_tcond"

    # Also populate fields used by the sampling path
    args.sampler       = "ddpm"
    args.ddim_steps    = -1
    args.cond_mask_prob = getattr(args, "cond_mask_prob", 0.0)
    args.guidance_scale = getattr(args, "guidance_scale", 1.0)
    args.lambda_fs      = getattr(args, "lambda_fs", 0.0)
    args.lambda_geo     = getattr(args, "lambda_geo", 0.0)
    args.lambda_kl      = getattr(args, "lambda_kl", 0.0)

    fixseed(10)
    dist_util.setup_dist(device)
    opt = get_opt(device, dataset=dataset)

    print("[blend.py] Loading model…")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    state_dict = torch.load(model_path, map_location="cpu")
    load_model(model, state_dict)
    model.to(dist_util.dev())
    model.eval()

    print(f"[blend.py] Loading T5 ({args.t5_name})…")
    t5 = T5Conditioner(
        name=args.t5_name,
        finetune=False,
        word_dropout=0.0,
        normalize_text=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("[blend.py] Loading cond_dict…")
    if dataset == "truebones":
        cond_file = os.path.join(_REPO_ROOT, "data", "truebones_cond.npy")
    else:
        cond_file = opt.cond_file if os.path.isabs(opt.cond_file) else os.path.join(_ROOT, opt.cond_file)
    cond_dict = np.load(cond_file, allow_pickle=True).item()

    print("[blend.py] Model ready.")
    return ModelBundle(model, diffusion, t5, cond_dict, opt, args)


# ── Main inference entry point ───────────────────────────────────────────────

def run_blend(
    bundle: ModelBundle,
    ref_bvh_bytes: bytes,
    tgt_bvh_bytes: bytes,
    meta: dict,
    *,
    object_type: Optional[str] = None,
    alpha: float = 0.5,
    blend_schedule: str = "ease",
    sampler: str = "ddpm",
    seed: int = 10,
) -> bytes:
    """
    Run neural blending on two BVH clips and return the blended BVH as bytes.

    Delegates to call.blend() which owns the authoritative pipeline implementation.

    Parameters
    ----------
    bundle          ModelBundle returned by load_model_once().
    ref_bvh_bytes   Processed reference BVH bytes (after apply_clip_edits).
    tgt_bvh_bytes   Processed target BVH bytes.
    meta            Full metadata dict from the /blend payload:
                      {"reference": {...}, "targets": [...],
                       "control_mode": "both"|"tgt", "blend_mode": "BLEND"|"RETARGET"}
    object_type     Hint for ref skeleton type; if None, inferred from strip name.
    alpha           Blend weight (0 = pure ref, 1 = pure tgt).
    blend_schedule  "ease" | "linear" | "static"
    sampler         "ddpm" | "ddim" — ignored here, set at load_model_once time.
    seed            Random seed.
    """
    from blendanything_server.model_bridge import BlendPipeline, blend as call_blend

    # Wrap ModelBundle fields into the BlendPipeline dataclass that call.py expects.
    # Both hold the same objects under different names.
    pipeline = BlendPipeline(
        model          = bundle.model,
        diffusion      = bundle.diffusion,
        t5_conditioner = bundle.t5_conditioner,
        cond_dict      = bundle.cond_dict,
        opt            = bundle.opt,
        args           = bundle.args,
        sampling_fun   = (
            bundle.diffusion.ddim_sample_loop if sampler == "ddim"
            else bundle.diffusion.p_sample_loop
        ),
    )

    ref_meta     = meta.get("reference", {})
    tgt_meta     = meta.get("targets",   [{}])[0]
    control_mode = meta.get("control_mode", "both")

    return call_blend(
        pipeline      = pipeline,
        ref_bvh_bytes = ref_bvh_bytes,
        tgt_bvh_bytes = tgt_bvh_bytes,
        ref_meta      = ref_meta,
        tgt_meta      = tgt_meta,
        npy_cache_dir = os.path.join(_REPO_ROOT, ".bvh_cache"),
        control_mode  = control_mode,
        alpha         = alpha,
        blend_schedule = blend_schedule,
        seed          = seed,
        **({"ref_object_type": object_type} if object_type else {}),
    )
