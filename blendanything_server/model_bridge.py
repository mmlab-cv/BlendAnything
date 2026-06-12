"""
call.py — Server-callable interface to the neural blending pipeline.

This module is the bridge between blendanything_server.app's /blend endpoint and the neural
diffusion model implemented in mix.py.  It exposes a single high-level
function, blend(), that accepts two BVH clips (as bytes) plus the NLA metadata
dict that Blender's neural_nla_blend.py plugin already sends, runs the full
neural blending pipeline, and returns the result as BVH bytes.

It also exposes load_pipeline() to initialise the model once at server startup
(expensive — T5 + diffusion model), and re-use the same state for every
subsequent request.

Typical usage from blendanything_server.app
-----------------------------
    from sample.call import load_pipeline, blend

    # Once at startup:
    pipeline = load_pipeline(model_path="save/.../model######.pt")

    # Per /blend request:
    result_bvh: bytes = blend(
        pipeline       = pipeline,
        ref_bvh_bytes  = ref_bytes,
        tgt_bvh_bytes  = tgt_bytes,
        ref_meta       = meta["reference"],   # from Blender metadata dict
        tgt_meta       = meta["targets"][0],  # first target
    )

BVH ↔ .npy conversion
-----------------------
Blender exports BVH; the model works on .npy arrays of shape (T, J, 13).
We convert via the Motion library's BVH.load() and a forward-kinematics pass
identical to what data_loaders/truebones uses, then convert back via
animation_from_positions() + BVH.save().

Blend parameters
-----------------
All blending knobs (control_mode, blend_schedule, alpha, overlap_length, …)
can be passed individually to blend().  Their defaults match the mix.py
command-line defaults so the server works out of the box without the caller
specifying every option.
"""

import io
import os
import sys
import math
import logging
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
from einops import rearrange

# ── Path setup ───────────────────────────────────────────────────────────────
# blendanything_server/ lives inside BlendAnything/; the neural package is a sibling dir.

_SERVER_DIR   = os.path.dirname(os.path.abspath(__file__))           # BlendAnything/blendanything_server/
_REPO_ROOT    = os.path.dirname(_SERVER_DIR)                         # BlendAnything/
_PKG_ROOT     = os.path.join(_REPO_ROOT, "neural_motion_blending")  # BlendAnything/neural_motion_blending/
for _p in (_PKG_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import BVH
from InverseKinematics import animation_from_positions

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
from utils.misc import char_from_npy_stem

from model.motion_diffusion_ae import ControlConfig, MoDiffAE
from model.modules.conditioners import T5Conditioner
from model.modules.util.geom import slerp

from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from data_loaders.tensors import truebones_batch_collate

# Re-use helpers already defined in mix.py (no copy-paste)
from sample.mix import (
    create_sample_in_batch,
    encode_joints_names,
    format_control_batch,
    format_gen_batch,
    format_alpha_tensor,
    trim_control_to_intersection,
    build_aligned_ddim_noise,
)

log = logging.getLogger("neural_nla.call")


# ─────────────────────────────────────────────────────────────────────────────
# Public data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BlendPipeline:
    """All heavy objects that should be loaded once and reused per request."""
    model:          object          # MoDiffAE (or AnyTop)
    diffusion:      object          # SpacedDiffusion
    t5_conditioner: T5Conditioner
    cond_dict:      dict            # per-skeleton mean/std/parents/offsets/…
    opt:            object          # get_opt() namespace
    args:           object          # argparse namespace used to build the model
    sampling_fun:   object          # diffusion.ddim_sample_loop or .p_sample_loop


@dataclass
class BlendRequest:
    """
    All parameters for a single blending call.

    The caller (server.py) fills this from the Blender metadata dict and
    whatever user-facing controls are exposed.  Every field has a sensible
    default matching the mix.py command-line defaults.
    """
    # ── Motion paths or npy arrays ───────────────────────────────────────────
    # Exactly one of (ref_npy_path / ref_motion) must be set; same for tgt.
    ref_npy_path:   str                         = ""
    tgt_npy_path:   str                         = ""

    # ── Object-type hints ────────────────────────────────────────────────────
    # If None, derived from the npy filename stem via char_from_npy_stem().
    ref_object_type: Optional[str]              = None
    tgt_object_type: Optional[str]              = None

    # ── NLA placement ────────────────────────────────────────────────────────
    overlap_length: int                         = -1    # negative → use frame_starts
    ref_frame_start: int                        = 0
    tgt_frame_start: int                        = 0
    ref_num_loops:  int                         = 1
    tgt_num_loops:  int                         = 1

    # ── Blend knobs ──────────────────────────────────────────────────────────
    control_mode:       str                     = "both"   # "ref" | "tgt" | "both"
    blend_schedule:     str                     = "ease"   # "static" | "linear" | "ease"
    alpha:              float                   = 0.5      # used when blend_schedule="static"
    ease_slope:         float                   = 1.0
    interpolation_mode: str                     = "lerp"   # "lerp" | "slerp"
    intersection_only:  bool                    = False

    # ── Diffusion sampling ───────────────────────────────────────────────────
    num_repetitions:    int                     = 1
    ddim_inversion:     bool                    = False
    transition_slerp:   bool                    = False
    seed:               int                     = 10

    # ── Output ───────────────────────────────────────────────────────────────
    out_dir:            str                     = ""   # leave "" to skip disk I/O
    render_video:       bool                    = False


# ─────────────────────────────────────────────────────────────────────────────
# Startup: load the model once
# ─────────────────────────────────────────────────────────────────────────────


def load_pipeline(
    model_path: str,
    device:     int  = 0,
    dataset:    str  = "truebones",
    sampler:    str  = "ddpm",
    cond_path:  str  = "",
    t5_name:    str  = "t5-base",
    seed:       int  = 10,
) -> BlendPipeline:
    """
    Initialise the neural blending pipeline.

    Call this once at server startup.  The returned BlendPipeline object is
    thread-safe for read-only inference (model.eval(), no gradient).

    Parameters
    ----------
    model_path : str
        Path to model######.pt checkpoint.  An args.json must exist in the
        same directory (created during training).
    device : int
        CUDA device index (0-based).
    dataset : str
        "truebones" or "mixamo".
    sampler : str
        "ddpm" or "ddim".
    cond_path : str
        Path to a custom cond.npy (leave "" to use the dataset default).
    t5_name : str
        T5 model variant — must match what was used during training.
    seed : int
        Global random seed (set once; subsequent calls may override per-request).
    """
    fixseed(seed)
    dist_util.setup_dist(device)

    # Build a minimal args namespace that satisfies create_model_and_diffusion
    # and parse_and_load_from_model.  We re-use parse_and_load_from_model
    # indirectly by calling create_model_and_diffusion_general_skeleton, which
    # only needs the model/diffusion fields that are loaded from args.json.
    from argparse import Namespace
    from utils.parser_util import (
        add_base_options, add_data_options, add_model_options,
        add_sampling_options, add_mix_options,
    )
    from argparse import ArgumentParser

    # Build a parser with ALL needed groups and parse an empty argv so every
    # argument gets its declared default.
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_sampling_options(parser)
    add_mix_options(parser)
    # parse_known_args so we don't crash on unknown flags from the server env
    args, _ = parser.parse_known_args([
        "--model_path", model_path,
        "--device",     str(device),
        "--dataset",    dataset,
        "--sampler",    sampler,
        "--t5_name",    t5_name,
        "--seed",       str(seed),
    ])
    args.model_path = model_path

    # Overwrite model/diffusion/dataset args from the saved args.json
    from utils.parser_util import extract_args, get_args_per_group_name
    args_to_overwrite = []
    for group_name in ["dataset", "model", "diffusion"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)
    args = extract_args(deepcopy(args), args_to_overwrite, model_path)
    # Restore the explicit overrides that must not be clobbered by args.json
    args.device  = device
    args.sampler = sampler
    args.t5_name = t5_name
    args.seed    = seed

    opt = get_opt(device, dataset=dataset)

    log.info("Loading cond_dict …")
    if cond_path:
        _cond_file = cond_path
    elif dataset == "truebones":
        _cond_file = os.path.join(_REPO_ROOT, "data", "truebones_cond.npy")
    else:
        _cond_file = opt.cond_file if os.path.isabs(opt.cond_file) else os.path.join(_PKG_ROOT, opt.cond_file)
    cond_dict = np.load(_cond_file, allow_pickle=True).item()

    log.info("Creating model and diffusion …")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)

    log.info("Loading checkpoint from %s …", model_path)
    state_dict = torch.load(model_path, map_location="cpu")
    load_model(model, state_dict)

    log.info("Loading T5 conditioner (%s) …", t5_name)
    t5_conditioner = T5Conditioner(
        name=t5_name, finetune=False, word_dropout=0.0,
        normalize_text=False, device="cuda",
    )

    model.to(dist_util.dev())
    model.eval()

    sampling_fun = (
        diffusion.ddim_sample_loop if sampler == "ddim"
        else diffusion.p_sample_loop
    )

    log.info("Pipeline ready.")
    return BlendPipeline(
        model=model,
        diffusion=diffusion,
        t5_conditioner=t5_conditioner,
        cond_dict=cond_dict,
        opt=opt,
        args=args,
        sampling_fun=sampling_fun,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BVH ↔ .npy helpers
# ─────────────────────────────────────────────────────────────────────────────


def _bvh_bytes_to_npy(bvh_bytes: bytes, object_type: str, cond_dict: dict) -> np.ndarray:
    """
    Convert raw BVH bytes to a (T, J, 13) float32 numpy array in the same
    feature space the model was trained on.

    The 13-feature representation is:
        [0]     global scale (1.0 placeholder)
        [1:7]   6D rotation (first two columns of the rotation matrix)
        [7:10]  root-relative joint velocity (ric positions delta)
        [10:13] padding / extra (zeros if not provided)

    We obtain it via recover_from_bvh_ric_np, which is the same function
    used in mix.py for denormalisation in reverse.

    NOTE: This is a *best-effort* conversion.  The truebones pre-processing
    pipeline (process_new_skeleton.py) computes per-skeleton statistics (mean,
    std, offsets, parents …) that must already exist in cond_dict.  If the
    incoming BVH skeleton does not match any known object_type the caller must
    supply a pre-populated cond_dict entry.
    """
    with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
        f.write(bvh_bytes)
        tmp = f.name
    try:
        anim, names, frametime = BVH.load(tmp)
    finally:
        os.unlink(tmp)

    # Forward kinematics → global joint positions (T, J, 3)
    global_positions = recover_from_bvh_ric_np(
        anim.positions[:, 0:1, :],   # root translation, shape (T,1,3)
    )
    # recover_from_bvh_ric_np expects the full (T, J, F) motion array
    # We reconstruct a minimal (T, J, 13) array from the BVH rotations.
    # Use the same approach as data preprocessing: rotations → 6D, then pack.
    from scipy.spatial.transform import Rotation as R_scipy

    n_frames, n_joints_bvh, _ = anim.rotations.qs.shape
    rots_mat = anim.rotations.qs   # quaternions (T, J, 4) w,x,y,z
    # Convert quaternions → rotation matrices (T, J, 3, 3)
    rots_scipy = R_scipy.from_quat(
        np.concatenate([rots_mat[..., 1:], rots_mat[..., :1]], axis=-1)
        .reshape(-1, 4)
    )
    rot_mat = rots_scipy.as_matrix().reshape(n_frames, n_joints_bvh, 3, 3)  # (T,J,3,3)

    # 6D rotation: first two columns of the rotation matrix → (T, J, 6)
    rot_6d = rot_mat[..., :2].reshape(n_frames, n_joints_bvh, 6)            # (T,J,6)

    # Root velocity from positions (T, J, 3) — approximate via finite differences
    root_pos = anim.positions[:, 0:1, :]           # (T,1,3)
    root_vel = np.zeros_like(root_pos)
    root_vel[1:] = root_pos[1:] - root_pos[:-1]
    vel = np.tile(root_vel, (1, n_joints_bvh, 1))  # (T,J,3)

    # Scale placeholder
    scale = np.ones((n_frames, n_joints_bvh, 1), dtype=np.float32)

    # Padding (3 zeros)
    pad = np.zeros((n_frames, n_joints_bvh, 3), dtype=np.float32)

    motion = np.concatenate([scale, rot_6d, vel, pad], axis=-1).astype(np.float32)
    return motion   # (T, J, 13)


def _npy_to_bvh_bytes(
    motion:      np.ndarray,   # (T, J, 13) — denormalised
    object_type: str,
    cond_dict:   dict,
    frametime:   float = 1.0 / 20.0,
) -> bytes:
    """
    Convert a denormalised (T, J, 13) motion array back to BVH bytes.

    Uses animation_from_positions via IK, which is the same path as mix.py.
    """
    parents  = cond_dict[object_type]["parents"]
    offsets  = cond_dict[object_type]["offsets"]
    joints_names = cond_dict[object_type]["joints_names"]

    global_positions = recover_from_bvh_ric_np(motion)
    out_anim, _1, _2 = animation_from_positions(
        positions=global_positions,
        parents=parents,
        offsets=offsets,
        iterations=150,
    )

    with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
        tmp = f.name
    try:
        BVH.save(tmp, out_anim, joints_names, frametime=frametime)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# Core blending function
# ─────────────────────────────────────────────────────────────────────────────


def blend_npy(
    pipeline: BlendPipeline,
    ref_npy_path: str,
    tgt_npy_path: str,
    *,
    control_mode:       str   = "both",
    blend_schedule:     str   = "ease",
    alpha:              float = 0.5,
    ease_slope:         float = 1.0,
    interpolation_mode: str   = "lerp",
    overlap_length:     int   = -1,
    ref_frame_start:    int   = 0,
    tgt_frame_start:    int   = 0,
    ref_num_loops:      int   = 1,
    tgt_num_loops:      int   = 1,
    intersection_only:  bool  = False,
    num_repetitions:    int   = 1,
    ddim_inversion:     bool  = False,
    transition_slerp:   bool  = False,
    seed:               int   = 10,
) -> List[np.ndarray]:
    """
    Run the neural blending model on two .npy motion files.

    Returns a list of (T, J, 13) float32 arrays, one per repetition
    (length = num_repetitions).  The arrays are **denormalised** (in the
    original physical unit space, not the normalised model space).

    This is the lowest-level public function; blend() and blend_bvh() build on
    top of it.

    Parameters
    ----------
    pipeline        : BlendPipeline returned by load_pipeline().
    ref_npy_path    : Path to the reference motion .npy file.
    tgt_npy_path    : Path to the target  motion .npy file.
    control_mode    : "ref" | "tgt" | "both"
    blend_schedule  : "static" | "linear" | "ease"
    alpha           : Static blending factor (used when blend_schedule="static").
    ease_slope      : Sharpness of the sine ease curve (1.0 = standard sine).
    interpolation_mode : "lerp" | "slerp" — latent interpolation mode.
    overlap_length  : Number of overlapping frames.  Negative → use frame_starts.
    ref_frame_start : Frame offset for the reference motion.
    tgt_frame_start : Frame offset for the target motion.
    ref_num_loops   : How many times to tile the reference motion.
    tgt_num_loops   : How many times to tile the target motion.
    intersection_only : If True, generate only the overlapping portion.
    num_repetitions : Number of independent samples to draw.
    ddim_inversion  : Invert clean motions to noise before sampling.
    transition_slerp : SLERP between inverted noises in the transition zone.
    seed            : Random seed.
    """
    fixseed(seed)

    model     = pipeline.model
    diffusion = pipeline.diffusion
    t5        = pipeline.t5_conditioner
    cond_dict = pipeline.cond_dict
    opt       = pipeline.opt
    args      = pipeline.args   # model/diffusion configuration; we shadow the
                                # mix-specific fields below

    # ── Build a lightweight args-like namespace for the mix helpers ───────────
    from argparse import Namespace
    mix_args = Namespace(
        **vars(args),                   # inherit model/diffusion settings
        control_mode       = control_mode,
        blend_schedule     = blend_schedule,
        alpha_values       = [alpha],
        ease_slope         = ease_slope,
        interpolation_mode = interpolation_mode,
        overlap_length     = overlap_length,
        ref_frame_start    = ref_frame_start,
        tgt_frame_start    = tgt_frame_start,
        ref_num_loops      = ref_num_loops,
        tgt_num_loops      = tgt_num_loops,
        intersection_only  = intersection_only,
        num_repetitions    = num_repetitions,
        ddim_inversion     = ddim_inversion,
        transition_slerp   = transition_slerp,
        seed               = seed,
        batch_size         = 1,
        temporal_window    = getattr(args, "temporal_window", 31),
    )

    # ── Load control motions and build the ControlConfig ─────────────────────
    ref_batch, tgt_batch, _filenames = _get_control_batches(
        ref_npy_path, tgt_npy_path, cond_dict, t5, mix_args, opt,
    )
    control = format_control_batch(ref_batch, tgt_batch, mix_args)

    if intersection_only:
        control = trim_control_to_intersection(control)

    batch, model_kwargs = format_gen_batch(control, cond_dict, t5, mix_args, opt)

    # ── Alpha / blend mask ────────────────────────────────────────────────────
    alpha_tensor, blend_mask = format_alpha_tensor(batch, control, alpha, mix_args)
    control.alpha      = alpha_tensor.to(control.x.device)
    control.blend_mask = blend_mask.to(control.x.device)

    # ── Optional DDIM inversion ───────────────────────────────────────────────
    noise = None
    if ddim_inversion:
        log.info("DDIM inversion …")
        control_cpy = deepcopy(control)
        img = rearrange(control_cpy.x, "b s ... -> (b s) ...")
        control_cpy.x = img.unsqueeze(1)

        if isinstance(model, MoDiffAE):
            inv_kwargs = {"y": control_cpy.y, "control": control_cpy}
        else:
            inv_kwargs = {"y": control_cpy.y}

        inverted_noise = diffusion.ddim_inversion_loop(
            model=model, img=img, model_kwargs=inv_kwargs,
            device=dist_util.dev(), clip_denoised=False, progress=True,
            skip_timesteps=0,
        )

        if control_mode == "ref":
            noise = inverted_noise[0].unsqueeze(0)[:, :, :, : batch.shape[-1]]
        elif control_mode == "tgt":
            noise = inverted_noise[1].unsqueeze(0)[:, :, :, : batch.shape[-1]]
        else:
            noise = build_aligned_ddim_noise(
                batch, control, inverted_noise[0], inverted_noise[1],
                alpha_tensor, transition_slerp,
            )

    # ── Wire control into model_kwargs ────────────────────────────────────────
    if isinstance(model, MoDiffAE):
        model_kwargs["control"] = control

    # ── Sampling ─────────────────────────────────────────────────────────────
    results = []
    for rep_i in range(num_repetitions):
        log.info("Sampling repetition %d / %d …", rep_i + 1, num_repetitions)
        sampling_fun = (
            diffusion.ddim_sample_loop if ddim_inversion else pipeline.sampling_fun
        )
        sample = sampling_fun(
            model,
            batch.shape,
            clip_denoised  = False,
            model_kwargs   = model_kwargs,
            skip_timesteps = 0,
            init_image     = None,
            progress       = True,
            dump_steps     = None,
            noise          = noise,
            const_noise    = False,
        )

        # ── Denormalise ───────────────────────────────────────────────────────
        for i, motion_tensor in enumerate(sample):
            n_joints    = model_kwargs["y"]["n_joints"][i].item()
            object_type = model_kwargs["y"]["object_type"][i]
            motion_t    = motion_tensor[:n_joints]
            mean        = cond_dict[object_type]["mean"][None, :]
            std         = cond_dict[object_type]["std"][None, :]
            motion_np   = motion_t.cpu().permute(2, 0, 1).numpy() * std + mean
            results.append(motion_np)

    return results   # list of (T, J_actual, 13) float32 arrays


def blend_bvh(
    pipeline:      BlendPipeline,
    ref_npy_path:  str,
    tgt_npy_path:  str,
    **blend_kwargs,
) -> bytes:
    """
    Convenience wrapper: run blend_npy() and return the **first** result as
    BVH bytes.

    All keyword arguments are forwarded to blend_npy().
    """
    motions = blend_npy(pipeline, ref_npy_path, tgt_npy_path, **blend_kwargs)
    if not motions:
        raise RuntimeError("blend_npy() returned no results.")

    motion      = motions[0]
    object_type = char_from_npy_stem(os.path.basename(ref_npy_path))
    fps         = pipeline.opt.fps
    return _npy_to_bvh_bytes(motion, object_type, pipeline.cond_dict,
                             frametime=1.0 / fps)


# ─────────────────────────────────────────────────────────────────────────────
# Server-facing entry point (BVH bytes → BVH bytes)
# ─────────────────────────────────────────────────────────────────────────────


def blend(
    pipeline:      BlendPipeline,
    ref_bvh_bytes: bytes,
    tgt_bvh_bytes: bytes,
    ref_meta:      dict,
    tgt_meta:      dict,
    *,
    npy_cache_dir: str   = ".bvh_cache",
    progress_callback: Optional[Callable[[float, str], None]] = None,
    **blend_kwargs,
) -> bytes:
    """
    Full server-to-server interface.

    Accepts the two processed BVH clips (already clip-edited by server.py's
    apply_clip_edits()) plus the NLA strip metadata dicts that Blender sends.
    Writes temporary .npy files, runs the neural blending model, and returns
    the result as BVH bytes.

    Parameters
    ----------
    pipeline       : BlendPipeline from load_pipeline().
    ref_bvh_bytes  : Reference clip BVH file contents (bytes).
    tgt_bvh_bytes  : Target clip BVH file contents (bytes).
    ref_meta       : Blender metadata dict for the reference strip
                     (keys: name, repeat, use_reverse, frame_start, frame_end,
                      action_frame_start, action_frame_end, strength, …).
    tgt_meta       : Same for the first target strip.
    npy_cache_dir  : Directory to cache intermediate .npy files for debugging.
    **blend_kwargs : Forwarded to blend_npy() — override any blending param.

    Returns
    -------
    bytes
        BVH file contents of the blended motion.
    """
    os.makedirs(npy_cache_dir, exist_ok=True)

    # ── Derive object types from strip names ──────────────────────────────────
    ref_name = ref_meta.get("name", "ref")
    tgt_name = tgt_meta.get("name", "tgt")

    ref_object_type = (
        blend_kwargs.pop("ref_object_type", None)
        or ref_meta.get("object_type")
        or _infer_object_type(ref_name, pipeline.cond_dict)
    )
    tgt_object_type = (
        blend_kwargs.pop("tgt_object_type", None)
        or tgt_meta.get("object_type")
        or _infer_object_type(tgt_name, pipeline.cond_dict)
    )

    log.info("Blend request | ref_type=%s  tgt_type=%s", ref_object_type, tgt_object_type)

    # ── Convert BVH bytes → (T, J, 13) npy arrays ────────────────────────────
    # bvh_to_npy handles joint-count mismatches (Blender strips end-sites on
    # export) by inserting missing leaf joints with identity rotations before
    # feature extraction — matching the full truebones skeleton in cond_dict.
    from blendanything_server.bvh_pipeline import bvh_to_npy
    ref_npy = bvh_to_npy(
        ref_bvh_bytes,
        pipeline.cond_dict[ref_object_type],
        coordinate_space=ref_meta.get("coordinate_space", "MODEL_Y_UP"),
        face_joint_names=ref_meta.get("face_joints"),
    )
    tgt_npy = bvh_to_npy(
        tgt_bvh_bytes,
        pipeline.cond_dict[tgt_object_type],
        coordinate_space=tgt_meta.get("coordinate_space", "MODEL_Y_UP"),
        face_joint_names=tgt_meta.get("face_joints"),
    )
    if ref_npy is None or tgt_npy is None:
        raise RuntimeError("BVH→npy conversion failed for one or both inputs.")

    log.info("Converted BVH→npy | ref (%d frames, %d joints), tgt (%d frames, %d joints)",
             *ref_npy.shape[:2], *tgt_npy.shape[:2])

    # ── Map Blender metadata → blend_npy() kwargs ─────────────────────────────
    blend_kwargs.setdefault("control_mode", "both")
    blend_kwargs = _meta_to_blend_kwargs(ref_meta, tgt_meta, **blend_kwargs)

    # ── Run the neural model (in-memory path, no npy filenames needed) ────────
    result_bvh = _blend_bvh_from_arrays(
        pipeline,
        ref_npy,
        tgt_npy,
        ref_object_type,
        tgt_object_type,
        progress_callback=progress_callback,
        **blend_kwargs,
    )

    log.info("Blend complete — result BVH: %d bytes", len(result_bvh))
    return result_bvh


def retarget_single(
    pipeline: BlendPipeline,
    motion_bvh_bytes: bytes,
    motion_meta: dict,
    destination_object_type: str,
    *,
    npy_cache_dir: str,
    output_mode: str = "POSITIONS_IK",
    ik_iterations: int = 150,
    ddim_inversion: bool = False,
    transition_slerp: bool = False,
    seed: int = 10,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bytes:
    """Apply one source motion to a known destination skeleton."""
    os.makedirs(npy_cache_dir, exist_ok=True)
    if destination_object_type not in pipeline.cond_dict:
        raise ValueError(
            f"Unknown destination skeleton profile: {destination_object_type!r}"
        )

    source_name = motion_meta.get("name", "motion")
    source_object_type = (
        motion_meta.get("object_type")
        or _infer_object_type(source_name, pipeline.cond_dict)
    )
    if source_object_type not in pipeline.cond_dict:
        raise ValueError(f"Unknown source skeleton profile: {source_object_type!r}")

    from blendanything_server.bvh_pipeline import bvh_to_npy
    source_npy = bvh_to_npy(
        motion_bvh_bytes,
        pipeline.cond_dict[source_object_type],
        coordinate_space=motion_meta.get("coordinate_space", "MODEL_Y_UP"),
        face_joint_names=motion_meta.get("face_joints"),
    )
    if source_npy is None:
        raise RuntimeError("BVH→npy conversion failed for the source motion.")

    # The network expects two controls. The destination side has no user
    # motion, so represent it with its neutral conditioning pose for the same
    # duration; control_mode=tgt supplies motion from the uploaded source.
    destination_cond = pipeline.cond_dict[destination_object_type]
    neutral_frame = np.asarray(destination_cond["tpos_first_frame"])
    destination_npy = np.repeat(neutral_frame[None], len(source_npy), axis=0)

    log.info(
        "Single-motion retarget | source=%s (%d frames) destination=%s",
        source_object_type,
        len(source_npy),
        destination_object_type,
    )
    result_bvh = _blend_bvh_from_arrays(
        pipeline,
        destination_npy,
        source_npy,
        destination_object_type,
        source_object_type,
        control_mode="tgt",
        blend_schedule="static",
        alpha=1.0,
        overlap_length=-1,
        ddim_inversion=ddim_inversion,
        transition_slerp=transition_slerp,
        output_mode=output_mode,
        ik_iterations=ik_iterations,
        seed=seed,
        progress_callback=progress_callback,
    )
    log.info("Single-motion retarget complete — result BVH: %d bytes", len(result_bvh))
    return result_bvh


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _blend_bvh_from_arrays(
    pipeline, ref_npy, tgt_npy, ref_object_type, tgt_object_type, **blend_kwargs
) -> bytes:
    """
    Run the full blending pipeline on in-memory npy arrays and return BVH bytes.
    Bypasses all filename/stem inference — object types are already known.
    """
    model     = pipeline.model
    diffusion = pipeline.diffusion
    t5        = pipeline.t5_conditioner
    cond_dict = pipeline.cond_dict
    opt       = pipeline.opt
    args      = pipeline.args

    control_mode       = blend_kwargs.get("control_mode", "both")
    blend_schedule     = blend_kwargs.get("blend_schedule", "ease")
    alpha              = blend_kwargs.get("alpha", 0.5)
    ease_slope         = blend_kwargs.get("ease_slope", 1.0)
    interpolation_mode = blend_kwargs.get("interpolation_mode", "lerp")
    overlap_length     = blend_kwargs.get("overlap_length", -1)
    ref_frame_start    = blend_kwargs.get("ref_frame_start", 0)
    tgt_frame_start    = blend_kwargs.get("tgt_frame_start", 0)
    ref_num_loops      = blend_kwargs.get("ref_num_loops", 1)
    tgt_num_loops      = blend_kwargs.get("tgt_num_loops", 1)
    intersection_only  = blend_kwargs.get("intersection_only", False)
    ddim_inversion     = blend_kwargs.get("ddim_inversion", False)
    transition_slerp   = blend_kwargs.get("transition_slerp", False)
    seed               = blend_kwargs.get("seed", 10)
    output_mode        = blend_kwargs.get("output_mode", "POSITIONS_IK")   # "POSITIONS_IK" | "ROTATIONS"
    ik_iterations      = blend_kwargs.get("ik_iterations", 150)
    ref_strength_samples = blend_kwargs.get("ref_strength_samples", [])
    tgt_strength_samples = blend_kwargs.get("tgt_strength_samples", [])
    progress_callback = blend_kwargs.get("progress_callback")

    phase_weights = [("semantic", 0.1)]
    if ddim_inversion:
        phase_weights.append(("inversion", 4.6))
    phase_weights.append(("sampling", 2.5))
    if output_mode == "POSITIONS_IK":
        phase_weights.append(("ik", 13.8))
    total_weight = sum(weight for _, weight in phase_weights)
    phase_bounds = {}
    cursor = 0.0
    for phase_name, weight in phase_weights:
        start = cursor / total_weight
        cursor += weight
        phase_bounds[phase_name] = (start, cursor / total_weight)

    def report_phase(phase_name: str, at_end: bool, label: str) -> None:
        if progress_callback is None:
            return
        bounds = phase_bounds[phase_name]
        progress_callback(bounds[1 if at_end else 0], label)

    fixseed(seed)

    from argparse import Namespace
    temporal_window = getattr(args, "temporal_window", 31)
    ref_n_frames = ref_npy.shape[0]
    tgt_n_frames = tgt_npy.shape[0]

    if overlap_length >= 0:
        _ref_frame_start = 0
        _tgt_frame_start = (ref_n_frames * ref_num_loops) - overlap_length
    else:
        _ref_frame_start = ref_frame_start
        _tgt_frame_start = tgt_frame_start

    _base = vars(args).copy()
    _base.update(
        control_mode=control_mode, blend_schedule=blend_schedule,
        alpha_values=[alpha], ease_slope=ease_slope,
        interpolation_mode=interpolation_mode, overlap_length=overlap_length,
        ref_frame_start=_ref_frame_start, tgt_frame_start=_tgt_frame_start,
        ref_num_loops=ref_num_loops, tgt_num_loops=tgt_num_loops,
        intersection_only=intersection_only, ddim_inversion=ddim_inversion,
        transition_slerp=transition_slerp, seed=seed,
        batch_size=1, temporal_window=temporal_window,
    )
    mix_args = Namespace(**_base)

    report_phase("semantic", False, "Semantic encoding")
    ref_sample = create_sample_in_batch(
        ref_object_type, cond_dict[ref_object_type],
        ref_n_frames, opt.max_joints, opt.feature_len, temporal_window,
        t5, motion=ref_npy, frame_start=_ref_frame_start, loop_times=ref_num_loops,
    )
    tgt_sample = create_sample_in_batch(
        tgt_object_type, cond_dict[tgt_object_type],
        tgt_n_frames, opt.max_joints, opt.feature_len, temporal_window,
        t5, motion=tgt_npy, frame_start=_tgt_frame_start, loop_times=tgt_num_loops,
    )

    control = format_control_batch([ref_sample], [tgt_sample], mix_args)
    if intersection_only:
        control = trim_control_to_intersection(control)
    batch, model_kwargs = format_gen_batch(control, cond_dict, t5, mix_args, opt)

    alpha_tensor, blend_mask = format_alpha_tensor(batch, control, alpha, mix_args)
    alpha_tensor = _normalized_relative_alpha(
        alpha_tensor,
        control,
        ref_strength_samples,
        tgt_strength_samples,
    )
    control.alpha      = alpha_tensor.to(control.x.device)
    control.blend_mask = blend_mask.to(control.x.device)
    report_phase("semantic", True, "Semantic encoding complete")

    noise = None
    if ddim_inversion:
        report_phase("inversion", False, "DDIM inversion")
        control_cpy = deepcopy(control)
        img = rearrange(control_cpy.x, "b s ... -> (b s) ...")
        control_cpy.x = img.unsqueeze(1)
        inv_kwargs = {"y": control_cpy.y, "control": control_cpy} if isinstance(model, MoDiffAE) else {"y": control_cpy.y}
        inverted_noise = diffusion.ddim_inversion_loop(
            model=model, img=img, model_kwargs=inv_kwargs,
            device=dist_util.dev(), clip_denoised=False, progress=True, skip_timesteps=0,
        )
        if control_mode == "ref":
            noise = inverted_noise[0].unsqueeze(0)[:, :, :, :batch.shape[-1]]
        elif control_mode == "tgt":
            noise = inverted_noise[1].unsqueeze(0)[:, :, :, :batch.shape[-1]]
        else:
            noise = build_aligned_ddim_noise(batch, control, inverted_noise[0], inverted_noise[1], alpha_tensor, transition_slerp)
        report_phase("inversion", True, "DDIM inversion complete")

    if isinstance(model, MoDiffAE):
        model_kwargs["control"] = control

    sampling_fun = (
        diffusion.ddim_sample_loop if ddim_inversion else pipeline.sampling_fun
    )
    sampler_name = "DDIM" if ddim_inversion else str(
        getattr(args, "sampler", "DDPM")
    ).upper()
    sampling_label = f"{sampler_name} sampling"
    report_phase(
        "sampling",
        False,
        sampling_label,
    )
    sample = sampling_fun(
        model, batch.shape, clip_denoised=False, model_kwargs=model_kwargs,
        skip_timesteps=0, init_image=None, progress=True,
        dump_steps=None, noise=noise, const_noise=False,
    )
    report_phase("sampling", True, f"{sampling_label} complete")

    # Denormalise
    motion_tensor = sample[0]
    n_joints    = model_kwargs["y"]["n_joints"][0].item()
    object_type = model_kwargs["y"]["object_type"][0]
    mean        = cond_dict[object_type]["mean"][None, :]
    std         = cond_dict[object_type]["std"][None, :]
    motion_np   = motion_tensor[:n_joints].cpu().permute(2, 0, 1).numpy() * std + mean  # (T, J, 13)

    parents      = cond_dict[object_type]["parents"]
    offsets      = cond_dict[object_type]["offsets"]
    joints_names = cond_dict[object_type]["joints_names"]

    if output_mode == "ROTATIONS":
        from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_rot_np
        _global_positions, out_anim = recover_from_bvh_rot_np(motion_np, parents, offsets)
    else:  # POSITIONS_IK (default)
        report_phase("ik", False, "IK post-processing")
        global_positions = recover_from_bvh_ric_np(motion_np)
        out_anim, _, _ = animation_from_positions(
            positions=global_positions, parents=parents, offsets=offsets,
            iterations=ik_iterations,
        )
        report_phase("ik", True, "IK post-processing complete")

    with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
        tmp = f.name
    try:
        BVH.save(tmp, out_anim, joints_names, frametime=1.0 / opt.fps)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp)


def _resample_strength(samples, length: int, *, device, dtype):
    """Resample a client strength envelope to one processed control length."""
    if length <= 0 or not samples:
        return None
    values = torch.as_tensor(samples, device=device, dtype=dtype).flatten()
    if values.numel() == 1:
        return values.expand(length)
    source = torch.linspace(0.0, 1.0, values.numel(), device=device, dtype=dtype)
    target = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)
    right = torch.searchsorted(source, target, right=True).clamp(1, values.numel() - 1)
    left = right - 1
    span = source[right] - source[left]
    fraction = torch.where(span > 0, (target - source[left]) / span, 0.0)
    return torch.lerp(values[left], values[right], fraction).clamp(0.0, 1.0)


def _normalized_relative_alpha(alpha_tensor, control, ref_samples, tgt_samples):
    """
    Convert independent strip strengths into the model's target alpha.

    In the shared control range, alpha = target / (reference + target).
    Frames where both strengths are zero retain the existing model schedule.
    """
    if not ref_samples or not tgt_samples:
        return alpha_tensor

    result = alpha_tensor.clone()
    starts = control.y["crop_start_ind"]
    lengths = control.y["lengths"]
    ref_start, tgt_start = int(starts[0]), int(starts[1])
    ref_length, tgt_length = int(lengths[0]), int(lengths[1])
    ref_end, tgt_end = ref_start + ref_length, tgt_start + tgt_length
    overlap_start, overlap_end = max(ref_start, tgt_start), min(ref_end, tgt_end)
    if overlap_end <= overlap_start:
        return result

    ref_curve = _resample_strength(
        ref_samples, ref_length, device=result.device, dtype=result.dtype
    )
    tgt_curve = _resample_strength(
        tgt_samples, tgt_length, device=result.device, dtype=result.dtype
    )
    ref_overlap = ref_curve[overlap_start - ref_start:overlap_end - ref_start]
    tgt_overlap = tgt_curve[overlap_start - tgt_start:overlap_end - tgt_start]
    total = ref_overlap + tgt_overlap
    fallback = result[overlap_start:overlap_end]
    normalized = torch.where(
        total > 1e-6,
        tgt_overlap / total.clamp_min(1e-6),
        fallback,
    )
    result[overlap_start:overlap_end] = normalized
    return result


def _get_control_batches(ref_motion_path, tgt_motion_path, cond_dict, t5_conditioner, args, opt):
    """
    Thin wrapper around mix.get_control_batches that accepts an args-like
    namespace instead of the full argparse Namespace produced by mix_args().
    """
    ref_motion = np.load(ref_motion_path)
    tgt_motion = np.load(tgt_motion_path)
    ref_object_type = char_from_npy_stem(os.path.basename(ref_motion_path))
    tgt_object_type = char_from_npy_stem(os.path.basename(tgt_motion_path))
    ref_n_frames = ref_motion.shape[0]
    tgt_n_frames = tgt_motion.shape[0]

    ref_frame_start = args.ref_frame_start if args.overlap_length < 0 else 0
    tgt_frame_start = (args.tgt_frame_start if args.overlap_length < 0
                       else (ref_n_frames * args.ref_num_loops) - args.overlap_length)

    ref_sample = create_sample_in_batch(
        ref_object_type, cond_dict[ref_object_type],
        ref_n_frames, opt.max_joints, opt.feature_len, args.temporal_window,
        t5_conditioner, motion=ref_motion,
        frame_start=ref_frame_start, loop_times=args.ref_num_loops,
    )
    tgt_sample = create_sample_in_batch(
        tgt_object_type, cond_dict[tgt_object_type],
        tgt_n_frames, opt.max_joints, opt.feature_len, args.temporal_window,
        t5_conditioner, motion=tgt_motion,
        frame_start=tgt_frame_start, loop_times=args.tgt_num_loops,
    )

    filenames = [(os.path.basename(ref_motion_path), os.path.basename(tgt_motion_path))]
    return [ref_sample], [tgt_sample], filenames


def _infer_object_type(strip_name: str, cond_dict: dict) -> str:
    """
    Try to derive the character/skeleton type from the strip name by matching
    against known cond_dict keys (which are the truebones character names).

    Falls back to the first key in cond_dict when no match is found.
    """
    # Direct match
    if strip_name in cond_dict:
        return strip_name
    # Prefix match (e.g. "Flamingo_Walk_001" → "Flamingo")
    for key in cond_dict:
        if strip_name.startswith(key):
            return key
    log.warning(
        "Could not infer object_type from strip name '%s'. "
        "Using '%s' as fallback.  Pass ref_object_type / tgt_object_type explicitly "
        "to suppress this warning.",
        strip_name, next(iter(cond_dict)),
    )
    return next(iter(cond_dict))


def _meta_to_blend_kwargs(ref_meta: dict, tgt_meta: dict, **overrides) -> dict:
    """
    Translate Blender NLA strip metadata into blend_npy() keyword arguments.

    The top-level metadata dict (wrapping ref_meta / tgt_meta) carries:
      - control_mode : "both" | "tgt"  (set by the Mode selector in the panel)
      - blend_mode   : "BLEND" | "RETARGET"  (human-readable alias)

    Per-strip metadata carries:
      - strength.profile    : "CONSTANT" | "LINEAR" | "SMOOTH"
      - strength.samples    : per-frame influence values
      - frame_start / frame_end : scene-time strip placement
      - repeat, use_reverse : playback modifiers (already applied by server.py)

    We map these to the overlap / blend_schedule / alpha / control_mode
    parameters that blend_npy() understands, unless the caller has already
    provided explicit overrides.

    Retarget mode (control_mode="tgt"):
      Timeline placement carries no meaning — the model is guided solely by
      the target motion style applied to the reference skeleton.  We therefore
      skip overlap / schedule inference and force control_mode="tgt".
    """
    kwargs: dict = {}

    # ── Control mode (from panel Mode selector) ───────────────────────────────
    # ref_meta is the per-strip dict; the top-level control_mode is passed via
    # overrides because blend() receives the full metadata and extracts it.
    control_mode = overrides.pop("control_mode", ref_meta.get("control_mode", "both"))
    kwargs["control_mode"] = control_mode

    ref_strength = ref_meta.get("strength", {})
    tgt_strength = tgt_meta.get("strength", {})
    kwargs["ref_strength_samples"] = ref_strength.get("samples", [])
    kwargs["tgt_strength_samples"] = tgt_strength.get("samples", [])

    if control_mode == "tgt":
        # Retarget: strip placement is irrelevant — use fixed defaults.
        kwargs.setdefault("blend_schedule", "static")
        kwargs.setdefault("alpha", 1.0)
        kwargs.setdefault("overlap_length", -1)
        kwargs.update(overrides)
        return kwargs

    # ── Blend mode: derive schedule / alpha / overlap from strength profiles ──

    if "blend_schedule" not in overrides:
        ref_profile = ref_strength.get("profile", "CONSTANT")
        if ref_profile == "CONSTANT":
            kwargs["blend_schedule"] = "static"
        elif ref_profile == "LINEAR":
            kwargs["blend_schedule"] = "linear"
        else:
            kwargs["blend_schedule"] = "ease"

    # Static alpha: use the tgt constant value (0 = pure ref, 1 = pure tgt).
    if "alpha" not in overrides and ref_strength.get("profile") == "CONSTANT":
        kwargs["alpha"] = float(tgt_strength.get("value", 0.5))

    # Preserve the NLA timeline order instead of assuming the reference starts
    # first. The model derives the overlap from these relative placements.
    if not {
        "overlap_length", "ref_frame_start", "tgt_frame_start"
    } & overrides.keys():
        ref_start = float(ref_meta.get("frame_start", 0.0))
        tgt_start = float(tgt_meta.get("frame_start", 0.0))
        origin = min(ref_start, tgt_start)
        kwargs["overlap_length"] = -1
        kwargs["ref_frame_start"] = int(round(ref_start - origin))
        kwargs["tgt_frame_start"] = int(round(tgt_start - origin))

    # Apply caller overrides last (they always win)
    kwargs.update(overrides)
    return kwargs
