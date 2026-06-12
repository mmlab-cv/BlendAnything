"""
Neural NLA Blending Server
==========================
FastAPI backend for motion blending and retargeting.

Endpoints
---------
GET  /health  → {"status": "online", "model_loaded": bool}
GET  /models  → available save-folder models and active model
POST /models/load → load a checkpoint by save-folder name
POST /blend   → multipart: reference_bvh, target_bvh[], metadata (JSON str)
              → returns blended BVH as a streaming file response
POST /blend/jobs → submits the same payload and returns a job ID
GET  /blend/jobs/{id} → status, phase, and normalized progress
GET  /blend/jobs/{id}/result → completed BVH result

Pipeline
--------
1. Receive reference + N target BVH bytes from the Blender plugin.
2. Apply NLA clip-editing instructions from the metadata (reverse, repeat).
3. Run neural blending via sample/blend.py (MoDiffAE diffusion model).
4. Return the blended BVH to Blender for import.

If neither an explicit MODEL_PATH nor the preferred catalog model
``truebones_attnpool`` is available, the server falls back to returning the
processed reference BVH unchanged so the plugin can still be tested end-to-end.

Run
---
    # Without a model (echo mode):
    uvicorn blendanything_server.app:app --host 0.0.0.0 --port 8000 --reload

    # With a model checkpoint:
    MODEL_PATH=/path/to/model010000.pt uvicorn blendanything_server.app:app --host 0.0.0.0 --port 8000

Environment variables
---------------------
MODEL_PATH      Optional explicit model####.pt checkpoint. It takes priority
                over catalog startup selection. The directory must contain a
                sibling args.json file.
MODEL_DEVICE    CUDA device index (default: 0).
MODEL_DATASET   Dataset key, "truebones" or "mixamo" (default: "truebones").
MODEL_COND_PATH Optional conditioning-file override. By default the server uses
                data/<model dataset>_cond.npy.
MODEL_SAVE_ROOT Model catalog root (default: neural_motion_blending/save).
                When MODEL_PATH is unset, truebones_attnpool is loaded from
                this catalog if available.

Dependencies
------------
    pip install fastapi uvicorn
    pip install git+https://github.com/inbar-2344/Motion.git  # for BVH processing
    cd neural_motion_blending && pip install -e .              # project package
"""

import io
import json
import logging
import math
import os
import sys
import tempfile
import threading
import time
import uuid
from typing import Callable, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Motion library (required for BVH I/O and preprocessing) ─────────────────

try:
    import BVH
    from Animation import Animation
    from Quaternions import Quaternions
    _MOTION_AVAILABLE = True
except ImportError:
    _MOTION_AVAILABLE = False

# ── Neural blending modules ──────────────────────────────────────────────────
# neural_motion_blending/ is a sibling of this package's parent directory.

_SERVER_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SERVER_ROOT)
_NMB_ROOT  = os.path.join(_REPO_ROOT, "neural_motion_blending")
for _p in (_REPO_ROOT, _NMB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_BLEND_IMPORT_ERROR = ""
try:
    from blendanything_server.model_bridge import (
        BlendPipeline,
        blend as neural_blend,
        load_pipeline,
        retarget_single as neural_retarget_single,
    )
    from blendanything_server.bvh_pipeline import build_user_conditioning
    _BLEND_MODULE_AVAILABLE = True
except ImportError as _e:
    _BLEND_MODULE_AVAILABLE = False
    _BLEND_IMPORT_ERROR = str(_e)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("neural_nla")

if not _MOTION_AVAILABLE:
    log.warning(
        "Motion library not found — BVH processing disabled. "
        "Install with: pip install git+https://github.com/inbar-2344/Motion.git"
    )

if not _BLEND_MODULE_AVAILABLE:
    log.warning("sample.blend import failed: %s", _BLEND_IMPORT_ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# Model — loaded once at startup
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_PATH    = os.environ.get("MODEL_PATH",    "")
_MODEL_DEVICE  = int(os.environ.get("MODEL_DEVICE",  "0"))
_MODEL_DATASET = os.environ.get("MODEL_DATASET", "truebones")
_MODEL_COND_PATH = os.environ.get("MODEL_COND_PATH", "")
_MODEL_SAVE_ROOT = os.environ.get(
    "MODEL_SAVE_ROOT",
    os.path.join(_NMB_ROOT, "save"),
)
_DEFAULT_MODEL_NAME = "truebones_attnpool"

_pipeline = None  # BlendPipeline, set at startup
_active_model_name = ""
_active_model_dataset = ""


def _available_models() -> dict:
    """Return folder name → checkpoint for valid save-directory models."""
    models = {}
    if not os.path.isdir(_MODEL_SAVE_ROOT):
        return models
    for folder_name in sorted(os.listdir(_MODEL_SAVE_ROOT)):
        folder = os.path.join(_MODEL_SAVE_ROOT, folder_name)
        if not os.path.isdir(folder):
            continue
        checkpoints = sorted(
            os.path.join(folder, filename)
            for filename in os.listdir(folder)
            if filename.startswith("model") and filename.endswith(".pt")
            and os.path.isfile(os.path.join(folder, filename))
        )
        if len(checkpoints) == 1:
            models[folder_name] = checkpoints[0]
        elif checkpoints:
            log.warning(
                "Ignoring model folder %s: expected one model*.pt, found %d",
                folder,
                len(checkpoints),
            )
    return models


def _model_name_for_path(path: str) -> str:
    absolute = os.path.abspath(path)
    for name, checkpoint in _available_models().items():
        if os.path.abspath(checkpoint) == absolute:
            return name
    return os.path.basename(os.path.dirname(absolute))


def _startup_model() -> tuple:
    """Return the startup (model name, checkpoint), honoring explicit config."""
    if _MODEL_PATH:
        return _model_name_for_path(_MODEL_PATH), _MODEL_PATH
    checkpoint = _available_models().get(_DEFAULT_MODEL_NAME, "")
    return (_DEFAULT_MODEL_NAME, checkpoint) if checkpoint else ("", "")


def _model_dataset(model_path: str) -> str:
    """Read the training dataset from args.json, defaulting to Truebones."""
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    try:
        with open(args_path, "r") as handle:
            dataset = str(json.load(handle).get("dataset") or "truebones").lower()
    except (OSError, json.JSONDecodeError):
        dataset = _MODEL_DATASET
    return dataset if dataset in {"truebones", "mixamo"} else _MODEL_DATASET


def _merge_conditioning_catalogs(cond_dict: dict) -> None:
    """Add runtime skeleton conditioning from every supported dataset."""
    for catalog_dataset in ("truebones", "mixamo"):
        catalog_path = os.path.join(
            _REPO_ROOT, "data", f"{catalog_dataset}_cond.npy"
        )
        if not os.path.isfile(catalog_path):
            continue
        catalog = np.load(catalog_path, allow_pickle=True).item()
        for skeleton, conditioning in catalog.items():
            cond_dict.setdefault(skeleton, conditioning)


def _load_model_checkpoint(model_name: str, model_path: str):
    """Build a pipeline for one catalog checkpoint."""
    if not _BLEND_MODULE_AVAILABLE:
        raise RuntimeError(f"Model bridge unavailable: {_BLEND_IMPORT_ERROR}")
    if not _MOTION_AVAILABLE:
        raise RuntimeError("Motion library unavailable.")
    dataset = _model_dataset(model_path)
    cond_path = (
        _MODEL_COND_PATH
        or os.path.join(_REPO_ROOT, "data", f"{dataset}_cond.npy")
    )
    pipeline = load_pipeline(
        model_path=model_path,
        device=_MODEL_DEVICE,
        dataset=dataset,
        cond_path=cond_path,
    )
    _merge_conditioning_catalogs(pipeline.cond_dict)
    return pipeline


def _try_load_model() -> None:
    """Attempt to load the neural model; log a warning on failure."""
    global _pipeline, _active_model_name, _active_model_dataset
    model_name, model_path = _startup_model()
    if not model_path:
        log.warning(
            "MODEL_PATH not set and preferred model %s was not found under %s "
            "— neural blending disabled.",
            _DEFAULT_MODEL_NAME,
            _MODEL_SAVE_ROOT,
        )
        return
    if not os.path.isfile(model_path):
        log.warning("MODEL_PATH=%s not found — neural blending disabled.", model_path)
        return
    try:
        _pipeline = _load_model_checkpoint(
            model_name,
            model_path,
        )
        _active_model_name = model_name
        _active_model_dataset = _model_dataset(model_path)
        log.info("Neural model loaded from %s", model_path)
    except Exception as exc:
        log.exception("Failed to load neural model: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Cache directory for processed BVH files
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_DIR = os.path.join(_REPO_ROOT, ".bvh_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

app = FastAPI(
    title="Neural NLA Blending Server",
    description="Receives BVH clips from Blender, applies clip edits, blends them.",
    version="2.0.0",
)

_jobs: dict = {}
_jobs_lock = threading.Lock()
_pipeline_lock = threading.Lock()


class ModelSelection(BaseModel):
    name: str


@app.on_event("startup")
async def startup_event() -> None:
    _try_load_model()


# ─────────────────────────────────────────────────────────────────────────────
# BVH I/O helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_bvh_bytes(bvh_bytes: bytes):
    """Write *bvh_bytes* to a temp file, load with BVH.load, return (anim, names, frametime)."""
    with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
        f.write(bvh_bytes)
        tmp = f.name
    try:
        return BVH.load(tmp)
    finally:
        os.unlink(tmp)


def _save_bvh_bytes(anim, names: list, frametime: float) -> bytes:
    """Serialise *anim* to BVH and return the file contents as bytes."""
    with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
        tmp = f.name
    try:
        BVH.save(tmp, anim, names=names, frametime=frametime)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# Clip editing  (reverse + repeat — applied before feeding the model)
# ─────────────────────────────────────────────────────────────────────────────


def apply_clip_edits(bvh_bytes: bytes, strip_meta: dict) -> bytes:
    """
    Apply NLA clip-editing instructions to a raw BVH clip.

    The incoming BVH already covers exactly the action_frame_start →
    action_frame_end range (exported by the Blender plugin).

      1. use_reverse  — flip the frame order of the clip
      2. repeat       — tile the clip *repeat* times
                        fractional value → trim the last pass to
                        floor(frac * n_frames) frames

    Returns the edited clip as BVH bytes.
    """
    anim, names, frametime = _load_bvh_bytes(bvh_bytes)
    n_frames = anim.shape[0]

    # ── 1. Reverse ────────────────────────────────────────────────────────────
    if strip_meta.get("use_reverse", False):
        anim = anim[::-1]

    # ── 2. Repeat (float) ─────────────────────────────────────────────────────
    repeat = float(strip_meta.get("repeat", 1.0))
    if repeat != 1.0 and n_frames > 0:
        full_copies  = int(math.floor(repeat))
        partial_frac = repeat - full_copies
        partial_n    = int(math.floor(partial_frac * n_frames))

        pieces = [anim] * full_copies
        if partial_n > 0:
            pieces.append(anim[:partial_n])

        if not pieces:
            log.warning("repeat=%.3f produced zero frames — returning original.", repeat)
        else:
            rot_qs    = np.concatenate([p.rotations.qs for p in pieces], axis=0)
            positions = np.concatenate([p.positions    for p in pieces], axis=0)
            anim = Animation(
                Quaternions(rot_qs), positions,
                anim.orients, anim.offsets, anim.parents,
            )

    return _save_bvh_bytes(anim, names, frametime)


def _cache_bvh(data: bytes, strip_name: str, role: str) -> None:
    """Write *data* to .bvh_cache/<strip_name>_<role>.bvh."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in strip_name)
    path = os.path.join(_CACHE_DIR, f"{safe}_{role}.bvh")
    try:
        with open(path, "wb") as f:
            f.write(data)
        log.info("Cached %s BVH → %s (%d bytes)", role, path, len(data))
    except OSError as exc:
        log.warning("Could not write cache file %s: %s", path, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Blend parameters extracted from the Blender plugin's strength metadata
# ─────────────────────────────────────────────────────────────────────────────


def _blend_params_from_meta(ref_meta: dict, tgt_meta: dict) -> dict:
    """
    Derive neural blending parameters from the strength envelope sent by
    the Blender plugin.

    Returns a dict with keys: alpha, blend_schedule, object_type (may be None).
    """
    ref_strength = ref_meta.get("strength", {})
    tgt_strength = tgt_meta.get("strength", {})
    profile = ref_strength.get("profile", "CONSTANT")
    ref_samples = np.asarray(ref_strength.get("samples", []), dtype=np.float32)
    tgt_samples = np.asarray(tgt_strength.get("samples", []), dtype=np.float32)

    # This scalar is only the fallback for the model's built-in schedule. The
    # model bridge applies the full per-frame normalized relative strengths.
    if ref_samples.size and tgt_samples.size:
        size = max(ref_samples.size, tgt_samples.size)
        positions = np.linspace(0.0, 1.0, size)
        ref_curve = np.interp(
            positions, np.linspace(0.0, 1.0, ref_samples.size), ref_samples
        )
        tgt_curve = np.interp(
            positions, np.linspace(0.0, 1.0, tgt_samples.size), tgt_samples
        )
        total = ref_curve + tgt_curve
        normalized = np.divide(
            tgt_curve,
            total,
            out=np.full_like(total, 0.5),
            where=total > 1e-6,
        )
        alpha = float(np.clip(np.mean(normalized), 0.0, 1.0))
    else:
        alpha = 0.5

    # Blend schedule: map plugin profiles → diffusion schedules
    schedule_map = {"SMOOTH": "ease", "LINEAR": "linear", "CONSTANT": "static"}
    blend_schedule = schedule_map.get(profile, "ease")

    return {
        "alpha":          alpha,
        "blend_schedule": blend_schedule,
        "object_type":    None,   # let call.py's _infer_object_type match against cond_dict keys
    }


def _skeleton_identity(motion_meta: dict) -> tuple:
    """Return a normalized (dataset, skeleton) identity for policy matching."""
    profile = str(motion_meta.get("skeleton_profile") or "").strip()
    dataset = str(motion_meta.get("skeleton_dataset") or "").strip()
    skeleton = str(motion_meta.get("object_type") or "").strip()

    if "::" in profile:
        profile_dataset, profile_skeleton = profile.split("::", 1)
        dataset = dataset or profile_dataset
        skeleton = profile_skeleton or skeleton
    elif profile:
        skeleton = profile
    elif "::" in skeleton:
        object_dataset, object_skeleton = skeleton.split("::", 1)
        dataset = dataset or object_dataset
        skeleton = object_skeleton

    return dataset.casefold(), skeleton.casefold()


def _should_use_ddim_inversion(meta: dict, ref_meta: dict, tgt_meta: dict) -> bool:
    """Resolve whether an active control matches the output/reference skeleton."""
    policy = str(meta.get("ddim_inversion_policy", "SAME_SKELETON")).upper()
    if policy == "ALWAYS":
        return True
    if policy not in {
        "SAME_SKELETON",
        "ON_SAME_SKELETON",
        "WHEN_SAME_SKELETON",
    }:
        return False

    ref_dataset, ref_skeleton = _skeleton_identity(ref_meta)
    tgt_dataset, tgt_skeleton = _skeleton_identity(tgt_meta)
    if not ref_skeleton:
        return False

    control_mode = str(meta.get("control_mode", "both")).lower()
    if control_mode in {"both", "ref"}:
        # The generated skeleton is the reference skeleton, so its own active
        # control is always eligible for inversion. Cross-skeleton targets are
        # represented by Gaussian noise in build_aligned_ddim_noise().
        return True
    if control_mode != "tgt" or ref_skeleton != tgt_skeleton:
        return False
    return not (ref_dataset and tgt_dataset) or ref_dataset == tgt_dataset


def _distribution_warnings(meta: dict) -> list:
    """Return compact, structured model-compatibility warnings."""
    if not _active_model_dataset:
        return []

    entries = []
    if meta.get("blend_mode") == "SINGLE_RETARGET":
        entries.append((
            "Reference",
            meta.get("destination_object_type", ""),
            meta.get("destination_dataset", ""),
        ))
        ref = meta.get("reference", {})
        entries.append((
            "Target 1",
            ref.get("object_type", ""),
            ref.get("skeleton_dataset", ""),
        ))
    else:
        ref = meta.get("reference", {})
        entries.append((
            "Reference",
            ref.get("object_type", ""),
            ref.get("skeleton_dataset", ""),
        ))
        for index, target in enumerate(meta.get("targets", []), start=1):
            entries.append((
                f"Target {index}",
                target.get("object_type", ""),
                target.get("skeleton_dataset", ""),
            ))

    warnings = []
    for role, skeleton, dataset in entries:
        dataset = dataset.lower()
        if not skeleton or not dataset:
            continue
        if dataset == "custom":
            kind = "estimated_statistics"
        elif dataset != _active_model_dataset:
            kind = "out_of_distribution"
        else:
            continue
        warnings.append({
            "role": role,
            "skeleton": skeleton,
            "dataset": dataset,
            "model_dataset": _active_model_dataset,
            "kind": kind,
        })
    return warnings


def _training_conditioning_catalog() -> dict:
    """Load the active model's training conditioning catalog."""
    dataset = _active_model_dataset or _MODEL_DATASET
    path = os.path.join(_REPO_ROOT, "data", f"{dataset}_cond.npy")
    return np.load(path, allow_pickle=True).item()


def _install_custom_conditioning(
    bvh_bytes: bytes,
    motion_meta: dict,
) -> str:
    """Install one request-scoped custom skeleton and return its object type."""
    if not motion_meta.get("custom_skeleton"):
        return ""
    object_type = str(motion_meta.get("object_type") or "")
    if not object_type.startswith("user::"):
        raise ValueError("Custom skeleton is missing a valid user::<name> identifier.")
    training_catalog = _training_conditioning_catalog()
    conditioning = build_user_conditioning(
        bvh_bytes,
        object_type=object_type,
        coordinate_space=motion_meta.get("coordinate_space", "RAW_TPOSE_RELATIVE"),
        face_joint_names=motion_meta.get("face_joints") or [],
        training_catalog=training_catalog,
    )
    max_joints = int(getattr(_pipeline.opt, "max_joints", 0) or 0)
    if max_joints and len(conditioning["parents"]) > max_joints:
        raise ValueError(
            f"Custom skeleton {object_type[6:]} has "
            f"{len(conditioning['parents'])} joints; this model supports "
            f"at most {max_joints}."
        )
    existing = _pipeline.cond_dict.get(object_type)
    if existing is not None:
        same_names = list(existing.get("joints_names", [])) == list(
            conditioning["joints_names"]
        )
        same_parents = np.array_equal(
            np.asarray(existing.get("parents", [])),
            np.asarray(conditioning["parents"]),
        )
        if not (same_names and same_parents):
            raise ValueError(
                f"Custom skeleton name {object_type[6:]!r} is used for "
                "incompatible BVH hierarchies. Give each skeleton a unique name."
            )
    else:
        _pipeline.cond_dict[object_type] = conditioning
    return object_type


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
async def health() -> dict:
    """Liveness probe used by the Blender plugin's 'Test Connection' button."""
    return {
        "status":       "online",
        "model_loaded": _pipeline is not None,
        "active_model": _active_model_name,
        "active_model_dataset": _active_model_dataset,
        "motion_lib":   _MOTION_AVAILABLE,
    }


@app.get("/models", tags=["Models"])
async def list_models() -> dict:
    """List loadable model folders and the currently active model."""
    return {
        "models": list(_available_models()),
        "active_model": _active_model_name,
        "active_model_dataset": _active_model_dataset,
        "model_loaded": _pipeline is not None,
    }


@app.post("/models/load", tags=["Models"])
async def select_model(selection: ModelSelection) -> dict:
    """Load one catalog model and atomically make it active for new jobs."""
    global _pipeline, _active_model_name, _active_model_dataset
    models = _available_models()
    model_path = models.get(selection.name)
    if model_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model folder: {selection.name}",
        )
    if _active_model_name == selection.name and _pipeline is not None:
        return {
            "status": "ready",
            "active_model": _active_model_name,
            "active_model_dataset": _active_model_dataset,
            "model_loaded": True,
        }

    with _pipeline_lock:
        try:
            pipeline = _load_model_checkpoint(selection.name, model_path)
        except Exception as exc:
            log.exception("Failed to load model %s: %s", selection.name, exc)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model {selection.name}: {exc}",
            ) from exc
        _pipeline = pipeline
        _active_model_name = selection.name
        _active_model_dataset = _model_dataset(model_path)

    log.info("Active model changed to %s (%s)", selection.name, model_path)
    return {
        "status": "ready",
        "active_model": _active_model_name,
        "active_model_dataset": _active_model_dataset,
        "model_loaded": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Blend
# ─────────────────────────────────────────────────────────────────────────────


def _set_job_progress(job_id: str, progress: float, phase: str, **extra) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.update(
            progress=max(0.0, min(1.0, float(progress))),
            phase=phase,
            updated_at=time.time(),
            **extra,
        )


def _job_snapshot(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Blend job not found.")
        return {
            key: value for key, value in job.items()
            if key not in {"result_bytes"}
        }


def _run_blend_pipeline(
    meta: dict,
    ref_bytes: bytes,
    target_bytes_list: list,
    progress: Optional[Callable[[float, str], None]] = None,
) -> tuple:
    """Run one blend request and return ``(result_bytes, filename)``."""
    report = progress or (lambda _value, _phase: None)
    ref_meta = meta.get("reference", {})
    targets_meta = meta.get("targets", [])
    single_retarget = meta.get("blend_mode") == "SINGLE_RETARGET"

    report(0.12, "Validating uploaded BVH data")
    if not _MOTION_AVAILABLE:
        log.warning("Motion library unavailable — returning reference BVH unchanged.")
        report(0.95, "Motion library unavailable; preparing fallback result")
        return ref_bytes, "blended_result.bvh"

    report(0.22, "Applying NLA clip edits")
    try:
        ref_processed = apply_clip_edits(ref_bytes, ref_meta)
        targets_processed = [
            apply_clip_edits(t_bytes, t_meta)
            for t_bytes, t_meta in zip(
                target_bytes_list,
                targets_meta + [{}] * len(target_bytes_list),
            )
        ]
    except Exception as exc:
        log.exception("BVH clip-editing failed: %s", exc)
        raise RuntimeError(f"BVH clip-editing error: {exc}") from exc

    report(0.34, "Caching processed motion clips")
    _cache_bvh(ref_processed, ref_meta.get("name", "ref"), "ref_processed")
    for i, (t_proc, t_meta) in enumerate(
        zip(targets_processed, targets_meta + [{}] * len(targets_processed))
    ):
        _cache_bvh(t_proc, t_meta.get("name", f"target_{i}"), f"target_{i}_processed")

    if _pipeline is not None and single_retarget:
        destination = meta.get("destination_object_type", "")
        if not destination:
            raise ValueError("destination_object_type is required for SINGLE_RETARGET.")
        destination_meta = {
            "skeleton_profile": meta.get("destination_skeleton_profile", ""),
            "skeleton_dataset": meta.get("destination_dataset", ""),
        }
        use_ddim_inversion = _should_use_ddim_inversion(
            meta,
            destination_meta,
            ref_meta,
        )
        report(0.45, "Preparing skeleton conditioning")
        custom_key = ""
        try:
            custom_key = _install_custom_conditioning(ref_processed, ref_meta)
            result_bytes = neural_retarget_single(
                pipeline=_pipeline,
                motion_bvh_bytes=ref_processed,
                motion_meta=ref_meta,
                destination_object_type=destination,
                npy_cache_dir=_CACHE_DIR,
                output_mode=meta.get("output_mode", "POSITIONS_IK"),
                ik_iterations=int(meta.get("ik_iterations", 150)),
                ddim_inversion=use_ddim_inversion,
                transition_slerp=use_ddim_inversion,
                progress_callback=lambda value, phase: report(
                    0.45 + 0.48 * value,
                    phase,
                ),
            )
        except Exception as exc:
            log.exception("Single-motion retargeting failed: %s", exc)
            raise RuntimeError(f"Single-motion retargeting failed: {exc}") from exc
        finally:
            if custom_key:
                _pipeline.cond_dict.pop(custom_key, None)
        report(0.95, "Caching retargeted result")
        _cache_bvh(
            result_bytes,
            f"{destination}_from_{ref_meta.get('name', 'motion')}",
            "retargeted",
        )
        return result_bytes, "retargeted_result.bvh"

    if _pipeline is not None and targets_processed:
        custom_keys = []
        try:
            tgt_meta_0 = targets_meta[0] if targets_meta else {}
            for motion_bytes, motion_meta in (
                (ref_processed, ref_meta),
                (targets_processed[0], tgt_meta_0),
            ):
                custom_key = _install_custom_conditioning(
                    motion_bytes, motion_meta
                )
                if custom_key:
                    custom_keys.append(custom_key)
            blend_params = _blend_params_from_meta(ref_meta, tgt_meta_0)
            use_ddim_inversion = _should_use_ddim_inversion(
                meta, ref_meta, tgt_meta_0
            )
            log.info(
                "Running neural blend | alpha=%.3f schedule=%s ddim_inversion=%s",
                blend_params["alpha"],
                blend_params["blend_schedule"],
                use_ddim_inversion,
            )
            report(0.45, "Preparing skeleton conditioning")
            result_bytes = neural_blend(
                pipeline=_pipeline,
                ref_bvh_bytes=ref_processed,
                tgt_bvh_bytes=targets_processed[0],
                ref_meta=ref_meta,
                tgt_meta=tgt_meta_0,
                npy_cache_dir=_CACHE_DIR,
                alpha=blend_params["alpha"],
                blend_schedule=blend_params["blend_schedule"],
                control_mode=meta.get("control_mode", "both"),
                ddim_inversion=use_ddim_inversion,
                transition_slerp=use_ddim_inversion,
                output_mode=meta.get("output_mode", "POSITIONS_IK"),
                ik_iterations=int(meta.get("ik_iterations", 150)),
                progress_callback=lambda value, phase: report(
                    0.45 + 0.48 * value,
                    phase,
                ),
            )
            report(0.95, "Caching blended result")
            _cache_bvh(result_bytes, ref_meta.get("name", "result"), "blended")
            log.info("Neural blend succeeded (%d bytes).", len(result_bytes))
            return result_bytes, "blended_result.bvh"
        except Exception as exc:
            log.exception("Neural blending failed — falling back to reference: %s", exc)
        finally:
            for custom_key in custom_keys:
                _pipeline.cond_dict.pop(custom_key, None)

    log.warning(
        "Neural model %s — returning processed reference BVH.",
        "failed" if _pipeline is not None else "not loaded",
    )
    report(0.92, "Preparing processed reference fallback")
    return ref_processed, "blended_result.bvh"


def _execute_blend_job(
    job_id: str,
    meta: dict,
    ref_bytes: bytes,
    target_bytes_list: list,
) -> None:
    try:
        _set_job_progress(job_id, 0.08, "Waiting for model worker", status="running")
        with _pipeline_lock:
            _set_job_progress(job_id, 0.10, "Model worker acquired")
            result_bytes, filename = _run_blend_pipeline(
                meta,
                ref_bytes,
                target_bytes_list,
                progress=lambda value, phase: _set_job_progress(job_id, value, phase),
            )
        _set_job_progress(
            job_id,
            1.0,
            "Result ready",
            status="done",
            result_bytes=result_bytes,
            filename=filename,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Blend job %s failed: %s", job_id, exc)
        _set_job_progress(
            job_id,
            1.0,
            "Failed",
            status="error",
            error_msg=str(exc),
        )


async def _read_blend_uploads(
    reference_bvh: UploadFile,
    target_bvh: List[UploadFile],
    metadata: str,
) -> tuple:
    try:
        meta: dict = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid metadata JSON: {exc}") from exc

    ref_bytes = await reference_bvh.read()
    if not ref_bytes:
        raise HTTPException(status_code=400, detail="reference_bvh file is empty.")
    target_bytes_list = []
    for i, upload in enumerate(target_bvh):
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"target_bvh[{i}] is empty.")
        target_bytes_list.append(data)
    return meta, ref_bytes, target_bytes_list


@app.post("/blend/jobs", tags=["Blend"])
async def create_blend_job(
    reference_bvh: UploadFile = File(..., description="Reference motion BVH"),
    target_bvh: List[UploadFile] = File(default=[], description="Target motion BVHs"),
    metadata: str = Form(..., description="JSON string of NLA strip metadata"),
) -> dict:
    """Submit a blend and return immediately with a progress-reporting job ID."""
    meta, ref_bytes, target_bytes_list = await _read_blend_uploads(
        reference_bvh, target_bvh, metadata
    )
    job_id = uuid.uuid4().hex
    now = time.time()
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.02,
            "phase": "Queued",
            "error_msg": "",
            "warnings": _distribution_warnings(meta),
            "filename": "",
            "created_at": now,
            "updated_at": now,
        }
    threading.Thread(
        target=_execute_blend_job,
        args=(job_id, meta, ref_bytes, target_bytes_list),
        daemon=True,
        name=f"BlendJob-{job_id[:8]}",
    ).start()
    return _job_snapshot(job_id)


@app.get("/blend/jobs/{job_id}", tags=["Blend"])
async def get_blend_job(job_id: str) -> dict:
    """Return current status, phase, and normalized progress for a blend job."""
    return _job_snapshot(job_id)


@app.get("/blend/jobs/{job_id}/result", tags=["Blend"])
async def get_blend_job_result(job_id: str):
    """Download a completed blend job's BVH result."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Blend job not found.")
        if job.get("status") == "error":
            raise HTTPException(status_code=500, detail=job.get("error_msg", "Blend failed."))
        if job.get("status") != "done":
            raise HTTPException(status_code=409, detail="Blend job is not complete.")
        result_bytes = job["result_bytes"]
        filename = job.get("filename") or "blended_result.bvh"
        del _jobs[job_id]
    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/blend", tags=["Blend"])
async def blend(
    reference_bvh: UploadFile       = File(..., description="Reference motion BVH"),
    target_bvh:    List[UploadFile] = File(default=[], description="Target motion BVHs"),
    metadata:      str              = Form(..., description="JSON string of NLA strip metadata"),
):
    """
    Accept a reference BVH + N target BVHs plus NLA metadata, apply clip
    edits to each, run neural blending, and return the result.

    When the neural model is unavailable the processed reference BVH is
    returned unchanged (echo mode for pipeline testing).
    """
    meta, ref_bytes, target_bytes_list = await _read_blend_uploads(
        reference_bvh, target_bvh, metadata
    )
    try:
        with _pipeline_lock:
            result_bytes, filename = _run_blend_pipeline(
                meta, ref_bytes, target_bytes_list
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "blendanything_server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # reload=True breaks the model singleton; use False in production
        log_level="info",
    )
