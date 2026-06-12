"""
Neural NLA Blending — Blender Add-on
=====================================
Sidebar panel in the NLA Editor that exports selected NLA strips as BVH,
posts them to a neural blending server, and imports the result back into
the scene without blocking Blender's main thread.

Tested against Blender 4.5.4 LTS.

Installation
------------
1. Edit → Preferences → Add-ons → Install → select this file → Enable
2. Set the Server URL in the add-on preferences (default: http://localhost:8000)
3. Open the NLA Editor → select 2+ strips (active = reference) → N-panel → "Neural Blend"

Influence profiles
------------------
Each selected strip can carry one of three influence envelopes, written
directly to the strip's built-in NlaStrip.influence F-curve so Blender's
own "Show Strip Curves" overlay and NLA evaluator both see the result.

  CONSTANT  Flat value held for the full strip duration.

  LINEAR    Trapezoidal, same structure as SMOOTH but with strictly linear
            keyframe interpolation.  floor at the strip edges, peak over
            the central sustain region; fade_in / fade_out set the rise /
            fall fractions (0–1).

  SMOOTH    Trapezoidal envelope composed of up to four keyframes:
              ease-in   floor → peak  over the first  fade_in  fraction
              sustain   peak          across the middle (if fade_in+fade_out < 1)
              ease-out  peak → floor  over the last   fade_out fraction
            fade_in / fade_out are 0–1 fractions of strip duration (0 = instant,
            1 = full strip).  If their sum exceeds 1 they are scaled down
            proportionally.

            Additional Bezier controls (SMOOTH only):
              Floor      Influence at the strip boundaries instead of hard 0.
              Shape      Interpolation type for the outer keyframes — Smooth
                         (Auto Bezier), Ease In, Ease Out, or Ease In/Out.
              Overshoot  Handle tangent offset when Shape is Smooth: positive
                         values add anticipation/bounce beyond the peak;
                         negative values dip below the floor.

Quick presets
-------------
  ↗ In      Fade in over the full strip  (fade_in=1, fade_out=0)
  ↘ Out     Fade out over the full strip (fade_in=0, fade_out=1)
  ~ Ease    25 % rise / sustain / 25 % fall (fade_in=0.25, fade_out=0.25)
  1.0 / 0.5 / 0.0  Set the peak amplitude for the active profile.

Presets respect the current profile: CONSTANT is upgraded to LINEAR for
directional/ease presets; LINEAR and SMOOTH both support all three.  All
property changes update the F-curve in real time via Blender property update
callbacks; a 20 Hz timer also catches moves missed by the depsgraph handler.

Sync
----
A depsgraph_update_post handler detects when a strip is shifted or scaled
in the NLA editor and re-applies the stored profile so the influence
keyframes always stay anchored to the strip boundaries.
"""

from . import bl_info

_ADDON_VERSION = ".".join(str(part) for part in bl_info["version"])

import os
import re
import shutil
import tempfile
import threading
import time
from datetime import datetime
from typing import Optional, Tuple

import bpy

from .network import (
    active_model as _active_server_model,
    fetch_models as _fetch_server_models,
    load_model as _load_server_model,
    model_enum_items as _model_enum_items,
    read_state as _read_state,
    redraw_nla_editors as _redraw_nla_editors,
    send_request as _thread_send_request,
    set_state as _set_state,
)
from .profiles import (
    display_name as _profile_display_name,
    enum_items as _skeleton_profile_enum_items,
    load_profiles as _load_skeleton_profiles,
    match_source as _match_source_profile,
    profile_parts as _profile_parts,
    qualify_profile_id as _qualify_profile_id,
)

_PROJECT_URL = "https://mmlab-cv.github.io/BlendAnything/"
_MAINTAINER_URL = "https://github.com/LuCazzola"
_STRENGTH_PLOT_IMAGE = ".BlendAnything Relative Strength Preview"
_strength_plot_signature = None


class BlendAnythingPreferences(bpy.types.AddonPreferences):
    """Project information shown in Blender's Add-ons preferences."""

    bl_idname = "blendanything_client"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        box = layout.box()
        col = box.column(align=False)
        col.label(text="About BlendAnything", icon="INFO")
        col.separator(factor=0.35)
        col.label(text="BlendAnything: A Blender Plugin for Cross-Topology Motion Blending")
        col.label(text="University of Trento")
        col.label(text="Plugin maintainer: Luca Cazzola")
        col.label(text="Work coauthors: Giulia Martinelli and Nicola Conci")
        col.label(text="Accepted to SIGGRAPH Posters 2026")
        col.separator(factor=0.5)
        row = col.row(align=True)
        op = row.operator("wm.url_open", text="Project Website", icon="URL")
        op.url = _PROJECT_URL
        op = row.operator("wm.url_open", text="Maintainer on GitHub", icon="URL")
        op.url = _MAINTAINER_URL

# ─────────────────────────────────────────────────────────────────────────────
# Dependency check (requests is not bundled with Blender)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import requests as _requests  # noqa: F401

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ─────────────────────────────────────────────────────────────────────────────
# Strip Strength — data model
#
# bpy.types.NlaStrip is a non-ID struct.  Blender does not provide writable
# storage for bpy.props registered on non-ID types (reading returns the
# declared default; writing raises AttributeError: read-only).
# PointerProperty to a custom PropertyGroup also silently fails on NlaStrip.
#
# Solution: store per-strip data in a CollectionProperty on bpy.types.Scene,
# keyed by strip name.  This is the standard Blender idiom for associating
# custom data with non-ID structs.
#
# Three profiles
# ──────────────
#  CONSTANT  single value held throughout the strip
#  LINEAR    floor … peak … floor, trapezoidal with fade_in / fade_out fractions, LINEAR interpolation
#  SMOOTH    trapezoidal envelope written as a native NlaStrip influence F-curve;
#              ease-in  over first fade_in  fraction of duration (BEZIER, 0→peak)
#              sustain  at peak_value across the middle
#              ease-out over last  fade_out fraction of duration (BEZIER, peak→0)
# ─────────────────────────────────────────────────────────────────────────────


def _on_strength_update(self, context) -> None:
    """Property update callback: re-apply influence F-curve immediately on any change."""
    strip = find_strip_by_name(self.strip_name)
    if strip is not None:
        _apply_profile_to_strip(strip, self)


class NeuralArmatureReference(bpy.types.PropertyGroup):
    """
    Maps an armature object → its original source BVH.

    Both reference and target strips pass through the model, often on different
    armatures (e.g. Elephant vs Skunk). Each armature keeps its original BVH so
    the plugin can upload model-native channels instead of Blender's lossy BVH
    re-export. Keyed by the owning object's name.
    """

    object_name: bpy.props.StringProperty(  # type: ignore[assignment]
        name="Armature Name",
        description="Name of the armature object this reference BVH belongs to",
    )
    reference_bvh: bpy.props.StringProperty(  # type: ignore[assignment]
        name="Reference BVH",
        description="Original source .bvh with the full, correctly-named joint hierarchy",
        subtype="FILE_PATH",
        default="",
    )


def _skeleton_profile_items(self, context):
    return _skeleton_profile_enum_items()


def _destination_profile_items(self, context):
    return [
        item for item in _skeleton_profile_items(self, context)
        if item is not None and item[0] != "CUSTOM"
    ]


def _on_skeleton_profile_update(self, context) -> None:
    if self.profile_id == "CUSTOM":
        return
    profile = _load_skeleton_profiles().get(_qualify_profile_id(self.profile_id), {})
    face = profile.get("face_joints", [])
    fields = ("face_right", "face_left", "face_upper_right", "face_upper_left")
    for field, value in zip(fields, list(face) + [""] * (4 - len(face))):
        setattr(self, field, value)


def _on_input_mode_update(self, context) -> None:
    action = bpy.data.actions.get(self.action_name)
    if action is not None:
        action[_ACTION_INPUT_MODE_PROP] = self.input_mode


class NeuralActionSource(bpy.types.PropertyGroup):
    """Persistent action-to-source-BVH association."""

    action_name: bpy.props.StringProperty(  # type: ignore[assignment]
        name="Action Name",
        description="Name of the Blender action created from this BVH",
    )
    source_bvh: bpy.props.StringProperty(  # type: ignore[assignment]
        name="Source BVH",
        description="Original model-native BVH backing this action",
        subtype="FILE_PATH",
        default="",
    )
    input_mode: bpy.props.EnumProperty(  # type: ignore[assignment]
        name="Input Mode",
        description="How this action's source BVH should be converted to model features",
        items=[
            ("MODEL_PROCESSED", "Model Processed",
             "The BVH is already in the model/Truebones processed convention"),
            ("RAW_TPOSE_RELATIVE", "Raw T-pose Relative",
             "The BVH has a custom world convention but local rotations are T-pose-relative"),
        ],
        default="MODEL_PROCESSED",
        update=_on_input_mode_update,
    )
    profile_id: bpy.props.EnumProperty(  # type: ignore[assignment]
        name="Character",
        description="Known model skeleton/profile matched from conditioning data",
        items=_skeleton_profile_items,
        update=_on_skeleton_profile_update,
    )
    face_right: bpy.props.StringProperty(name="Lower-R", default="")  # type: ignore[assignment]
    face_left: bpy.props.StringProperty(name="Lower-L", default="")  # type: ignore[assignment]
    face_upper_right: bpy.props.StringProperty(name="Upper-R", default="")  # type: ignore[assignment]
    face_upper_left: bpy.props.StringProperty(name="Upper-L", default="")  # type: ignore[assignment]


class NeuralStripSkeletonEntry(bpy.types.PropertyGroup):
    """Skeleton conditioning selected for one NLA strip."""

    strip_name: bpy.props.StringProperty(  # type: ignore[assignment]
        name="Strip Name",
        description="Name of the NLA strip this skeleton configuration belongs to",
    )
    profile_id: bpy.props.EnumProperty(  # type: ignore[assignment]
        name="Skeleton",
        description="Dataset skeleton used to condition this strip",
        items=_skeleton_profile_items,
        update=_on_skeleton_profile_update,
    )
    custom_name: bpy.props.StringProperty(  # type: ignore[assignment]
        name="Skeleton Name",
        description="Stable name for this user-provided skeleton",
        default="",
    )
    face_right: bpy.props.StringProperty(name="Lower-R", default="")  # type: ignore[assignment]
    face_left: bpy.props.StringProperty(name="Lower-L", default="")  # type: ignore[assignment]
    face_upper_right: bpy.props.StringProperty(name="Upper-R", default="")  # type: ignore[assignment]
    face_upper_left: bpy.props.StringProperty(name="Upper-L", default="")  # type: ignore[assignment]


_ACTION_SOURCE_PROP = "blendanything_source_bvh"
_ACTION_PROFILE_PROP = "blendanything_skeleton_profile"
_ACTION_INPUT_MODE_PROP = "blendanything_input_mode"
def _get_armature_reference(scene, object_name: str) -> str:
    """Return the reference BVH path stored for *object_name* (or '')."""
    for item in scene.neural_nla_armature_refs:
        if item.object_name == object_name:
            return item.reference_bvh
    return ""


def _set_armature_reference(scene, object_name: str, path: str) -> None:
    """Store/overwrite the reference BVH for *object_name*."""
    for item in scene.neural_nla_armature_refs:
        if item.object_name == object_name:
            item.reference_bvh = path
            return
    item = scene.neural_nla_armature_refs.add()
    item.object_name = object_name
    item.reference_bvh = path


def _get_action_source(scene, action) -> str:
    """Return an action's explicit source path, or an empty string."""
    if action is None:
        return ""
    custom = action.get(_ACTION_SOURCE_PROP, "")
    if custom:
        return str(custom)
    for item in scene.neural_nla_action_sources:
        if item.action_name == action.name:
            return item.source_bvh
    return ""


def _get_action_source_entry(scene, action):
    if action is None:
        return None
    for item in scene.neural_nla_action_sources:
        if item.action_name == action.name:
            return item
    return None


def _migrate_profile_id(item) -> None:
    """Replace a pre-dataset enum value without touching adjacent settings."""
    raw_profile = item.get("profile_id")
    if not isinstance(raw_profile, str):
        return
    qualified = _qualify_profile_id(raw_profile)
    if qualified not in _load_skeleton_profiles():
        return
    face_fields = (
        "face_right",
        "face_left",
        "face_upper_right",
        "face_upper_left",
    )
    custom_face = {
        field: getattr(item, field)
        for field in face_fields
        if hasattr(item, field)
    }
    item.property_unset("profile_id")
    item.profile_id = qualified
    for field, value in custom_face.items():
        setattr(item, field, value)


def _ensure_action_source_entry(scene, action):
    item = _get_action_source_entry(scene, action)
    if item is None and action is not None:
        item = scene.neural_nla_action_sources.add()
        item.action_name = action.name
        item.source_bvh = str(action.get(_ACTION_SOURCE_PROP, ""))
        stored_profile = _qualify_profile_id(
            str(action.get(_ACTION_PROFILE_PROP, ""))
        )
        if stored_profile in _load_skeleton_profiles():
            item.profile_id = stored_profile
        mode = str(action.get(_ACTION_INPUT_MODE_PROP, ""))
        if mode in {"MODEL_PROCESSED", "RAW_TPOSE_RELATIVE"}:
            item.input_mode = mode
    if item is not None:
        _migrate_profile_id(item)
        _sync_action_source_profile(action, item)
    return item


def _sync_action_source_profile(action, item) -> None:
    """Fill a missing profile from the action's known source BVH."""
    if action is None or item.profile_id != "NONE":
        return
    source = item.source_bvh or str(action.get(_ACTION_SOURCE_PROP, ""))
    source = bpy.path.abspath(source) if source else ""
    if not source or not os.path.isfile(source):
        return
    profile_id, _, _ = _match_source_profile(source)
    if profile_id:
        action[_ACTION_PROFILE_PROP] = profile_id
        _apply_profile_defaults(item, profile_id)


def _apply_profile_defaults(item, profile_id: str) -> None:
    profile_id = _qualify_profile_id(profile_id)
    profile = _load_skeleton_profiles().get(profile_id, {})
    item.profile_id = profile_id if profile_id else "NONE"
    face = profile.get("face_joints", [])
    fields = ("face_right", "face_left", "face_upper_right", "face_upper_left")
    for field, value in zip(fields, list(face) + [""] * (4 - len(face))):
        setattr(item, field, value)


def _get_strip_skeleton(scene, strip_name: str):
    for item in scene.neural_nla_strip_skeletons:
        if item.strip_name == strip_name:
            return item
    return None


def _ensure_strip_skeleton(scene, strip):
    """Return per-strip skeleton data, migrating or detecting defaults."""
    item = _get_strip_skeleton(scene, strip.name)
    if item is None:
        item = scene.neural_nla_strip_skeletons.add()
        item.strip_name = strip.name

        action_entry = _get_action_source_entry(scene, strip.action)
        if action_entry is not None and action_entry.profile_id != "NONE":
            _apply_profile_defaults(item, action_entry.profile_id)

    _migrate_profile_id(item)

    if item.profile_id == "NONE" and strip.action is not None:
        stored_profile = _qualify_profile_id(
            str(strip.action.get(_ACTION_PROFILE_PROP, ""))
        )
        if stored_profile in _load_skeleton_profiles():
            _apply_profile_defaults(item, stored_profile)
        else:
            source = _get_action_source(scene, strip.action)
            source = bpy.path.abspath(source) if source else ""
            if source and os.path.isfile(source):
                profile_id, _, _ = _match_source_profile(source)
                if profile_id:
                    _apply_profile_defaults(item, profile_id)
    return item


def _set_action_source(
    scene,
    action,
    path: str,
    *,
    profile_id: str = "",
) -> None:
    """Persist a source path on the Action ID and in the scene registry."""
    if action is None:
        return
    action[_ACTION_SOURCE_PROP] = path
    item = _ensure_action_source_entry(scene, action)
    item.source_bvh = path
    if (not profile_id or profile_id == "NONE") and item.profile_id == "NONE":
        profile_id, _, _ = _match_source_profile(path)
        if not profile_id:
            item.input_mode = "RAW_TPOSE_RELATIVE"
    if profile_id:
        action[_ACTION_PROFILE_PROP] = profile_id
        _apply_profile_defaults(item, profile_id)
    action[_ACTION_INPUT_MODE_PROP] = item.input_mode


def _source_bvh_info(path: str) -> Tuple[int, set]:
    """Return (frame_count, named hierarchy joints) from a BVH file."""
    frame_count = -1
    names = set()
    with open(path, "r") as handle:
        for raw in handle:
            line = raw.strip()
            match = re.match(r"(?:ROOT|JOINT)\s+(\S+)", line)
            if match:
                names.add(match.group(1))
            elif line.startswith("End Site"):
                match = re.search(r"#name:\s*(\S+)", line)
                if match:
                    names.add(match.group(1))
            elif line.startswith("Frames:"):
                frame_count = int(line.split(":", 1)[1].strip())
                break
    return frame_count, names


def _validate_source_bvh(path: str, action, owner=None) -> Tuple[bool, str]:
    """Validate that *path* can back the given imported BVH action."""
    if not path or not os.path.isfile(path):
        return False, "file does not exist"
    try:
        frame_count, source_names = _source_bvh_info(path)
    except (OSError, ValueError) as exc:
        return False, str(exc)

    expected_frames = int(round(action.frame_range[1] - action.frame_range[0])) + 1
    if frame_count < expected_frames:
        return False, f"source has {frame_count} frames; action needs {expected_frames}"

    if owner is not None and owner.type == "ARMATURE":
        blender_names = {bone.name for bone in owner.data.bones}
        missing_non_end = [
            name for name in blender_names
            if not name.endswith("_end_site") and name not in source_names
        ]
        if missing_non_end:
            return False, f"source is missing Blender bones: {missing_non_end[:4]}"

    return True, ""


def _resolve_strip_source(scene, obj, strip, *, persist_auto=False) -> str:
    """Resolve explicitly tracked action or legacy armature source provenance."""
    action = strip.action
    explicit = _get_action_source(scene, action)
    explicit_path = bpy.path.abspath(explicit) if explicit else ""
    valid, _ = _validate_source_bvh(explicit_path, action, obj)
    if valid:
        if persist_auto:
            _set_action_source(scene, action, explicit_path)
        return explicit_path

    legacy = _get_armature_reference(scene, obj.name)
    legacy_path = bpy.path.abspath(legacy) if legacy else ""
    valid, _ = _validate_source_bvh(legacy_path, action, obj)
    if valid:
        if persist_auto:
            _set_action_source(scene, action, legacy_path)
        return legacy_path
    return ""


class NeuralStripStrengthEntry(bpy.types.PropertyGroup):
    """
    Per-strip strength envelope stored in Scene.neural_nla_strengths.
    Identified by strip_name.
    """

    strip_name: bpy.props.StringProperty(  # type: ignore[assignment]
        name="Strip Name",
        description="Name of the NlaStrip this entry belongs to",
    )

    profile: bpy.props.EnumProperty(  # type: ignore[assignment]
        name="Profile",
        description="Shape of the strength curve over the strip's duration",
        items=[
            ("CONSTANT", "Constant",
             "Uniform influence for the full duration", "SNAP_ON", 0),
            ("LINEAR",   "Linear",
             "Linearly ramp from Start to End value", "IPO_LINEAR", 1),
            ("SMOOTH",   "Smooth",
             "Ease-in → sustain at peak → ease-out trapezoidal envelope",
             "IPO_EASE_IN_OUT", 2),
        ],
        default="CONSTANT",
        update=_on_strength_update,
    )

    # CONSTANT
    value: bpy.props.FloatProperty(  # type: ignore[assignment]
        name="Value",
        description="Constant influence (0 = none, 1 = full)",
        min=0.0, max=1.0, default=1.0, precision=3, subtype="FACTOR",
        update=_on_strength_update,
    )

    # SMOOTH and LINEAR — shared trapezoidal parameters
    peak_value: bpy.props.FloatProperty(  # type: ignore[assignment]
        name="Peak", description="Maximum influence reached after the fade-in",
        min=0.0, max=1.0, default=1.0, precision=3, subtype="FACTOR",
        update=_on_strength_update,
    )
    value_floor: bpy.props.FloatProperty(  # type: ignore[assignment]
        name="Floor",
        description="Influence at strip boundaries — replaces the hard zero at fade edges",
        min=0.0, max=1.0, default=0.0, precision=3, subtype="FACTOR",
        update=_on_strength_update,
    )
    fade_in: bpy.props.FloatProperty(  # type: ignore[assignment]
        name="Fade In",
        description="Fraction of strip duration for the ease-in (0 = instant, 1 = full strip)",
        min=0.0, max=1.0, default=0.25, precision=2, subtype="FACTOR",
        update=_on_strength_update,
    )
    fade_out: bpy.props.FloatProperty(  # type: ignore[assignment]
        name="Fade Out",
        description="Fraction of strip duration for the ease-out (0 = instant, 1 = full strip)",
        min=0.0, max=1.0, default=0.25, precision=2, subtype="FACTOR",
        update=_on_strength_update,
    )
    ease_shape: bpy.props.EnumProperty(  # type: ignore[assignment]
        name="Shape",
        description="Interpolation type applied to the outer (fade) keyframes",
        items=[
            ("BEZIER",      "Smooth",      "Symmetric S-curve — Auto Bezier handles"),
            ("EASE_IN",     "Ease In",     "Starts slow, accelerates toward the peak"),
            ("EASE_OUT",    "Ease Out",    "Starts fast, decelerates toward floor"),
            ("EASE_IN_OUT", "Ease In/Out", "Blender built-in symmetric easing function"),
        ],
        default="BEZIER",
        update=_on_strength_update,
    )
    # Last strip boundaries at which the F-curve was applied.
    # Used by the depsgraph sync handler to detect strip movement without
    # relying on reading keyframe positions (which can be in flux).
    last_frame_start: bpy.props.FloatProperty(default=-1e9)  # type: ignore[assignment]
    last_frame_end:   bpy.props.FloatProperty(default=-1e9)  # type: ignore[assignment]

    overshoot: bpy.props.FloatProperty(  # type: ignore[assignment]
        name="Overshoot",
        description=(
            "Bezier handle tangent offset at strip boundaries. "
            "Positive = bounce/anticipation beyond peak; negative = dip below floor. "
            "Only active when Shape is Smooth (Auto Bezier)."
        ),
        min=-1.0, max=1.0, default=0.0, precision=2,
        update=_on_strength_update,
    )


# ── Lookup helpers ────────────────────────────────────────────────────────────

def _get_strength(
    scene: bpy.types.Scene,
    strip_name: str,
) -> Optional["NeuralStripStrengthEntry"]:
    """
    Read-only lookup of a strength entry by strip name.
    Returns None when no entry exists.
    Safe to call from draw() — never mutates the collection.
    """
    for item in scene.neural_nla_strengths:
        if item.strip_name == strip_name:
            return item
    return None


def _ensure_strength(
    scene: bpy.types.Scene,
    strip_name: str,
) -> "NeuralStripStrengthEntry":
    """
    Return the existing strength entry or create one with defaults.
    NEVER call this from a draw() callback — use _get_strength() there.
    """
    item = _get_strength(scene, strip_name)
    if item is None:
        item = scene.neural_nla_strengths.add()
        item.strip_name = strip_name
    return item


# ── Strength math ─────────────────────────────────────────────────────────────


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def evaluate_strength(
    entry: "NeuralStripStrengthEntry",
    frame: float,
    frame_start: float,
    frame_end: float,
) -> float:
    """
    Evaluate the strength envelope at *frame*.

    CONSTANT  →  entry.value  (flat)
    LINEAR    →  trapezoidal with linear ramps: floor at edges, peak in centre
    SMOOTH    →  trapezoidal: smoothstep ease-in, flat sustain, smoothstep ease-out
    """
    duration = max(frame_end - frame_start, 1.0)
    local_f  = max(0.0, min(duration, frame - frame_start))

    if entry.profile == "CONSTANT":
        return float(entry.value)

    # Both LINEAR and SMOOTH share peak / floor / fade_in / fade_out
    peak  = float(entry.peak_value)
    floor = float(entry.value_floor)
    fi_frac = float(entry.fade_in)
    fo_frac = float(entry.fade_out)
    total = fi_frac + fo_frac
    if total > 1.0:
        fi_frac /= total
        fo_frac /= total
    fi = fi_frac * duration
    fo = fo_frac * duration

    if entry.profile == "LINEAR":
        if fi > 0.0 and local_f < fi:
            return floor + (peak - floor) * (local_f / fi)
        if fo > 0.0 and local_f > (duration - fo):
            return floor + (peak - floor) * ((duration - local_f) / fo)
        return peak

    # SMOOTH — same regions but with smoothstep curves
    if fi > 0.0 and local_f < fi:
        return floor + (peak - floor) * _smoothstep(local_f / fi)

    if fo > 0.0 and local_f > (duration - fo):
        remaining = duration - local_f
        return floor + (peak - floor) * _smoothstep(remaining / fo)

    return peak


def sample_strength(
    strip: "bpy.types.NlaStrip",
    scene: bpy.types.Scene,
) -> list:
    """Return one strength value per frame, from frame_start to frame_end inclusive."""
    entry = _ensure_strength(scene, strip.name)
    return [
        round(evaluate_strength(entry, f, strip.frame_start, strip.frame_end), 5)
        for f in range(int(strip.frame_start), int(strip.frame_end) + 1)
    ]


def strength_to_metadata(strip: "bpy.types.NlaStrip", scene: bpy.types.Scene) -> dict:
    """Serialise the strip's strength profile for the /blend payload."""
    entry = _ensure_strength(scene, strip.name)
    blob: dict = {
        "profile":    entry.profile,
        "frame_start": strip.frame_start,
        "frame_end":   strip.frame_end,
        "samples":     sample_strength(strip, scene),  # one value per frame
    }
    if entry.profile == "CONSTANT":
        blob["value"] = entry.value
    elif entry.profile == "LINEAR":
        duration = strip.frame_end - strip.frame_start
        blob["peak_value"]      = entry.peak_value
        blob["value_floor"]     = entry.value_floor
        blob["fade_in_frames"]  = entry.fade_in  * duration
        blob["fade_out_frames"] = entry.fade_out * duration
    else:  # SMOOTH — convert fractions to frame counts for the server
        duration = strip.frame_end - strip.frame_start
        blob["peak_value"]      = entry.peak_value
        blob["fade_in_frames"]  = entry.fade_in  * duration
        blob["fade_out_frames"] = entry.fade_out * duration
    return blob


def _resulting_relative_strength(
    ref_strip: "bpy.types.NlaStrip",
    tgt_strip: "bpy.types.NlaStrip",
    scene: bpy.types.Scene,
    sample_count: int = 128,
) -> list:
    """Preview the server's target alpha over the strips' shared range."""
    overlap_start = max(ref_strip.frame_start, tgt_strip.frame_start)
    overlap_end = min(ref_strip.frame_end, tgt_strip.frame_end)
    if overlap_end <= overlap_start or sample_count < 2:
        return []

    ref_entry = _get_strength(scene, ref_strip.name)
    tgt_entry = _get_strength(scene, tgt_strip.name)
    if ref_entry is None or tgt_entry is None:
        return []

    values = []
    valid = []
    for index in range(sample_count):
        t = index / (sample_count - 1)
        frame = overlap_start + (overlap_end - overlap_start) * t
        ref_value = evaluate_strength(
            ref_entry, frame, ref_strip.frame_start, ref_strip.frame_end
        )
        tgt_value = evaluate_strength(
            tgt_entry, frame, tgt_strip.frame_start, tgt_strip.frame_end
        )
        total = ref_value + tgt_value
        value = tgt_value / total if total > 1e-6 else None
        values.append(value)
        if value is not None:
            valid.append(value)

    fallback = sum(valid) / len(valid) if valid else 0.5
    return [fallback if value is None else value for value in values]


def _strength_plot_image(values: list):
    """Create or update the compact relative-strength preview image."""
    global _strength_plot_signature
    # template_icon always reserves a square. Keep the source square too so
    # Blender does not letterbox the plot with large empty bands.
    width, height = 320, 320
    signature = tuple(round(float(value), 4) for value in values)
    image = bpy.data.images.get(_STRENGTH_PLOT_IMAGE)
    if image is not None and _strength_plot_signature != signature:
        # Blender caches an Image's icon thumbnail independently from its
        # pixels. Replacing the datablock guarantees a live UI preview.
        bpy.data.images.remove(image)
        image = None
    if image is None:
        image = bpy.data.images.new(
            _STRENGTH_PLOT_IMAGE,
            width=width,
            height=height,
            alpha=True,
        )
    elif _strength_plot_signature == signature:
        return image

    background = (0.055, 0.065, 0.08, 1.0)
    border = (0.24, 0.27, 0.31, 1.0)
    guide = (0.16, 0.18, 0.22, 1.0)
    curve = (0.20, 0.72, 1.0, 1.0)
    pixels = list(background) * (width * height)

    def set_pixel(x: int, y: int, color) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = 4 * (y * width + x)
            pixels[offset:offset + 4] = color

    left, right = 8, width - 9
    bottom, top = 8, height - 9
    for x in range(left, right + 1):
        set_pixel(x, bottom, border)
        set_pixel(x, top, border)
        set_pixel(x, (bottom + top) // 2, guide)
    for y in range(bottom, top + 1):
        set_pixel(left, y, border)
        set_pixel(right, y, border)

    points = []
    for index, value in enumerate(values):
        x = left + round(index * (right - left) / max(len(values) - 1, 1))
        y = bottom + round(max(0.0, min(1.0, value)) * (top - bottom))
        points.append((x, y))

    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for step in range(steps + 1):
            amount = step / steps
            x = round(x0 + (x1 - x0) * amount)
            y = round(y0 + (y1 - y0) * amount)
            for offset_y in (-1, 0, 1):
                set_pixel(x, y + offset_y, curve)

    image.pixels.foreach_set(pixels)
    image.update()
    _strength_plot_signature = signature
    return image


def find_strip_by_name(name: str) -> "Optional[bpy.types.NlaStrip]":
    """Return the first NlaStrip named *name* across all objects."""
    for obj in bpy.data.objects:
        if not obj.animation_data:
            continue
        for track in obj.animation_data.nla_tracks:
            for strip in track.strips:
                if strip.name == name:
                    return strip
    return None


# ─────────────────────────────────────────────────────────────────────────────
# NLA strip utilities
# ─────────────────────────────────────────────────────────────────────────────


def collect_selected_strips() -> Tuple[list, Optional["bpy.types.NlaStrip"]]:
    """
    Iterate all objects and return (selected_list, active_strip).

    *active_strip* is the strip whose ``strip.active`` flag is True — this is
    the flag Blender sets directly on the strip object when the user clicks it
    in the NLA editor, and is reliable regardless of which object is currently
    the scene's active object.  Falls back to ``None`` when no strip reports
    itself as active.
    """
    selected: list = []
    active: Optional[bpy.types.NlaStrip] = None

    for obj in bpy.data.objects:
        if not obj.animation_data:
            continue
        for track in obj.animation_data.nla_tracks:
            for strip in track.strips:
                if strip.select:
                    selected.append(strip)
                # strip.active is the ground-truth flag; getattr guards against
                # any future API change where the attribute might not exist.
                if getattr(strip, "active", False):
                    active = strip

    return selected, active


def get_reference_and_targets(
    context: bpy.types.Context,
) -> Tuple[Optional[bpy.types.NlaStrip], list]:
    """
    Return (reference_strip, [target_strips]).

    The reference strip is the active one (first match wins):
      1. ``strip.active`` flag — set by Blender on every LMB click.
      2. ``context.active_nla_strip`` — context-dependent fallback.
      3. First strip in scene-iteration order.

    All other selected strips become targets. Returns (None, []) when no
    strips are selected; a single selected strip is returned as the reference.
    """
    selected, active_by_flag = collect_selected_strips()
    if not selected:
        return None, []

    # ── Resolve the reference strip ───────────────────────────────────────────
    ref = active_by_flag  # priority 1

    if ref is None or ref not in selected:
        ref = getattr(context, "active_nla_strip", None)  # priority 2

    if ref is None or ref not in selected:
        ref = selected[0]  # priority 3

    targets = [s for s in selected if s is not ref]
    return ref, targets


def _active_selected_strip(context: bpy.types.Context):
    """Return the active selected NLA strip, falling back to the first selected."""
    selected, active = collect_selected_strips()
    if not selected:
        return None
    if active is None or active not in selected:
        active = getattr(context, "active_nla_strip", None)
    return active if active in selected else selected[0]


def find_strip_owner(target: bpy.types.NlaStrip) -> Optional[bpy.types.Object]:
    """Return the Object whose NLA tracks contain *target*."""
    for obj in bpy.data.objects:
        if not obj.animation_data:
            continue
        for track in obj.animation_data.nla_tracks:
            for strip in track.strips:
                if strip == target:
                    return obj
    return None


# ─────────────────────────────────────────────────────────────────────────────
# BVH export
#
# Blender exports its native Z-up basis and may discard names on End Site nodes.
# The server owns both repairs: it converts the world basis once, then reconciles
# names/order/topology against the selected cond.npy skeleton. Keeping hierarchy
# repair server-side avoids mixing offsets from two coordinate systems.
# ─────────────────────────────────────────────────────────────────────────────


def _write_source_bvh_clip(
    source_path: str,
    output_path: str,
    action_frame_start: float,
    action_frame_end: float,
    action_range_start: float,
) -> None:
    """Copy the selected action range from an original BVH without Blender loss."""
    with open(source_path, "r") as handle:
        lines = handle.readlines()

    motion_idx = next((i for i, line in enumerate(lines) if line.strip() == "MOTION"), None)
    if motion_idx is None:
        raise ValueError(f"BVH has no MOTION section: {source_path}")

    frames_idx = next(
        (i for i in range(motion_idx + 1, len(lines)) if lines[i].strip().startswith("Frames:")),
        None,
    )
    frame_time_idx = next(
        (i for i in range(motion_idx + 1, len(lines)) if lines[i].strip().startswith("Frame Time:")),
        None,
    )
    if frames_idx is None or frame_time_idx is None:
        raise ValueError(f"BVH has an incomplete MOTION header: {source_path}")

    frames = [line for line in lines[frame_time_idx + 1:] if line.strip()]
    first = max(0, int(round(action_frame_start - action_range_start)))
    last = min(len(frames), int(round(action_frame_end - action_range_start)) + 1)
    if last <= first:
        raise ValueError(
            f"Action range {action_frame_start:g}..{action_frame_end:g} "
            f"selects no frames from {source_path}"
        )

    selected = frames[first:last]
    output = lines[:frame_time_idx + 1]
    output[frames_idx] = f"Frames: {len(selected)}\n"
    output.extend(selected)
    with open(output_path, "w") as handle:
        handle.writelines(output)


def export_strip_as_bvh(obj: bpy.types.Object, strip: bpy.types.NlaStrip) -> str:
    """
    Build the BVH payload for *strip*.

    With a configured source BVH, copy the selected action-frame range directly
    so model-native joint names and local rotations survive. Without one, fall
    back to Blender's lossy BVH exporter.

    The export covers exactly the action clip region (action_frame_start →
    action_frame_end).  Repeat / scale / reverse / cyclic are passed as metadata.

    Returns the path to the temporary .bvh file (caller must clean it up).
    Raises ValueError if *obj* is not an ARMATURE.
    """
    if obj.type != "ARMATURE":
        raise ValueError(f"Object '{obj.name}' is not an ARMATURE — BVH export requires one.")
    if strip.action is None:
        raise ValueError(f"Strip '{strip.name}' has no linked Action.")

    tmp = tempfile.NamedTemporaryFile(suffix=".bvh", delete=False, prefix="nla_src_")
    tmp_path = tmp.name
    tmp.close()

    ref = _resolve_strip_source(
        bpy.context.scene,
        obj,
        strip,
        persist_auto=True,
    )
    if ref and os.path.exists(ref):
        _write_source_bvh_clip(
            ref,
            tmp_path,
            strip.action_frame_start,
            strip.action_frame_end,
            strip.action.frame_range[0],
        )
        return tmp_path

    anim_data = obj.animation_data
    orig_action = anim_data.action
    orig_use_nla = anim_data.use_nla
    orig_active = bpy.context.view_layer.objects.active
    orig_selected = {o: o.select_get() for o in bpy.context.scene.objects}

    try:
        for o in bpy.context.scene.objects:
            o.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        anim_data.use_nla = False
        anim_data.action = strip.action

        # Blender 4.x export_anim.bvh has no axis_forward/axis_up options.
        # It writes native Z-up; metadata tells the server to apply the fixed
        # Blender-to-model basis conversion before feature extraction.
        bpy.ops.export_anim.bvh(
            filepath=tmp_path,
            frame_start=int(strip.action_frame_start),
            frame_end=int(strip.action_frame_end),
            root_transform_only=True,
            rotate_mode="NATIVE",
        )

        print(
            f"[Neural NLA] No source BVH set for '{obj.name}'; using Blender's "
            "lossy BVH export fallback."
        )
    finally:
        anim_data.action = orig_action
        anim_data.use_nla = orig_use_nla
        for o, was_selected in orig_selected.items():
            try:
                o.select_set(was_selected)
            except ReferenceError:
                pass
        try:
            bpy.context.view_layer.objects.active = orig_active
        except ReferenceError:
            pass

    return tmp_path


def _generated_motion_dir() -> str:
    """Return the persistent per-user directory for generated BVH files."""
    override = os.environ.get("BLENDANYTHING_GENERATED_DIR", "").strip()
    if override:
        os.makedirs(override, exist_ok=True)
        return os.path.abspath(override)
    return bpy.utils.user_resource(
        "DATAFILES",
        path=os.path.join("blendanything", "generated"),
        create=True,
    )


def _persist_generated_bvh(bvh_path: str, result_name: str) -> str:
    """Copy a temporary result into the persistent generated-motion cache."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", result_name).strip("._")
    destination = os.path.join(_generated_motion_dir(), f"{safe_name}.bvh")
    shutil.copy2(bvh_path, destination)
    return destination


def import_bvh_as_result(bvh_path: str, profile_id: str = "") -> str:
    """
    Import *bvh_path*, rename the new object to Neural_Result_<timestamp>,
    move it into the output collection, and retain a durable source BVH.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_name = f"Neural_Result_{timestamp}"
    collection_name = "Neural Blending Outputs"
    persistent_path = _persist_generated_bvh(bvh_path, result_name)

    # Ensure output collection exists and is linked to the scene
    if collection_name not in bpy.data.collections:
        col = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(col)
    output_col = bpy.data.collections[collection_name]

    # Snapshot selection so we can detect the newly created object
    before_import = set(bpy.context.scene.objects)

    # The server writes the result in the model's native space (Y-up, Z-forward).
    # Blender's BVH importer maps Y-up → Blender's native Z-up by default
    # (axis_up="Y", axis_forward="-Z"); we set them explicitly so the result
    # lands consistently with the +Z-up authoring contract regardless of any
    # change to Blender's import defaults.
    bpy.ops.import_anim.bvh(
        filepath=persistent_path,
        axis_forward="-Z",
        axis_up="Y",
    )

    # The newly added object(s) are those not present before the import
    new_objects = set(bpy.context.scene.objects) - before_import
    if not new_objects:
        try:
            os.unlink(persistent_path)
        except OSError:
            pass
        raise RuntimeError("BVH import produced no new objects.")

    for new_obj in new_objects:
        new_obj.name = result_name
        new_obj["blendanything_generated_bvh"] = persistent_path
        if new_obj.animation_data and new_obj.animation_data.action:
            action = new_obj.animation_data.action
            _set_action_source(
                bpy.context.scene,
                action,
                persistent_path,
                profile_id=profile_id,
            )
            _set_armature_reference(
                bpy.context.scene,
                new_obj.name,
                persistent_path,
            )
            track = new_obj.animation_data.nla_tracks.new()
            track.name = "BlendAnything"
            strip = track.strips.new(
                action.name,
                int(round(action.frame_range[0])),
                action,
            )
            strip.select = True
            new_obj.animation_data.action = None
            _ensure_strip_skeleton(bpy.context.scene, strip)
            strength = _ensure_strength(bpy.context.scene, strip.name)
            _apply_profile_to_strip(strip, strength)
        # Re-link into the output collection only
        for col in list(new_obj.users_collection):
            col.objects.unlink(new_obj)
        output_col.objects.link(new_obj)

    print(
        f"[Neural NLA] Imported '{result_name}' → collection '{collection_name}' "
        f"| source: {persistent_path}"
    )

    try:
        os.unlink(bvh_path)
    except OSError:
        pass
    return persistent_path


# ─────────────────────────────────────────────────────────────────────────────
# bpy.app.timers callback — polls thread result at 0.1 s intervals
# ─────────────────────────────────────────────────────────────────────────────


def _timer_poll_result() -> Optional[float]:
    """
    Registered with bpy.app.timers.  Called every 0.1 s on the main thread.
    Returns 0.1 to re-schedule, or None to unregister.
    """
    st = _read_state()

    if st["status"] == "running":
        _redraw_nla_editors()
        return 0.1  # still waiting — keep polling

    if st["status"] == "done":
        result_path = st["result_path"]
        result_profile_id = st.get("result_profile_id", "")
        _set_state(phase="Importing result")
        _redraw_nla_editors()
        try:
            persistent_path = import_bvh_as_result(result_path, result_profile_id)
            _set_state(
                status="idle",
                result_path=None,
                result_profile_id="",
                generated_path=persistent_path,
                error_msg="",
                phase="",
                started_at=0.0,
                progress=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            _set_state(status="error", error_msg=str(exc), phase="", progress=0.0)
        _redraw_nla_editors()
        return None  # unregister timer

    if st["status"] == "error":
        # Error message is already stored; panel will display it.
        # Do NOT reset to idle here — let the user see the message.
        _redraw_nla_editors()
        return None  # unregister timer

    return None  # unexpected state — stop timer


# ─────────────────────────────────────────────────────────────────────────────
# Server configuration
# ─────────────────────────────────────────────────────────────────────────────


def _server_url(context: bpy.types.Context) -> str:
    return context.scene.neural_nla_server_url.rstrip("/")


def _server_model_items(self, context):
    return _model_enum_items()


def _sync_model_selection(scene: bpy.types.Scene, payload: dict) -> None:
    models = payload.get("models", [])
    active = payload.get("active_model") or ""
    selection = active if active in models else (models[0] if models else "NONE")
    try:
        scene.neural_nla_server_model = selection
    except TypeError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Operators
# ─────────────────────────────────────────────────────────────────────────────


class NEURAL_NLA_OT_TestConnection(bpy.types.Operator):
    """Ping the /health endpoint and report the result."""

    bl_idname = "neural_nla.test_connection"
    bl_label = "Test Connection"
    bl_description = "Ping the neural blending server /health endpoint"
    bl_options = {"REGISTER"}

    def execute(self, context: bpy.types.Context):
        if not HAS_REQUESTS:
            self.report(
                {"ERROR"},
                "The 'requests' library is missing. "
                "Install it: pip install requests  (into Blender's Python).",
            )
            return {"CANCELLED"}

        import requests

        url = _server_url(context) + "/health"
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            payload = _fetch_server_models(_server_url(context))
            _sync_model_selection(context.scene, payload)
            active = payload.get("active_model") or "none"
            self.report({"INFO"}, f"Server online; active model: {active}")
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Connection failed: {exc}")
        return {"FINISHED"}


class NEURAL_NLA_OT_RefreshModels(bpy.types.Operator):
    """Refresh loadable model folders from the configured server."""

    bl_idname = "neural_nla.refresh_models"
    bl_label = "Refresh Models"
    bl_options = {"REGISTER"}

    def execute(self, context: bpy.types.Context):
        if not HAS_REQUESTS:
            self.report({"ERROR"}, "The 'requests' library is missing.")
            return {"CANCELLED"}
        try:
            payload = _fetch_server_models(_server_url(context))
            _sync_model_selection(context.scene, payload)
            self.report({"INFO"}, f"Found {len(payload.get('models', []))} model(s)")
            return {"FINISHED"}
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Could not refresh models: {exc}")
            return {"CANCELLED"}


class NEURAL_NLA_OT_LoadModel(bpy.types.Operator):
    """Load the selected server model checkpoint."""

    bl_idname = "neural_nla.load_model"
    bl_label = "Load Selected Model"
    bl_options = {"REGISTER"}

    def execute(self, context: bpy.types.Context):
        model_name = context.scene.neural_nla_server_model
        if model_name == "NONE":
            self.report({"ERROR"}, "Refresh models and select a valid model.")
            return {"CANCELLED"}
        try:
            payload = _load_server_model(_server_url(context), model_name)
            self.report({"INFO"}, f"Loaded model: {payload['active_model']}")
            return {"FINISHED"}
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Could not load model: {exc}")
            return {"CANCELLED"}


class NEURAL_NLA_OT_RunBlend(bpy.types.Operator):
    """Export the two selected NLA strips as BVH and send them to the server."""

    bl_idname = "neural_nla.run_blend"
    bl_label = "Run Neural Blend"
    bl_description = (
        "Export the selected NLA strips as BVH, post them to the "
        "neural blending server, and import the result"
    )
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        selected, _ = collect_selected_strips()
        st = _read_state()
        target_only = (
            context.scene.neural_nla_blend_mode == "RETARGET"
            and context.scene.neural_nla_target_only
        )
        required = 1 if target_only else 2
        return len(selected) >= required and st["status"] not in ("running",)

    def execute(self, context: bpy.types.Context):
        if not HAS_REQUESTS:
            self.report(
                {"ERROR"},
                "The 'requests' library is missing. "
                "Install it into Blender's Python: "
                "<blender_python> -m pip install requests",
            )
            return {"CANCELLED"}

        # ── Resolve strips ────────────────────────────────────────────────────
        ref_strip, target_strips = get_reference_and_targets(context)
        blend_mode = context.scene.neural_nla_blend_mode
        single_retarget = (
            blend_mode == "RETARGET" and context.scene.neural_nla_target_only
        )
        if single_retarget:
            ref_strip = _active_selected_strip(context)
            target_strips = []
        if ref_strip is None or (not single_retarget and not target_strips):
            requirement = (
                "Select a target-motion NLA strip."
                if single_retarget
                else "Select at least 2 NLA strips (1 reference + 1 or more targets)."
            )
            self.report({"ERROR"}, requirement)
            return {"CANCELLED"}
        destination_profile = (
            context.scene.neural_nla_destination_profile if single_retarget else ""
        )
        destination_profile = _qualify_profile_id(destination_profile)
        if single_retarget and destination_profile not in _load_skeleton_profiles():
            self.report({"ERROR"}, "Choose a reference skeleton for Target Only.")
            return {"CANCELLED"}

        ref_obj = find_strip_owner(ref_strip)
        if ref_obj is None:
            self.report({"ERROR"}, "Could not locate armature owner for the reference strip.")
            return {"CANCELLED"}

        profiles = _load_skeleton_profiles()
        for strip in [ref_strip] + list(target_strips):
            if strip.action is None:
                self.report({"ERROR"}, f"Strip '{strip.name}' has no action.")
                return {"CANCELLED"}
            owner = find_strip_owner(strip)
            _resolve_strip_source(
                context.scene, owner, strip, persist_auto=True
            ) if owner else ""
            skeleton = _ensure_strip_skeleton(context.scene, strip)
            profile_id = _qualify_profile_id(skeleton.profile_id)
            if profile_id == "CUSTOM":
                if not skeleton.custom_name.strip():
                    skeleton.custom_name = (
                        os.path.splitext(os.path.basename(
                            _resolve_strip_source(
                                context.scene, owner, strip, persist_auto=True
                            )
                        ))[0]
                        if owner else strip.name
                    )
                if not skeleton.custom_name.strip():
                    self.report(
                        {"ERROR"},
                        f"Set a custom skeleton name for strip '{strip.name}'.",
                    )
                    return {"CANCELLED"}
            elif profile_id not in profiles:
                self.report(
                    {"ERROR"},
                    f"Choose a skeleton for strip '{strip.name}'.",
                )
                return {"CANCELLED"}
            face = [
                skeleton.face_right, skeleton.face_left,
                skeleton.face_upper_right, skeleton.face_upper_left,
            ]
            if any(not name for name in face):
                self.report(
                    {"ERROR"},
                    f"Set all four face joints for strip '{strip.name}'.",
                )
                return {"CANCELLED"}
            known_joints = set(profiles.get(profile_id, {}).get("joints_names", []))
            unknown_face = [name for name in face if known_joints and name not in known_joints]
            if unknown_face:
                self.report(
                    {"ERROR"},
                    f"Unknown face joints for '{profile_id}' on strip "
                    f"'{strip.name}': {', '.join(unknown_face)}",
                )
                return {"CANCELLED"}

        # ── Export BVH ────────────────────────────────────────────────────────
        try:
            ref_bvh = export_strip_as_bvh(ref_obj, ref_strip)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Reference BVH export failed: {exc}")
            return {"CANCELLED"}

        target_bvhs = []
        for i, strip in enumerate(target_strips):
            t_obj = find_strip_owner(strip)
            if t_obj is None:
                for p in [ref_bvh] + target_bvhs:
                    os.unlink(p)
                self.report({"ERROR"}, f"Could not locate armature owner for target strip '{strip.name}'.")
                return {"CANCELLED"}
            try:
                target_bvhs.append(export_strip_as_bvh(t_obj, strip))
            except Exception as exc:  # noqa: BLE001
                for p in [ref_bvh] + target_bvhs:
                    os.unlink(p)
                self.report({"ERROR"}, f"Target BVH export failed for '{strip.name}': {exc}")
                return {"CANCELLED"}

        # ── Build metadata ────────────────────────────────────────────────────
        def strip_meta(s: bpy.types.NlaStrip) -> dict:
            owner = find_strip_owner(s)
            source_path = (
                _resolve_strip_source(context.scene, owner, s, persist_auto=True)
                if owner else ""
            )
            source_exists = bool(source_path) and os.path.exists(source_path)
            source_entry = _ensure_action_source_entry(context.scene, s.action) if s.action else None
            skeleton = _ensure_strip_skeleton(context.scene, s)
            profile_id = (
                _qualify_profile_id(skeleton.profile_id)
                if skeleton and skeleton.profile_id != "NONE" else ""
            )
            if profile_id == "CUSTOM":
                skeleton_dataset = "custom"
                object_type = f"user::{skeleton.custom_name.strip()}"
            else:
                skeleton_dataset, object_type = (
                    _profile_parts(profile_id) if profile_id else ("", "")
                )
            input_mode = source_entry.input_mode if source_entry else "BLENDER_EXPORT"
            face_joints = (
                [
                    skeleton.face_right,
                    skeleton.face_left,
                    skeleton.face_upper_right,
                    skeleton.face_upper_left,
                ]
                if skeleton else []
            )
            return {
                # Strip identity
                "name":               s.name,
                "action":             s.action.name if s.action else None,
                # Scene-time placement of the strip
                "frame_start":        s.frame_start,
                "frame_end":          s.frame_end,
                # Action clip region — matches the BVH frame range
                "action_frame_start": s.action_frame_start,
                "action_frame_end":   s.action_frame_end,
                # Playback modifiers — server applies these to the raw BVH
                "repeat":             s.repeat,
                "scale":              s.scale,
                "use_reverse":        s.use_reverse,
                "extrapolation":      s.extrapolation,  # 'NOTHING' | 'HOLD' | 'HOLD_FORWARD'
                "coordinate_space": (
                    "MODEL_Y_UP" if source_exists and input_mode == "MODEL_PROCESSED"
                    else "RAW_TPOSE_RELATIVE" if source_exists else "BLENDER_Z_UP"
                ),
                "object_type":        object_type,
                "skeleton_dataset":   skeleton_dataset,
                "skeleton_profile":   profile_id,
                "custom_skeleton":    profile_id == "CUSTOM",
                "input_mode":         input_mode,
                "face_joints":        [name for name in face_joints if name],
                # Influence / blend weight envelope
                "strength":           strength_to_metadata(s, context.scene),
            }

        reference_meta = strip_meta(ref_strip)
        targets_meta = [strip_meta(s) for s in target_strips]
        metadata = {
            "reference":    reference_meta,
            "targets":      targets_meta,
            "blend_mode":   "SINGLE_RETARGET" if single_retarget else blend_mode,
            "control_mode": "tgt" if blend_mode == "RETARGET" else "both",
            "destination_object_type": (
                _profile_parts(destination_profile)[1]
                if destination_profile else ""
            ),
            "destination_skeleton_profile": destination_profile,
            "destination_dataset": (
                _profile_parts(destination_profile)[0]
                if destination_profile else ""
            ),
            "output_mode":  context.scene.neural_nla_output_mode,
            "ik_iterations": context.scene.neural_nla_ik_iterations,
            "ddim_inversion_policy": context.scene.neural_nla_ddim_inversion_policy,
        }

        # ── Fire background thread ────────────────────────────────────────────
        _set_state(
            status="running",
            result_path=None,
            result_profile_id=(
                destination_profile
                if single_retarget
                else reference_meta.get("skeleton_profile", "")
            ),
            generated_path="",
            error_msg="",
            warnings=[],
            phase="Preparing request",
            started_at=time.monotonic(),
            progress=0.01,
        )

        thread = threading.Thread(
            target=_thread_send_request,
            args=(_server_url(context), ref_bvh, target_bvhs, metadata),
            daemon=True,
            name="NeuralNLABlendThread",
        )
        thread.start()

        # Register polling timer (idempotent guard)
        if not bpy.app.timers.is_registered(_timer_poll_result):
            bpy.app.timers.register(_timer_poll_result, first_interval=0.1)

        self.report({"INFO"}, "Neural blend started — processing in background…")
        return {"FINISHED"}


class NEURAL_NLA_OT_ClearStatus(bpy.types.Operator):
    """Dismiss displayed status messages and reset the pipeline state."""

    bl_idname = "neural_nla.clear_status"
    bl_label = "Clear Status"
    bl_description = "Dismiss errors and model-distribution notes"
    bl_options = {"REGISTER"}

    def execute(self, context: bpy.types.Context):
        _set_state(
            status="idle",
            result_path=None,
            result_profile_id="",
            generated_path="",
            error_msg="",
            warnings=[],
            phase="",
            started_at=0.0,
            progress=0.0,
        )
        return {"FINISHED"}


class NEURAL_NLA_OT_DismissWarnings(bpy.types.Operator):
    """Dismiss model-compatibility warnings without resetting the active job."""

    bl_idname = "neural_nla.dismiss_warnings"
    bl_label = "Dismiss Warnings"
    bl_description = "Dismiss model compatibility warnings"
    bl_options = {"INTERNAL"}

    def execute(self, context: bpy.types.Context):
        _set_state(warnings=[])
        return {"FINISHED"}


class NEURAL_NLA_OT_SetReferenceBVH(bpy.types.Operator):
    """Pick the source BVH backing an NLA action."""

    bl_idname = "neural_nla.set_reference_bvh"
    bl_label = "Set Reference BVH"
    bl_description = (
        "Select the original .bvh for this action. The plugin sends its "
        "model-native channels instead of Blender's lossy BVH re-export"
    )
    bl_options = {"REGISTER"}

    action_name: bpy.props.StringProperty()  # type: ignore[assignment]
    object_name: bpy.props.StringProperty()  # type: ignore[assignment]
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")  # type: ignore[assignment]
    filter_glob: bpy.props.StringProperty(default="*.bvh", options={"HIDDEN"})  # type: ignore[assignment]

    def execute(self, context: bpy.types.Context):
        action = bpy.data.actions.get(self.action_name) if self.action_name else None
        if action is None:
            self.report({"WARNING"}, "The action for this source BVH no longer exists.")
            return {"CANCELLED"}

        owner = bpy.data.objects.get(self.object_name) if self.object_name else None
        valid, reason = _validate_source_bvh(self.filepath, action, owner)
        if not valid:
            self.report({"ERROR"}, f"Source BVH does not match '{action.name}': {reason}")
            return {"CANCELLED"}

        _set_action_source(context.scene, action, self.filepath)
        self.report({"INFO"},
                    f"Source for '{action.name}': {os.path.basename(self.filepath)}")
        return {"FINISHED"}

    def invoke(self, context: bpy.types.Context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class NEURAL_NLA_OT_InitStripSkeleton(bpy.types.Operator):
    """Create skeleton settings for the active NLA strip."""

    bl_idname = "neural_nla.init_strip_skeleton"
    bl_label = "Initialize Skeleton"
    bl_description = "Create per-strip skeleton settings and auto-fill them when possible"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        selected, active = collect_selected_strips()
        if not selected:
            self.report({"WARNING"}, "Select an NLA strip first.")
            return {"CANCELLED"}
        strip = active if (active is not None and active in selected) else selected[0]
        if strip.action is None:
            self.report({"ERROR"}, f"Strip '{strip.name}' has no action.")
            return {"CANCELLED"}

        owner = find_strip_owner(strip)
        if owner is not None:
            _resolve_strip_source(context.scene, owner, strip, persist_auto=True)
        skeleton = _ensure_strip_skeleton(context.scene, strip)
        if skeleton.profile_id == "NONE":
            self.report({"INFO"}, "Skeleton settings created; choose a dataset skeleton.")
        else:
            self.report({"INFO"}, f"Detected skeleton: {skeleton.profile_id}")
        return {"FINISHED"}


class NEURAL_NLA_OT_ImportTrackedBVH(bpy.types.Operator):
    """Import a BVH with tracked source, skeleton, and strength defaults."""

    bl_idname = "neural_nla.import_tracked_bvh"
    bl_label = "Import BVH (Tracked)"
    bl_description = (
        "Import a BVH into an NLA strip, retain its source, detect its skeleton, "
        "and apply the default strength profile"
    )
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")  # type: ignore[assignment]
    filter_glob: bpy.props.StringProperty(default="*.bvh", options={"HIDDEN"})  # type: ignore[assignment]

    def execute(self, context: bpy.types.Context):
        before = set(bpy.data.objects)
        bpy.ops.import_anim.bvh(
            filepath=self.filepath,
            axis_forward="-Z",
            axis_up="Y",
        )
        imported = [
            obj for obj in set(bpy.data.objects) - before
            if obj.type == "ARMATURE" and obj.animation_data
        ]
        if not imported:
            self.report({"ERROR"}, "BVH import produced no animated armature.")
            return {"CANCELLED"}

        imported_strips = []
        for obj in imported:
            action = obj.animation_data.action
            if action is not None:
                _set_action_source(context.scene, action, self.filepath)
                _set_armature_reference(context.scene, obj.name, self.filepath)

                track = obj.animation_data.nla_tracks.new()
                track.name = "BlendAnything"
                strip = track.strips.new(
                    action.name,
                    int(round(action.frame_range[0])),
                    action,
                )
                strip.select = True
                obj.animation_data.action = None
                _ensure_strip_skeleton(context.scene, strip)
                strength = _ensure_strength(context.scene, strip.name)
                _apply_profile_to_strip(strip, strength)
                imported_strips.append(strip)

        if not imported_strips:
            self.report({"ERROR"}, "BVH import produced no action to place in the NLA editor.")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Imported {os.path.basename(self.filepath)} with skeleton and strength defaults",
        )
        return {"FINISHED"}

    def invoke(self, context: bpy.types.Context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


# ─────────────────────────────────────────────────────────────────────────────
# Strength preset operator
# ─────────────────────────────────────────────────────────────────────────────


class NEURAL_NLA_OT_StrengthPreset(bpy.types.Operator):
    """Apply a quick strength preset to the named strip."""

    bl_idname  = "neural_nla.strength_preset"
    bl_label   = "Strength Preset"
    bl_options = {"REGISTER", "UNDO"}

    strip_name: bpy.props.StringProperty()  # type: ignore[assignment]

    preset: bpy.props.EnumProperty(         # type: ignore[assignment]
        name="Preset",
        items=[
            ("FULL",       "Full",        "Constant 1.0 — full influence"),
            ("HALF",       "Half",        "Constant 0.5"),
            ("SILENT",     "Silent",      "Constant 0.0 — strip present but inert"),
            ("FADE_IN",    "Fade In",     "Smooth ramp from 0 to peak over the whole strip"),
            ("FADE_OUT",   "Fade Out",    "Smooth ramp from peak to 0 over the whole strip"),
            ("EASE_IN_OUT","Ease In/Out", "Ease in first quarter, ease out last quarter"),
        ],
    )

    def execute(self, context: bpy.types.Context):
        strip = find_strip_by_name(self.strip_name)
        if strip is None:
            self.report({"WARNING"}, f"Strip '{self.strip_name}' not found.")
            return {"CANCELLED"}

        entry   = _ensure_strength(context.scene, self.strip_name)
        profile = entry.profile

        if self.preset in ("FADE_IN", "FADE_OUT", "EASE_IN_OUT"):
            # Directional presets need at least LINEAR; upgrade CONSTANT only.
            if profile == "CONSTANT":
                profile = entry.profile = "LINEAR"

        if self.preset == "FADE_IN":
            entry.peak_value = 1.0;  entry.value_floor = 0.0
            entry.fade_in = 1.0;  entry.fade_out = 0.0

        elif self.preset == "FADE_OUT":
            entry.peak_value = 1.0;  entry.value_floor = 0.0
            entry.fade_in = 0.0;  entry.fade_out = 1.0

        elif self.preset == "EASE_IN_OUT":
            entry.peak_value = 1.0;  entry.value_floor = 0.0
            entry.fade_in = 0.25;  entry.fade_out = 0.25

        elif self.preset == "FULL":
            if profile == "CONSTANT":    entry.value = 1.0
            else:                        entry.peak_value = 1.0

        elif self.preset == "HALF":
            if profile == "CONSTANT":    entry.value = 0.5
            else:                        entry.peak_value = 0.5

        elif self.preset == "SILENT":
            if profile == "CONSTANT":    entry.value = 0.0
            else:                        entry.peak_value = 0.0;  entry.value_floor = 0.0

        _apply_profile_to_strip(strip, entry)
        return {"FINISHED"}


class NEURAL_NLA_OT_ApplyInfluence(bpy.types.Operator):
    """Write the current strength preset to the strip's built-in influence curve."""

    bl_idname  = "neural_nla.apply_influence"
    bl_label   = "Apply to Strip Influence"
    bl_options = {"REGISTER", "UNDO"}

    strip_name: bpy.props.StringProperty()  # type: ignore[assignment]

    def execute(self, context: bpy.types.Context):
        strip = find_strip_by_name(self.strip_name)
        if strip is None:
            self.report({"WARNING"}, f"Strip '{self.strip_name}' not found.")
            return {"CANCELLED"}

        entry = _ensure_strength(context.scene, self.strip_name)
        _apply_profile_to_strip(strip, entry)

        # Enable "Show Strip Curves" overlay so the user can see the result
        if hasattr(context.space_data, "show_strip_curves"):
            context.space_data.show_strip_curves = True

        return {"FINISHED"}


class NEURAL_NLA_OT_ApplyLinkedCrossfade(bpy.types.Operator):
    """Create complementary relative-strength curves for two overlapping strips."""

    bl_idname = "neural_nla.apply_linked_crossfade"
    bl_label = "Apply Linked Crossfade"
    bl_description = (
        "Fade the earlier strip out while fading the later strip in across "
        "their shared timeline overlap"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        selected, _ = collect_selected_strips()
        if len(selected) != 2:
            self.report({"ERROR"}, "Select exactly two NLA strips.")
            return {"CANCELLED"}

        earlier, later = sorted(selected, key=lambda strip: strip.frame_start)
        overlap_start = max(earlier.frame_start, later.frame_start)
        overlap_end = min(earlier.frame_end, later.frame_end)
        overlap = overlap_end - overlap_start
        if overlap <= 0.0:
            self.report({"ERROR"}, "The selected strips do not overlap.")
            return {"CANCELLED"}
        if earlier.frame_end > later.frame_end:
            self.report(
                {"ERROR"},
                "Linked crossfade requires an edge overlap: the earlier strip "
                "must also finish first.",
            )
            return {"CANCELLED"}

        profile = context.scene.neural_nla_crossfade_shape
        earlier_strength = _ensure_strength(context.scene, earlier.name)
        later_strength = _ensure_strength(context.scene, later.name)

        earlier_strength.profile = profile
        earlier_strength.peak_value = 1.0
        earlier_strength.value_floor = 0.0
        earlier_strength.fade_in = 0.0
        earlier_strength.fade_out = min(
            1.0, overlap / max(earlier.frame_end - earlier.frame_start, 1.0)
        )

        later_strength.profile = profile
        later_strength.peak_value = 1.0
        later_strength.value_floor = 0.0
        later_strength.fade_in = min(
            1.0, overlap / max(later.frame_end - later.frame_start, 1.0)
        )
        later_strength.fade_out = 0.0

        _apply_profile_to_strip(earlier, earlier_strength)
        _apply_profile_to_strip(later, later_strength)
        self.report(
            {"INFO"},
            f"Linked {earlier.name} → {later.name} across {int(round(overlap))} frames",
        )
        return {"FINISHED"}


# ─────────────────────────────────────────────────────────────────────────────
# Apply strength profile to strip.influence
#
# Writes keyframes directly onto NlaStrip.fcurves["influence"] so Blender's
# built-in NLA strip curve visualisation (Overlays → Show Strip Curves) and
# its NLA evaluation engine both pick up the envelope natively.
# ─────────────────────────────────────────────────────────────────────────────


def _apply_profile_to_strip(
    strip: "bpy.types.NlaStrip",
    entry: "NeuralStripStrengthEntry",
) -> None:
    """
    Write the strength profile from *entry* as keyframes on strip.influence.

    Any existing influence F-curve on the strip is replaced.  Blender's NLA
    evaluation engine and the built-in "Show Strip Curves" overlay both pick
    up the result automatically — no custom GPU drawing needed.

    Interpolation per profile:
      CONSTANT  →  2 keyframes, CONSTANT interpolation
      LINEAR    →  2–4 keyframes (floor … peak … floor) driven by fade_in /
                   fade_out fractions; all keyframes use LINEAR interpolation.
      SMOOTH    →  2–4 keyframes (floor … peak … floor) driven by fade_in /
                   fade_out fractions.  Outer keyframes use the ease_shape
                   interpolation type; with BEZIER shape the overshoot property
                   offsets boundary handle tangents for bounce/anticipation.
    """
    # "Animated Influence" must be enabled on the strip before Blender allows
    # creating or editing the influence F-curve.
    if not strip.use_animated_influence:
        strip.use_animated_influence = True

    # Reuse an existing influence F-curve (NlaStripFCurves has no remove()),
    # or create one if none exists yet.
    x0 = float(strip.frame_start)
    x1 = float(strip.frame_end)
    fc  = strip.fcurves.find("influence")
    if fc is not None:
        fc.keyframe_points.clear()
    else:
        fc = strip.fcurves.new("influence")
    kps = fc.keyframe_points

    if entry.profile == "CONSTANT":
        kps.add(2)
        kps[0].co            = (x0, entry.value)
        kps[0].interpolation = "CONSTANT"
        kps[1].co            = (x1, entry.value)
        kps[1].interpolation = "CONSTANT"

    elif entry.profile == "LINEAR":
        # Same trapezoidal structure as SMOOTH; all keyframes use LINEAR interpolation.
        duration = max(x1 - x0, 1.0)
        fi_frac  = float(entry.fade_in)
        fo_frac  = float(entry.fade_out)
        total = fi_frac + fo_frac
        if total > 1.0:
            fi_frac /= total
            fo_frac /= total
        fi    = fi_frac * duration
        fo    = fo_frac * duration
        peak  = float(entry.peak_value)
        floor = float(entry.value_floor)

        pts: list = []
        if fi > 0:
            pts.append((x0, floor))
        plateau_start = x0 + fi
        plateau_end   = x1 - fo
        if plateau_end - plateau_start > 0.5:
            pts.append((plateau_start, peak))
            pts.append((plateau_end,   peak))
        else:
            pts.append(((plateau_start + plateau_end) / 2.0, peak))
        if fo > 0:
            pts.append((x1, floor))

        n = len(pts)
        kps.add(n)
        for i, (frame, val) in enumerate(pts):
            kps[i].co            = (frame, val)
            kps[i].interpolation = "LINEAR"

    else:  # SMOOTH — ease-in, flat sustain, ease-out
        duration = max(x1 - x0, 1.0)
        fi_frac  = float(entry.fade_in)
        fo_frac  = float(entry.fade_out)
        # Proportionally scale down if fade regions would overlap
        total = fi_frac + fo_frac
        if total > 1.0:
            fi_frac /= total
            fo_frac /= total
        fi    = fi_frac * duration
        fo    = fo_frac * duration
        peak  = float(entry.peak_value)
        floor = float(entry.value_floor)
        shape = entry.ease_shape   # "BEZIER" | "EASE_IN" | "EASE_OUT" | "EASE_IN_OUT"
        over  = float(entry.overshoot)

        # Build keyframe list dynamically so edge cases (fi=0, fo=0,
        # fi+fo≥duration) all produce clean, non-duplicate points.
        # Each entry: (frame, value, is_boundary)
        pts: list = []
        if fi > 0:
            pts.append((x0, floor, True))
        plateau_start = x0 + fi
        plateau_end   = x1 - fo
        if plateau_end - plateau_start > 0.5:
            pts.append((plateau_start, peak, False))   # sustain left edge
            pts.append((plateau_end,   peak, False))   # sustain right edge
        else:
            pts.append(((plateau_start + plateau_end) / 2.0, peak, False))  # merged peak
        if fo > 0:
            pts.append((x1, floor, True))

        n = len(pts)
        kps.add(n)
        for i, (frame, val, is_boundary) in enumerate(pts):
            kps[i].co                = (frame, val)
            kps[i].handle_left_type  = "AUTO"
            kps[i].handle_right_type = "AUTO"
            if is_boundary and shape != "BEZIER":
                # EASE_IN / EASE_OUT / EASE_IN_OUT are Blender easing modes,
                # not interpolation types.  Use SINE as the base curve and set
                # the easing direction separately.
                kps[i].interpolation = "SINE"
                kps[i].easing        = shape   # "EASE_IN" | "EASE_OUT" | "EASE_IN_OUT"
            else:
                kps[i].interpolation = "BEZIER"

        # Flatten handles at the sustain plateau edges
        if n == 4:
            kps[1].handle_right_type = "VECTOR"  # into flat sustain
            kps[2].handle_left_type  = "VECTOR"  # out of flat sustain

        # Overshoot — nudge boundary handles to create bounce/anticipation.
        # Only meaningful when using BEZIER interpolation; other shapes are
        # mathematical functions that don't expose handle control.
        if shape == "BEZIER" and abs(over) > 0.001:
            height = peak - floor
            h_off  = over * height            # vertical handle displacement
            if fi > 0:
                span = fi / 3.0
                kp   = kps[0]
                kp.handle_right_type = "FREE"
                kp.handle_left_type  = "FREE"
                kp.handle_right      = (x0 + span, floor + h_off)
                kp.handle_left       = (x0 - span, floor)
            if fo > 0:
                span = fo / 3.0
                kp   = kps[n - 1]
                kp.handle_left_type  = "FREE"
                kp.handle_right_type = "FREE"
                kp.handle_left       = (x1 - span, floor + h_off)
                kp.handle_right      = (x1 + span, floor)

    fc.update()

    # Record where we placed keyframes so the sync handler can detect movement
    # by comparing strip.frame_start/end against these stored values, rather
    # than reading back the (potentially stale) keyframe positions.
    entry.last_frame_start = strip.frame_start
    entry.last_frame_end   = strip.frame_end


# ─────────────────────────────────────────────────────────────────────────────
# Strength sub-panel helper + child panels
# ─────────────────────────────────────────────────────────────────────────────


def _draw_strength_ui(layout: bpy.types.UILayout, strip: bpy.types.NlaStrip) -> None:
    """Draw strength preset controls for *strip*."""
    scene = bpy.context.scene
    entry = _get_strength(scene, strip.name)

    if entry is None:
        op = layout.operator("neural_nla.apply_influence", text="Apply Default Profile",
                             icon="FCURVE")
        op.strip_name = strip.name
        return

    # Profile selector
    row = layout.row(align=True)
    row.label(text="Profile:")
    row.prop(entry, "profile", text="")

    # Profile-specific value controls
    if entry.profile == "CONSTANT":
        layout.prop(entry, "value", text="Value", slider=True)

    elif entry.profile == "LINEAR":
        col = layout.column(align=True)
        col.prop(entry, "peak_value",  text="Peak",  slider=True)
        col.prop(entry, "value_floor", text="Floor", slider=True)
        col.separator(factor=0.4)
        col.prop(entry, "fade_in",  text="Start", slider=True)
        col.prop(entry, "fade_out", text="End",   slider=True)

    else:  # SMOOTH
        col = layout.column(align=True)
        col.prop(entry, "peak_value",  text="Peak",     slider=True)
        col.prop(entry, "value_floor", text="Floor",    slider=True)
        col.separator(factor=0.4)
        col.prop(entry, "fade_in",     text="Fade In",  slider=True)
        col.prop(entry, "fade_out",    text="Fade Out", slider=True)
        col.separator(factor=0.4)
        row = col.row(align=True)
        row.label(text="Shape:")
        row.prop(entry, "ease_shape", text="")
        if entry.ease_shape == "BEZIER":
            col.prop(entry, "overshoot", text="Overshoot", slider=True)

    # Quick-preset row
    layout.separator(factor=0.5)
    row = layout.row(align=True)
    for preset, label in (
        ("FADE_IN",     "\u2197 In"),
        ("FADE_OUT",    "\u2198 Out"),
        ("EASE_IN_OUT", "\u223c Ease"),
        ("FULL",        "1.0"),
        ("HALF",        "0.5"),
        ("SILENT",      "0.0"),
    ):
        op            = row.operator("neural_nla.strength_preset", text=label)
        op.preset     = preset
        op.strip_name = strip.name



# ─────────────────────────────────────────────────────────────────────────────
# NLA Editor Sidebar Panels
# ─────────────────────────────────────────────────────────────────────────────


class NEURAL_NLA_PT_StripStrength(bpy.types.Panel):
    """Per-strip relative-strength panel in the NLA editor Strip tab."""

    bl_idname      = "NEURAL_NLA_PT_strip_strength"
    bl_label       = "Relative Strength"
    bl_space_type  = "NLA_EDITOR"
    bl_region_type = "UI"
    bl_category    = "Strip"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        if getattr(context.scene, "neural_nla_blend_mode", "BLEND") != "BLEND":
            return False
        selected, _ = collect_selected_strips()
        return len(selected) >= 1

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        selected, active = collect_selected_strips()

        # Prefer the active strip; fall back to first selected
        strip = active if (active is not None and active in selected) else selected[0]

        # Strip identity header
        box = layout.box()
        col = box.column(align=True)
        col.label(text=strip.name, icon="ACTION")
        action_name = strip.action.name if strip.action else "—"
        col.label(text=f"Action: {action_name}", icon="ARMATURE_DATA")
        col.label(
            text=f"Frames {int(strip.frame_start)}–{int(strip.frame_end)}",
            icon="PREVIEW_RANGE",
        )

        layout.separator(factor=0.3)
        _draw_strength_ui(layout, strip)


class NEURAL_NLA_PT_StripSkeleton(bpy.types.Panel):
    """Per-strip dataset skeleton conditioning."""

    bl_idname      = "NEURAL_NLA_PT_strip_skeleton"
    bl_label       = "Skeleton"
    bl_space_type  = "NLA_EDITOR"
    bl_region_type = "UI"
    bl_category    = "Strip"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        selected, _ = collect_selected_strips()
        return len(selected) >= 1

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        selected, active = collect_selected_strips()
        strip = active if (active is not None and active in selected) else selected[0]

        layout.label(text=strip.name, icon="ACTION")

        if not hasattr(context.scene, "neural_nla_strip_skeletons"):
            layout.label(text="Reload the BlendAnything add-on", icon="ERROR")
            return

        source_box = layout.box()
        source_col = source_box.column(align=True)
        source_col.label(text="Source BVH:", icon="BONE_DATA")
        owner = find_strip_owner(strip)
        action = strip.action
        source = (
            _resolve_strip_source(context.scene, owner, strip)
            if owner is not None and action is not None else ""
        )
        row = source_col.row(align=True)
        row.label(
            text=os.path.basename(source) if source else "No source linked",
            icon="FILE" if source else "ERROR",
        )
        op = row.operator(
            "neural_nla.set_reference_bvh",
            text="Set...",
            icon="FILEBROWSER",
        )
        op.action_name = action.name if action else ""
        op.object_name = owner.name if owner else ""
        if not source:
            source_col.label(text="Needed to reuse model-native BVH channels", icon="INFO")

        layout.separator(factor=0.3)

        skeleton = _get_strip_skeleton(context.scene, strip.name)
        if skeleton is None:
            layout.label(text="Skeleton settings are not initialized", icon="INFO")
            layout.operator(
                "neural_nla.init_strip_skeleton",
                text="Initialize Skeleton",
                icon="ARMATURE_DATA",
            )
            return

        layout.prop(skeleton, "profile_id", text="Skeleton")
        if skeleton.profile_id == "CUSTOM":
            layout.prop(skeleton, "custom_name")
            layout.label(
                text="Conditioning will be approximated from this BVH",
                icon="INFO",
            )

        col = layout.column(align=True)
        col.label(text="Face Orientation Joints:")
        if owner is not None and owner.type == "ARMATURE":
            for prop_name, label in (
                ("face_upper_right", "Upper-R"),
                ("face_upper_left", "Upper-L"),
                ("face_right", "Lower-R"),
                ("face_left", "Lower-L"),
            ):
                col.prop_search(
                    skeleton,
                    prop_name,
                    owner.data,
                    "bones",
                    text=label,
                    icon="BONE_DATA",
                )
        else:
            col.prop(skeleton, "face_upper_right", text="Upper-R")
            col.prop(skeleton, "face_upper_left", text="Upper-L")
            col.prop(skeleton, "face_right", text="Lower-R")
            col.prop(skeleton, "face_left", text="Lower-L")

        if skeleton.profile_id == "NONE":
            layout.label(text="Choose a skeleton profile", icon="INFO")


class NEURAL_NLA_PT_Panel(bpy.types.Panel):
    bl_idname      = "NEURAL_NLA_PT_panel"
    bl_label       = "Neural NLA Blending"
    bl_space_type  = "NLA_EDITOR"
    bl_region_type = "UI"
    bl_category    = "Neural Blend"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        st             = _read_state()
        ref_strip, target_strips = get_reference_and_targets(context)
        blend_mode = context.scene.neural_nla_blend_mode
        target_only = (
            blend_mode == "RETARGET" and context.scene.neural_nla_target_only
        )
        target_only_strip = _active_selected_strip(context) if target_only else None

        # ── Server connection ─────────────────────────────────────────────────
        box = layout.box()
        row = box.row(align=True)
        row.prop(
            context.scene,
            "neural_nla_show_server",
            text="Server",
            icon=(
                "TRIA_DOWN"
                if context.scene.neural_nla_show_server
                else "TRIA_RIGHT"
            ),
            emboss=False,
        )
        row.label(text=f"v{_ADDON_VERSION}")
        if context.scene.neural_nla_show_server:
            active = _active_server_model()
            col = box.column(align=False)
            col.separator(factor=0.35)
            col.label(
                text=f"Active model: {active or 'None'}",
                icon="CHECKMARK" if active else "INFO",
            )
            col.separator(factor=0.35)
            col.prop(context.scene, "neural_nla_server_url", text="URL")
            col.separator(factor=0.35)
            row = col.row(align=True)
            row.operator(
                "neural_nla.test_connection",
                text="Test Connection",
                icon="URL",
            )
            row.operator(
                "neural_nla.refresh_models",
                text="Refresh Models",
                icon="FILE_REFRESH",
            )
            col.separator(factor=0.5)
            col.label(text="Model", icon="FILE_3D")
            row = col.row(align=True)
            row.prop(context.scene, "neural_nla_server_model", text="")
            row.operator(
                "neural_nla.load_model",
                text="Load",
                icon="IMPORT",
            )

        layout.separator(factor=0.5)

        # ── Setup ─────────────────────────────────────────────────────────────
        box = layout.box()
        col = box.column(align=False)
        col.label(text="Setup", icon="SHADERFX")
        col.separator(factor=0.35)
        col.operator(
            "neural_nla.import_tracked_bvh",
            text="Import BVH",
            icon="IMPORT",
        )
        col.separator(factor=0.6)
        col.label(text="Selected Motions", icon="NLA")
        motion_col = col.column(align=True)
        if target_only:
            destination = context.scene.neural_nla_destination_profile
            destination_label = (
                _profile_display_name(destination)
                if destination != "NONE" else "Choose destination"
            )
            motion_col.label(text=f"Ref:  {destination_label}", icon="RADIOBUT_ON")
            if target_only_strip is not None:
                motion_col.label(
                    text=f"Target 1:  {target_only_strip.name}",
                    icon="RADIOBUT_OFF",
                )
            else:
                motion_col.label(text="Target 1:  Select an active strip", icon="INFO")
        elif ref_strip is None:
            motion_col.label(text="Select NLA motion strips", icon="INFO")
        else:
            motion_col.label(text=f"Ref:  {ref_strip.name}", icon="RADIOBUT_ON")
            for i, s in enumerate(target_strips):
                motion_col.label(
                    text=f"Target {i + 1}:  {s.name}",
                    icon="RADIOBUT_OFF",
                )

        # ── Mode selector ─────────────────────────────────────────────────────
        col.separator(factor=0.7)
        col.label(text="Operation")
        row = col.row(align=True)
        row.prop_enum(context.scene, "neural_nla_blend_mode", "BLEND")
        row.prop_enum(context.scene, "neural_nla_blend_mode", "RETARGET")

        if blend_mode == "RETARGET":
            col.separator(factor=0.5)
            col.prop(context.scene, "neural_nla_target_only", text="Target Only")
            if target_only:
                col.separator(factor=0.35)
                col.prop(
                    context.scene,
                    "neural_nla_destination_profile",
                    text="Reference Skeleton",
                )
                col.label(text="Active strip supplies Target 1 motion", icon="INFO")

        layout.separator(factor=0.5)

        # ── Relative strength ─────────────────────────────────────────────────
        if blend_mode == "BLEND":
            strength_box = layout.box()
            col = strength_box.column(align=True)
            row = col.row(align=True)
            row.label(text="Relative Strength", icon="FCURVE")
            selected_count = (1 if ref_strip is not None else 0) + len(target_strips)
            if selected_count != 2:
                col.label(text="Select exactly two strips", icon="INFO")
                col.label(text="They must overlap on separate NLA tracks")
            else:
                col.prop(
                    context.scene,
                    "neural_nla_strength_mode",
                    text="Editing",
                )

            if ref_strip is not None and len(target_strips) == 1:
                col = strength_box.column(align=True)
                if context.scene.neural_nla_strength_mode == "LINKED":
                    selected = [ref_strip, target_strips[0]]
                    earlier, later = sorted(
                        selected, key=lambda strip: strip.frame_start
                    )
                    overlap = min(
                        earlier.frame_end, later.frame_end
                    ) - max(earlier.frame_start, later.frame_start)
                    col.label(text=f"{earlier.name}  →  {later.name}")
                    if overlap > 0.0 and earlier.frame_end <= later.frame_end:
                        row = col.row(align=True)
                        row.label(text=f"{int(round(overlap))} frames")
                        row.prop(
                            context.scene,
                            "neural_nla_crossfade_shape",
                            text="",
                        )
                        col.operator(
                            "neural_nla.apply_linked_crossfade",
                            icon="IPO_EASE_IN_OUT",
                        )
                    else:
                        col.label(
                            text="Arrange strips with an edge overlap",
                            icon="ERROR",
                        )
                        col.label(text="Use separate NLA tracks")
                else:
                    col.label(
                        text="Edit each strip in Strip → Relative Strength",
                        icon="INFO",
                    )
                    values = _resulting_relative_strength(
                        ref_strip,
                        target_strips[0],
                        context.scene,
                    )
                    if values:
                        col.separator(factor=0.35)
                        plot_box = col.box()
                        plot_box.label(text="Resulting Target Alpha", icon="FCURVE")
                        plot_box.label(text="1.0  Target 1")
                        image = _strength_plot_image(values)
                        plot_box.template_icon(
                            icon_value=plot_box.icon(image),
                            scale=16.0,
                        )
                        plot_box.label(text="0.0  Reference")

            layout.separator(factor=0.5)

        # ── Status / error display ────────────────────────────────────────────
        if st["status"] == "running":
            box = layout.box()
            col = box.column(align=True)
            phase = st.get("phase") or "Waiting for server"
            started_at = float(st.get("started_at") or time.monotonic())
            elapsed = max(0.0, time.monotonic() - started_at)
            col.label(text=phase, icon="SORTTIME")
            if hasattr(col, "progress"):
                progress = max(0.0, min(1.0, float(st.get("progress", 0.0))))
                col.progress(
                    factor=progress,
                    type="BAR",
                    text=f"{progress * 100:.0f}%  |  {elapsed:.1f}s",
                )
            else:
                progress = max(0.0, min(1.0, float(st.get("progress", 0.0))))
                col.label(text=f"{progress * 100:.0f}%  |  {elapsed:.1f}s")
        elif st["status"] == "error":
            box = layout.box()
            box.alert = True
            box.label(text="Error:", icon="ERROR")
            msg = st["error_msg"] or "Unknown error"
            max_chars = 38
            while len(msg) > max_chars:
                box.label(text=msg[:max_chars])
                msg = msg[max_chars:]
            box.label(text=msg)
            box.operator("neural_nla.clear_status", text="Dismiss", icon="X")

        warnings = st.get("warnings") or []
        if warnings:
            box = layout.box()
            row = box.row(align=True)
            row.label(text="Compatibility Warning", icon="ERROR")
            row.operator(
                "neural_nla.dismiss_warnings",
                text="",
                icon="X",
                emboss=False,
            )
            for warning in warnings:
                if isinstance(warning, dict):
                    role = warning.get("role", "Skeleton")
                    role = "Ref" if role == "Reference" else role
                    skeleton = warning.get("skeleton", "Unknown")
                    kind = warning.get("kind", "out_of_distribution")
                    suffix = (
                        "estimated statistics"
                        if kind == "estimated_statistics"
                        else "out of distribution"
                    )
                    box.label(text=f"{role}: {skeleton} - {suffix}")
                else:
                    box.label(text=str(warning), icon="ERROR")

        generated_path = st.get("generated_path") or ""
        if generated_path:
            box = layout.box()
            col = box.column(align=False)
            col.label(text="Generated BVH Saved", icon="CHECKMARK")
            col.label(text=os.path.basename(generated_path), icon="FILE")
            col.label(text=os.path.dirname(generated_path), icon="FILE_FOLDER")
            op = col.operator(
                "wm.path_open",
                text="Open Generated Folder",
                icon="FILE_FOLDER",
            )
            op.filepath = os.path.dirname(generated_path)

        # ── Run button ────────────────────────────────────────────────────────
        row = layout.row()
        row.scale_y = 1.6
        selection_ok = (
            target_only_strip is not None
            if target_only
            else ref_strip is not None and len(target_strips) >= 1
        )
        destination_ok = (
            context.scene.neural_nla_destination_profile != "NONE"
            if target_only else True
        )
        row.enabled = selection_ok and destination_ok and st["status"] != "running"
        row.operator("neural_nla.run_blend", icon="SHADERFX")

        # ── Advanced settings ─────────────────────────────────────────────────
        box = layout.box()
        row = box.row()
        row.prop(
            context.scene,
            "neural_nla_show_advanced",
            icon=(
                "TRIA_DOWN"
                if context.scene.neural_nla_show_advanced
                else "TRIA_RIGHT"
            ),
            emboss=False,
        )
        if context.scene.neural_nla_show_advanced:
            col = box.column(align=False)
            col.separator(factor=0.45)
            col.label(text="Output Reconstruction", icon="ARMATURE_DATA")
            col.separator(factor=0.25)
            row = col.row(align=True)
            row.prop_enum(context.scene, "neural_nla_output_mode", "POSITIONS_IK")
            row.prop_enum(context.scene, "neural_nla_output_mode", "ROTATIONS")
            if context.scene.neural_nla_output_mode == "POSITIONS_IK":
                col.separator(factor=0.4)
                col.prop(context.scene, "neural_nla_ik_iterations", text="IK Iterations")
            col.separator(factor=0.8)
            col.label(text="DDIM Inversion + DDIM Sampling", icon="MOD_FLUIDSIM")
            col.separator(factor=0.25)
            col.prop(
                context.scene,
                "neural_nla_ddim_inversion_policy",
                text="Apply",
            )
            col.label(text="Enabled policies also use transition SLERP", icon="INFO")

        if not HAS_REQUESTS:
            layout.label(text="⚠ 'requests' not installed", icon="ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# Depsgraph handler — keep influence F-curves in sync with strip position
# ─────────────────────────────────────────────────────────────────────────────

_syncing: bool = False  # re-entrancy guard for depsgraph handler

_SYNC_POLL_INTERVAL: float = 0.05  # seconds between timer sync polls (~20 Hz)


def _timer_strip_sync() -> float:
    """
    Polling fallback for influence-curve sync.

    Runs at _SYNC_POLL_INTERVAL regardless of depsgraph activity.  When a
    strip's frame boundaries differ from the last-recorded values, a null
    property write fires _on_strength_update → _apply_profile_to_strip, which
    re-anchors the keyframes.  Using the property-update path (rather than
    calling _apply_profile_to_strip directly) ensures Blender processes the
    write outside any active modal operator, avoiding the occasional undo-stack
    conflict that causes direct F-curve writes to be silently reverted.
    """
    scene = getattr(bpy.context, "scene", None)
    if scene is None or not getattr(scene, "neural_nla_strengths", None):
        return _SYNC_POLL_INTERVAL
    for entry in scene.neural_nla_strengths:
        strip = find_strip_by_name(entry.strip_name)
        if strip is None:
            continue
        if (abs(entry.last_frame_start - strip.frame_start) > 0.001 or
                abs(entry.last_frame_end - strip.frame_end) > 0.001):
            # Null write: value unchanged, but fires _on_strength_update which
            # calls _apply_profile_to_strip and updates last_frame_start/end.
            entry.fade_in = entry.fade_in
    return _SYNC_POLL_INTERVAL


@bpy.app.handlers.persistent
def _sync_influence_positions(scene, depsgraph) -> None:
    """
    After any depsgraph update, check whether any strip that has a Neural NLA
    strength entry has been shifted or scaled.  If so, re-apply its profile so
    the influence F-curve keyframes stay aligned with the strip boundaries.

    The re-entrancy guard (_syncing) prevents the F-curve write from triggering
    a second call to this handler in the same Blender tick.
    """
    global _syncing
    if _syncing:
        return
    if not getattr(scene, "neural_nla_strengths", None):
        return  # no entries managed by this add-on — nothing to do

    _syncing = True
    try:
        for obj in scene.objects:
            anim = obj.animation_data
            if not anim:
                continue
            for track in anim.nla_tracks:
                for strip in track.strips:
                    entry = _get_strength(scene, strip.name)
                    if entry is None:
                        continue
                    # Skip strips whose F-curve hasn't been written yet
                    if strip.fcurves.find("influence") is None:
                        continue
                    # Detect movement by comparing stored boundaries against the
                    # current strip position.  This avoids reading keyframe
                    # positions, which can be stale or in an intermediate state
                    # during Blender's internal depsgraph processing.
                    if (abs(entry.last_frame_start - strip.frame_start) > 0.001 or
                            abs(entry.last_frame_end   - strip.frame_end)   > 0.001):
                        _apply_profile_to_strip(strip, entry)
    finally:
        _syncing = False


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

_CLASSES = [
    BlendAnythingPreferences,
    NeuralArmatureReference,         # PropertyGroup — must precede CollectionProperty use
    NeuralActionSource,
    NeuralStripSkeletonEntry,
    NeuralStripStrengthEntry,       # PropertyGroup — must precede CollectionProperty use
    NEURAL_NLA_OT_TestConnection,
    NEURAL_NLA_OT_RefreshModels,
    NEURAL_NLA_OT_LoadModel,
    NEURAL_NLA_OT_StrengthPreset,
    NEURAL_NLA_OT_ApplyInfluence,
    NEURAL_NLA_OT_ApplyLinkedCrossfade,
    NEURAL_NLA_OT_RunBlend,
    NEURAL_NLA_OT_ClearStatus,
    NEURAL_NLA_OT_DismissWarnings,
    NEURAL_NLA_OT_SetReferenceBVH,
    NEURAL_NLA_OT_InitStripSkeleton,
    NEURAL_NLA_OT_ImportTrackedBVH,
    NEURAL_NLA_PT_StripSkeleton,
    NEURAL_NLA_PT_StripStrength,
    NEURAL_NLA_PT_Panel,
]


def _unregister_stale_classes() -> None:
    """Remove classes left resident when add-on files were replaced in place."""
    for cls in reversed(_CLASSES):
        existing = getattr(bpy.types, cls.__name__, None)
        if existing is None or existing is cls:
            continue
        try:
            bpy.utils.unregister_class(existing)
        except RuntimeError:
            pass

    # Version 1.1.0 registered this obsolete preferences class from
    # blendanything_client.addon. It can survive an in-place update because the
    # new module no longer has the class in its normal unregister list.
    stale_preferences = getattr(bpy.types, "NeuralNLAPreferences", None)
    if stale_preferences is not None:
        try:
            bpy.utils.unregister_class(stale_preferences)
        except RuntimeError:
            pass


def register() -> None:
    _unregister_stale_classes()
    for cls in _CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Scene.neural_nla_server_url = bpy.props.StringProperty(
        name="Server URL",
        description="Base URL of the neural blending server",
        default="http://localhost:8000",
    )
    bpy.types.Scene.neural_nla_server_model = bpy.props.EnumProperty(
        name="Server Model",
        description="Model folder available under the server save directory",
        items=_server_model_items,
    )
    bpy.types.Scene.neural_nla_show_server = bpy.props.BoolProperty(
        name="Server",
        description="Show server URL and model controls",
        default=False,
    )

    # Per-strip strength storage — CollectionProperty on Scene (ID type).
    # NlaStrip is a non-ID struct and cannot hold custom bpy.props.
    bpy.types.Scene.neural_nla_strengths = bpy.props.CollectionProperty(
        type=NeuralStripStrengthEntry,
        name="Neural NLA Strip Strengths",
        description="Per-strip strength envelopes managed by the Neural NLA Blending add-on",
    )

    # Per-armature source BVH. Both ref and target armatures store their original
    # model-native motion here, keyed by object name.
    bpy.types.Scene.neural_nla_armature_refs = bpy.props.CollectionProperty(
        type=NeuralArmatureReference,
        name="Neural NLA Armature References",
        description="Original source BVH per armature, used to select the matching "
                    "cond.npy skeleton and restore Blender-lost leaf joints",
    )

    bpy.types.Scene.neural_nla_action_sources = bpy.props.CollectionProperty(
        type=NeuralActionSource,
        name="Neural NLA Action Sources",
        description="Original model-native BVH associated with each Blender action",
    )

    bpy.types.Scene.neural_nla_strip_skeletons = bpy.props.CollectionProperty(
        type=NeuralStripSkeletonEntry,
        name="Neural NLA Strip Skeletons",
        description="Dataset skeleton and face joints configured for each NLA strip",
    )

    bpy.types.Scene.neural_nla_show_advanced = bpy.props.BoolProperty(
        name="Advanced",
        description="Show advanced output settings",
        default=False,
    )

    bpy.types.Scene.neural_nla_output_mode = bpy.props.EnumProperty(
        name="Output Mode",
        description="How the model output is converted back to a skeleton",
        items=[
            ("POSITIONS_IK", "Positions + IK",
             "Recover global joint positions from features, then run Inverse Kinematics "
             "to obtain joint rotations. Slower but more accurate.",
             "OUTLINER_OB_ARMATURE", 0),
            ("ROTATIONS",    "Rotations",
             "Recover joint rotations directly from the 6D rotation features. "
             "Faster but may show more foot-sliding.",
             "IPO_BEZIER", 1),
        ],
        default="POSITIONS_IK",
    )

    bpy.types.Scene.neural_nla_ik_iterations = bpy.props.IntProperty(
        name="IK Iterations",
        description="Number of CCD-IK iterations used to fit the skeleton to the recovered positions. "
                    "Higher values are more accurate but slower.",
        min=1, max=1000, default=150, step=10,
    )
    bpy.types.Scene.neural_nla_ddim_inversion_policy = bpy.props.EnumProperty(
        name="DDIM Inversion + Sampling",
        description=(
            "When to use DDIM-inverted controls, DDIM sampling, and transition SLERP"
        ),
        items=[
            (
                "NEVER",
                "Never",
                "Skip inversion and SLERP; keep the configured sampler",
            ),
            (
                "SAME_SKELETON",
                "On Same Skeleton",
                "Use DDIM when an active control shares the reference/output skeleton; "
                "cross-skeleton blends combine reference inversion with Gaussian noise",
            ),
            (
                "ALWAYS",
                "Always",
                "Use DDIM inversion, DDIM sampling, and SLERP for every blend",
            ),
        ],
        default="SAME_SKELETON",
    )

    bpy.types.Scene.neural_nla_blend_mode = bpy.props.EnumProperty(
        name="Mode",
        description=(
            "Blend: mix ref and tgt motions with temporal crossfade (control_mode=both). "
            "Retarget: apply the target motion style to the reference skeleton "
            "(control_mode=tgt — strip timeline placement is ignored)"
        ),
        items=[
            ("BLEND",   "Blend",    "Crossfade between reference and target motions", "FORCE_MAGNETIC", 0),
            ("RETARGET","Retarget", "Transfer target motion to the reference skeleton",  "ARMATURE_DATA",  1),
        ],
        default="BLEND",
    )
    bpy.types.Scene.neural_nla_strength_mode = bpy.props.EnumProperty(
        name="Relative Strength Editing",
        description="How relative-strength curves are configured for a two-strip blend",
        items=[
            (
                "LINKED",
                "Linked Crossfade",
                "Create complementary fade-out and fade-in curves over the overlap",
            ),
            (
                "INDEPENDENT",
                "Independent",
                "Edit each strip's relative-strength curve separately",
            ),
        ],
        default="LINKED",
    )
    bpy.types.Scene.neural_nla_crossfade_shape = bpy.props.EnumProperty(
        name="Crossfade Shape",
        description="Interpolation used by both linked relative-strength curves",
        items=[
            ("SMOOTH", "Smooth", "Smooth ease-out and ease-in"),
            ("LINEAR", "Linear", "Straight linear crossfade"),
        ],
        default="SMOOTH",
    )
    bpy.types.Scene.neural_nla_target_only = bpy.props.BoolProperty(
        name="Target Only",
        description=(
            "Use the active strip as Target 1 and retarget it onto a selected "
            "dataset reference skeleton"
        ),
        default=False,
    )
    bpy.types.Scene.neural_nla_destination_profile = bpy.props.EnumProperty(
        name="Reference Skeleton",
        description="Dataset skeleton that receives the active target motion",
        items=_destination_profile_items,
    )

    # Enable "Show Strip Curves" in every open NLA editor so influence curves
    # are visible immediately without the user having to find the overlay toggle.
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "NLA_EDITOR":
                for space in area.spaces:
                    if hasattr(space, "show_strip_curves"):
                        space.show_strip_curves = True

    # Depsgraph handler: immediate sync on depsgraph updates.
    bpy.app.handlers.depsgraph_update_post.append(_sync_influence_positions)

    # Timer fallback: catches moves that the depsgraph handler misses (e.g.
    # final position after a modal G-move that commits outside a depsgraph tick).
    if not bpy.app.timers.is_registered(_timer_strip_sync):
        bpy.app.timers.register(
            _timer_strip_sync,
            first_interval=_SYNC_POLL_INTERVAL,
            persistent=True,
        )


def unregister() -> None:
    global _strength_plot_signature
    if _sync_influence_positions in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_sync_influence_positions)
    # Cancel any running timers before unregistering classes
    for timer_fn in (_timer_poll_result, _timer_strip_sync):
        if bpy.app.timers.is_registered(timer_fn):
            bpy.app.timers.unregister(timer_fn)

    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass

    image = bpy.data.images.get(_STRENGTH_PLOT_IMAGE)
    if image is not None:
        bpy.data.images.remove(image)
    _strength_plot_signature = None
    _unregister_stale_classes()

    for prop in (
        "neural_nla_strengths",
        "neural_nla_server_url",
        "neural_nla_server_model",
        "neural_nla_show_server",
        "neural_nla_show_advanced",
        "neural_nla_output_mode",
        "neural_nla_ik_iterations",
        "neural_nla_ddim_inversion_policy",
        "neural_nla_blend_mode",
        "neural_nla_strength_mode",
        "neural_nla_crossfade_shape",
        "neural_nla_target_only",
        "neural_nla_destination_profile",
        "neural_nla_armature_refs",
        "neural_nla_action_sources",
        "neural_nla_strip_skeletons",
    ):
        try:
            delattr(bpy.types.Scene, prop)
        except AttributeError:
            pass


if __name__ == "__main__":
    register()
