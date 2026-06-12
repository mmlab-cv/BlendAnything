# Usage & UI Guide

How the add-on works inside Blender's NLA Editor. For setup, see the
[main README](../README.md#quick-start).

This guide covers BlendAnything **1.0.0**, tested on Blender **4.5.4 LTS**.

---

## Workflow

1. **Import BVH** — **N-panel → Neural Blend → Import BVH**. Creates a tracked
   NLA strip and retains the original filepath so model-native BVH channels can
   be reused later. The panel stays available with an empty scene, so the
   importer doubles as a new-scene entry point.
2. **Arrange strips** — lay clips on NLA tracks. The **active** strip
   (last LMB-clicked) is the reference; the rest are targets.
3. **Configure per strip** — set the skeleton and relative strength
   (see below).
4. **Run** — **Neural Blend → Run Neural Blend**. Strips export as BVH, POST to
   the server, and the result imports back automatically into
   *Neural Blending Outputs* — without freezing Blender's main thread.

Existing BVH imports can be linked explicitly from the per-strip **Skeleton**
panel. The client never searches the filesystem for same-named BVH files.

---

## Modes

| Mode | Selection | Behaviour |
|---|---|---|
| **Blend** | 2+ strips | Active strip is the reference; the others are targets. |
| **Retarget** | 2 strips | Target motion is applied to the active reference skeleton. |
| **Retarget → Target Only** | 1 strip | Active strip becomes **Target 1**. Pick a **Reference Skeleton** (e.g. `Skunk`); the target motion is applied to it with no reference strip. |

---

## Panels

### Neural Blend *(N-panel → Neural Blend)*

| Element | Purpose |
|---|---|
| **Import BVH** | Creates a tracked NLA strip with skeleton + strength defaults |
| Strip names | Shows the reference/source strip and all target strips |
| **Run Neural Blend** | Exports strips as BVH, POSTs, imports the result |
| Server URL | Editable endpoint with a connection-test (globe) button |

> With **Retarget → Target Only**, the active strip becomes **Target 1** and
> the selected dataset profile is shown as the reference.

### Relative Strength *(N-panel → Strip)*

Per-strip weight envelope written to the strip's built-in `influence` F-curve,
so Blender's own "Show Strip Curves" overlay visualises it. The server
normalizes overlapping reference/target strengths per frame.

- **Linked Crossfade** (default, two-strip case) — select two strips with an
  edge overlap, choose Smooth or Linear, then **Apply Linked Crossfade** to
  write complementary fade-out/fade-in curves across the overlap.
- **Independent** — manual per-strip control. With exactly two strips selected,
  the Neural Blend panel shows a live plot of the normalized target alpha
  `target / (reference + target)`.

#### Strength profiles

| Profile | Parameters | Description |
|---|---|---|
| **Constant** | Value | Flat influence for the whole strip |
| **Linear** | Peak · Floor · Start · End | Trapezoidal ramp, linear interpolation |
| **Smooth** | Peak · Floor · Fade In · Fade Out | Trapezoid with configurable ease slopes |

Start/End (Linear) and Fade In/Out (Smooth) are 0–1 fractions of strip
duration; if their sum exceeds 1 they are scaled down proportionally.

**Smooth — advanced:**

| Parameter | Range | Description |
|---|---|---|
| **Floor** | 0 – 1 | Influence at strip boundaries (replaces the hard zero) |
| **Shape** | Smooth / Ease In / Ease Out / Ease In-Out | Fade-segment interpolation |
| **Overshoot** | −1 – 1 | Nudges Bezier handles beyond peak for bounce/anticipation (Smooth shape only) |

**Quick presets:** `↗ In` · `↘ Out` · `~ Ease` · `1.0` · `0.5` · `0.0` —
they respect the active profile and don't force a switch unless required.

#### DDIM Inversion policy *(advanced)*

| Policy | Behaviour |
|---|---|
| **Never** | Generation starts from fresh random noise. |
| **On Same Skeleton** *(default)* | DDIM when an active control shares the reference/output skeleton. Cross-skeleton: reference is inverted and SLERPed toward Gaussian target noise. |
| **Always** | Inversion even across skeletons — compatible reference regions use inverted noise, incompatible target regions stay Gaussian (`mix.py` aligned-noise behaviour). |

When the policy enables inversion, the server also uses DDIM sampling and
transition SLERP; otherwise it keeps the configured sampler and a normal
Gaussian transition. Same-skeleton matching uses the dataset-qualified
character identity and accepts policy identifiers from earlier add-on versions.

#### Server model catalog *(advanced)*

Refreshing queries the server's `neural_motion_blending/save/` directory; each
folder containing exactly one `model*.pt` appears by folder name.
**Load Selected Model** swaps the server pipeline to that checkpoint.

### Skeleton *(N-panel → Strip)*

See [skeletons.md](skeletons.md) for skeleton selection, face-orientation
joints, and custom / out-of-distribution skeletons.
