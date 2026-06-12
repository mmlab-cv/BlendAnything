# Skeletons & Conditioning

The per-strip **Skeleton** panel (*N-panel → Strip*) carries the only
model-conditioning information sent to the server.

---

## Per-strip skeleton settings

- **Skeleton** — a dataset-qualified profile. The menu lists clean character
  names under separate **Truebones** and **Mixamo** headings, Truebones first.
- **Face Orientation Joints** — `Upper-R`, `Upper-L`, `Lower-R`, `Lower-L`
  landmarks. Each is a searchable dropdown populated from the owning armature's
  bone hierarchy.
- **Source BVH** — the motion file linked to the strip. Inspect or replace it
  here; the client never searches the filesystem for same-named BVHs.
  Coordinate conversion stays an automatic transport concern.

For recognized Truebones/Mixamo BVHs, the skeleton and its standard face joints
are filled automatically from joint names and parent topology. Selecting a
dataset skeleton manually also fills its default face joints; all four fields
stay editable per strip.

---

## Custom / unrecognized skeletons

Select **Custom / User Skeleton**, assign a stable name, and specify all four
face joints. The server then derives, request-scoped, the topology, offsets,
graph relations, kinematic chains, and a first-frame template from the uploaded
BVH.

Normalization is estimated per joint by matching each custom joint to similar
training joints (semantic role, side, topology, depth, leaf/branch status, name
tokens). The prior is calibrated against the clip's empirical per-joint
statistics with regularization ≈ ten prior frames, so Blender shows an
out-of-distribution note while still running inference.

Custom skeletons can be uploaded strips, including **Target 1** in Target Only
mode. A Target Only *destination* stays catalog-backed, because no destination
BVH exists from which to derive its structure.

---

## Conditioning data

Runtime conditioning lives in [`data/`](../data/README.md):

- `data/truebones_cond.npy`, `data/mixamo_cond.npy` — per-skeleton topology,
  offsets, T-pose features, normalization stats, foot/face metadata, and graph
  relations. The server picks the file by the active model's training dataset
  and makes both catalogs available during inference.
- `blendanything_client/data/truebones_skeletons.json`,
  `blendanything_client/data/mixamo_skeletons.json` — Blender-safe matching
  subsets shipped with the add-on (joint names, parent arrays, face/foot
  joints).

Skeletons outside the checkpoint's training dataset stay usable and produce a
compact, non-error compatibility warning in Blender.
