# Runtime Data

This directory contains compact, derived artifacts needed by the
BlendAnything inference server and Blender add-on.

## Included

- `truebones_cond.npy`
  - Per-skeleton model conditioning.
  - Contains topology, offsets, T-pose features, normalization statistics,
    foot/face metadata, and graph relations.
  - Does not contain source BVH clips or complete motion sequences.

- `mixamo_cond.npy`
  - Equivalent runtime conditioning derived from the Mixamo training data.
  - Used by checkpoints whose training dataset is Mixamo.

- `../blendanything_client/data/truebones_skeletons.json`
  - Blender-safe subset packaged with the client for profile matching and UI defaults.
  - Contains joint names, parent arrays, face joints, and foot joints.

- `../blendanything_client/data/mixamo_skeletons.json`
  - Blender-safe Mixamo profile subset used for matching and UI defaults.

## Potential Additional Caches

- `truebones_t5_cache.npz` / `mixamo_t5_cache.npz`
  - Joint-name embeddings derived from the profile joint names.
  - Optional; the current inference path computes these with T5 at runtime.

## Not Required for Inference

- Processed or raw BVH motion collections
- Per-motion `.npy` tensors
- Rendered animation previews
- Training optimizer state (`opt*.pt`)
- Evaluation benchmark lists and generated samples

Client-generated BVHs are not stored here. Blender retains them in its
per-user data directory under `blendanything/generated/`.

The model checkpoint and any derived dataset metadata must still be reviewed
under the licenses and distribution terms of the model, source dataset, and
tools used to create them. This manifest is a technical inventory, not a legal
determination.
