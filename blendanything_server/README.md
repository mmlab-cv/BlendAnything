# Server Package

- `app.py`: FastAPI routes, upload validation, job lifecycle, progress
  reporting, and result delivery.
- `model_bridge.py`: model initialization plus high-level blend and retarget
  calls.
- `bvh_pipeline.py`: BVH-to-feature conversion, skeleton reconciliation, and
  output reconstruction.

Reference and target relative-strength samples are resampled onto their
processed controls and normalized per frame in `model_bridge.py`. The resulting
target ratio replaces the generated alpha tensor without modifying the
`neural_motion_blending` submodule.

The API also resolves the client's DDIM policy. **On Same Skeleton** enables
DDIM when an active control shares the generated reference/output skeleton.
For cross-topology blends, the compatible reference inversion is SLERPed
toward Gaussian target noise. **Always** forces the DDIM path, including
target-only cross-skeleton retargeting. An enabled policy consistently
activates DDIM inversion, DDIM sampling, and transition SLERP.

Progress jobs report semantic encoding, DDIM inversion, DDIM/DDPM sampling,
and IK post-processing. Component weights follow measured runtime proportions;
the user-facing phase labels do not expose average-time estimates.

`GET /models` catalogs folders under `neural_motion_blending/save/` that
contain exactly one `model*.pt`. `POST /models/load` accepts a folder name and
loads that checkpoint under the model-worker lock. Set `MODEL_SAVE_ROOT` to
override the catalog directory. At startup, an explicit `MODEL_PATH` takes
priority; otherwise the server automatically loads `truebones_attnpool` when
that valid model folder is present.

Run the server with:

```bash
uvicorn blendanything_server.app:app --host 0.0.0.0 --port 8000
```

The root `server.py` remains a compatibility entry point.

Known Truebones and Mixamo profiles remain usable outside the active
checkpoint's training distribution; this produces an advisory client warning,
not a server error. Request-scoped custom skeletons use derived topology and
estimated per-joint normalization statistics.
