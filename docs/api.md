# Server API

FastAPI backend in [`blendanything_server/`](../blendanything_server/README.md).
Listens on `http://localhost:8000` by default.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET`  | `/health` | Liveness probe → `{"status": "online"}` |
| `GET`  | `/models` | List loadable save-folder models and the active model |
| `POST` | `/models/load` | Load a model by folder name |
| `POST` | `/blend/jobs` | Submit a progress-aware blend job |
| `GET`  | `/blend/jobs/{id}` | Poll status, phase, and normalized progress |
| `GET`  | `/blend/jobs/{id}/result` | Download a completed job's `.bvh` |
| `POST` | `/blend` | Blocking multipart blend (compatibility) |

The add-on uses the progress-aware flow: `POST /blend/jobs` →
`GET /blend/jobs/{id}` → `GET /blend/jobs/{id}/result`. The blocking
`POST /blend` is retained for compatibility.

`/models` catalogs folders under `neural_motion_blending/save/` containing
exactly one `model*.pt`. Set `MODEL_SAVE_ROOT` to override the catalog
directory. When `MODEL_PATH` is unset, the server automatically loads
`truebones_attnpool` if that folder is available in the catalog.

---

## Request payload

`POST /blend` and `/blend/jobs` take multipart: `reference_bvh`, one or more
`target_bvh`, and a `metadata` JSON part.

```json
{
  "reference": {
    "name": "Run_Strip",
    "action": "Run",
    "frame_start": 10, "frame_end": 90,
    "action_frame_start": 1, "action_frame_end": 80,
    "repeat": 2.0, "scale": 1.0,
    "use_reverse": false, "extrapolation": "HOLD",
    "strength": { "profile": "SMOOTH", "samples": [] }
  },
  "targets": [
    { "name": "Walk_Strip" },
    { "name": "Jump_Strip" }
  ]
}
```

- The server applies `use_reverse` and `repeat` to each raw BVH before passing
  it to the neural model.
- `strength.samples` holds one influence value per frame across the strip's
  scene range. Reference/target samples are resampled onto the processed
  controls and normalized per frame; the resulting ratio replaces the generated
  alpha tensor (no `neural_motion_blending` submodule changes).

---

## Progress phases

Jobs report **semantic encoding**, **DDIM inversion**, **DDIM/DDPM sampling**,
and **IK post-processing**. Component weights follow measured runtime
proportions; phase labels do not expose average-time estimates. The client
polls once per second and shows weighted percentage progress with elapsed time.

---

## Result delivery

Downloaded results are copied into Blender's per-user data directory at
`blendanything/generated/`. The imported action and armature keep this BVH as
their tracked model-native source for future operations, land in a ready-to-use
NLA strip with skeleton + default strength, and the Neural Blend panel reports
the saved path with an **Open Generated Folder** button.
