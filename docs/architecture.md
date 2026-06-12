# Architecture

```
BlendAnything/
├── blendanything_client/   # Blender add-on package          → README
│   ├── __init__.py         #   register/unregister entry point
│   ├── addon.py            #   panels, operators, NLA import/export
│   ├── network.py          #   background HTTP jobs + progress state
│   ├── profiles.py         #   skeleton catalog + BVH profile matching
│   └── data/               #   Blender-safe skeleton catalogs (JSON)
├── blendanything_server/   # FastAPI backend package          → README
│   ├── app.py              #   API routes, job lifecycle, progress
│   ├── bvh_pipeline.py     #   BVH preprocessing / reconstruction
│   └── model_bridge.py     #   neural model loading + inference bridge
├── neural_motion_blending/ # Neural Motion Blending model (git submodule)
├── data/                   # server-side runtime conditioning  → README
├── devtools/               # diagnostics + round-trip checks    → README
└── samples/                # example BVH clips
```

Each package's own `README.md` documents its modules in detail. The client
never imports server modules or model dependencies.

---

## Data flow

```
Blender NLA Editor
  │  select 2+ strips  (active = reference, rest = targets)
  ▼
blendanything_client
  ├─ export each strip's action as a temporary BVH  (clip range only)
  ├─ POST /blend/jobs  ──►  blendanything_server.app
  │     reference_bvh + target_bvh[0..N] + metadata JSON
  └─ poll progress once per second
        └─ on result: retain BVH in Blender user data
                     → import as Neural_Result_<timestamp>
                     → create a tracked, ready-to-use NLA strip
                     → place in "Neural Blending Outputs"
```

---

## Extending

- **Model integration** — `blendanything_server/model_bridge.py`.
- **API / uploads / progress / results** — `blendanything_server/app.py`.
- **BVH feature conversion & reconstruction** — `blendanything_server/bvh_pipeline.py`.

The underlying cross-topology model
([Neural Motion Blending](https://mmlab-cv.github.io/neural_motion_blending/),
building on [AnyTop](https://anytop2025.github.io/Anytop-page/)) lives in the
`neural_motion_blending` submodule.
