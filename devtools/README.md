# Development Tools

These scripts validate or diagnose BlendAnything. They are not imported by the
Blender add-on or the production FastAPI server.

| Tool | Purpose |
|---|---|
| `test_blend_client.py` | Manually submits BVH files to a running server. |
| `blender_roundtrip.py` | Runs inside Blender and reproduces BVH import/export. |
| `verify_blender_roundtrip.py` | Launches Blender and compares round-trip model features. |
| `diag_bvh_npy.py` | Diagnoses BVH-to-feature conversion and skeleton ordering. |
| `make_user_space_bvh.py` | Generates transformed BVH fixtures for conversion testing. |
| `test_blender_addon.py` | Headless registration, UI state, generated-result tracking, and relative-strength preview smoke test. |
| `test_blender_zip.py` | Installs and enables the packaged add-on in a clean Blender profile. |
| `test_relative_strength.py` | Verifies server-side relative-strength alpha normalization. |
| `test_model_catalog.py` | Verifies save-folder discovery and runtime model switching. |
| `test_skeleton_profiles.py` | Verifies combined Truebones/Mixamo catalogs and Coyote matching. |
| `test_custom_skeleton.py` | Verifies request-scoped user skeleton conditioning. |
| `diag_custom_skeleton.py` | Compares a known skeleton against the custom/unknown path. |
| `diag_retarget_custom_ablation.py` | Runs known/estimated/copied-stat retarget ablations. |

The add-on test also verifies that linked crossfades produce a smooth
monotonic target-alpha preview, that independent profile changes invalidate
the cached graph, and that generated BVHs become reusable tracked NLA strips.

Examples:

```bash
python devtools/test_blend_client.py
python devtools/verify_blender_roundtrip.py /path/to/blender
python devtools/make_user_space_bvh.py input.bvh output.bvh
/path/to/blender --background --factory-startup --python devtools/test_blender_addon.py
```

The files under `scripts/` are compatibility wrappers for older commands.
