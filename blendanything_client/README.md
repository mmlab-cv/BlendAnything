# Blender Client

Install this directory as the Blender add-on package.

- Add-on release: **1.0.0**
- Tested and supported Blender version: **4.5.4 LTS**

- `__init__.py`: lightweight Blender registration entry point.
- `addon.py`: Blender properties, operators, panels, NLA import/export, and
  result import.
- `network.py`: background HTTP requests, progress polling, and shared request
  state.
- `profiles.py`: Truebones/Mixamo profile loading, custom skeleton choice, and
  BVH skeleton matching.
- `data/*_skeletons.json`: client-safe skeleton catalogs.

The client does not import server modules or model dependencies.
The add-on metadata identifies Luca Cazzola as maintainer and links both the
[project website](https://mmlab-cv.github.io/BlendAnything/) and
[maintainer GitHub profile](https://github.com/LuCazzola).
Skeleton dropdowns group clean character names under native **Truebones** and
**Mixamo** headings, in that order.
Face-orientation landmarks use searchable `Upper-R`, `Upper-L`, `Lower-R`, and
`Lower-L` dropdowns backed by the selected strip's armature bones.

Generated motions are retained as BVH files in Blender's per-user data
directory under `blendanything/generated/`. Imported result actions and
armatures reference that durable file, so later blends reuse the generated
model-native motion instead of exporting it back through Blender. Each result
is also placed in a ready-to-use NLA strip with skeleton and default strength
settings, and the Neural Blend panel displays its saved path.

For two selected strips, the main panel provides a linked crossfade workflow.
The per-strip **Relative Strength** controls remain available for independent
weighting in Blend mode; they are hidden in Retarget mode. The server
normalizes both curves into the model's alpha schedule. In Independent mode,
the Neural Blend panel shows a compact preview of the resulting normalized
target-alpha function.

The preview is regenerated whenever either independent profile changes. It
uses the same overlap-domain normalization and zero-sum fallback as the server.

Advanced settings include a server-backed model dropdown. Use its refresh
button to discover save-folder models, then **Load Selected Model** to activate
the chosen checkpoint.

Job status is polled once per second. The panel reports semantic encoding,
DDIM inversion, DDIM or DDPM sampling, and IK post-processing with weighted
percentage progress and elapsed time.
