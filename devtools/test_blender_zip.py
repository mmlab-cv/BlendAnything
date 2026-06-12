"""Install and enable the packaged add-on in a fresh headless Blender profile."""

import os
import sys

import bpy


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not args:
        raise SystemExit("Pass the add-on zip path after --")

    archive = os.path.abspath(args[0])
    bpy.ops.preferences.addon_install(filepath=archive, overwrite=True)
    bpy.ops.preferences.addon_enable(module="blendanything_client")

    from blendanything_client import addon

    assert hasattr(bpy.types.Scene, "neural_nla_server_url")
    preferences = bpy.context.preferences.addons["blendanything_client"].preferences
    assert isinstance(preferences, addon.BlendAnythingPreferences)
    assert addon.bl_info["author"] == "Luca Cazzola"
    assert addon.bl_info["version"] == (1, 0, 0)
    assert addon.bl_info["blender"] == (4, 5, 4)
    assert addon._ADDON_VERSION == "1.0.0"
    assert addon.bl_info["doc_url"] == "https://mmlab-cv.github.io/BlendAnything/"
    assert addon._MAINTAINER_URL == "https://github.com/LuCazzola"
    assert addon._server_url(bpy.context) == "http://localhost:8000"
    assert not hasattr(bpy.types, "NeuralNLAPreferences")

    bpy.ops.preferences.addon_disable(module="blendanything_client")
    assert not hasattr(bpy.types.Scene, "neural_nla_server_url")
    print("BlendAnything packaged add-on smoke test passed")


if __name__ == "__main__":
    main()
