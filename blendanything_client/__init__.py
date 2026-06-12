"""BlendAnything Blender add-on package.

The heavy Blender module is imported lazily so non-Blender utilities such as
``blendanything_client.profiles`` can be used from normal Python.
"""

bl_info = {
    "name": "BlendAnything",
    "author": "Luca Cazzola",
    "version": (1, 0, 0),
    "blender": (4, 5, 4),
    "location": "NLA Editor → N-panel → Neural Blend",
    "description": "Cross-topology neural motion blending from the SIGGRAPH Posters 2026 work",
    "doc_url": "https://mmlab-cv.github.io/BlendAnything/",
    "support": "COMMUNITY",
    "category": "Animation",
}


def register() -> None:
    from .addon import register as _register

    _register()


def unregister() -> None:
    from .addon import unregister as _unregister

    _unregister()


__all__ = ["bl_info", "register", "unregister"]
