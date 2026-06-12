"""Smoke-test server model discovery and runtime selection."""

import asyncio
import os
import sys
import tempfile


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blendanything_server import app as server


def main() -> None:
    old_root = server._MODEL_SAVE_ROOT
    old_path = server._MODEL_PATH
    with tempfile.TemporaryDirectory() as model_root:
        preferred = os.path.join(model_root, "truebones_attnpool")
        fallback = os.path.join(model_root, "other_model")
        invalid = os.path.join(model_root, "multiple_checkpoints")
        for folder in (preferred, fallback, invalid):
            os.makedirs(folder)
        preferred_checkpoint = os.path.join(preferred, "model000001.pt")
        fallback_checkpoint = os.path.join(fallback, "model000002.pt")
        open(preferred_checkpoint, "wb").close()
        open(fallback_checkpoint, "wb").close()
        open(os.path.join(invalid, "model000003.pt"), "wb").close()
        open(os.path.join(invalid, "model000004.pt"), "wb").close()

        server._MODEL_SAVE_ROOT = model_root
        server._MODEL_PATH = ""
        models = server._available_models()
        assert set(models) == {"other_model", "truebones_attnpool"}
        assert server._startup_model() == (
            "truebones_attnpool",
            preferred_checkpoint,
        )

        old_loader = server._load_model_checkpoint
        old_pipeline = server._pipeline
        old_name = server._active_model_name
        old_dataset = server._active_model_dataset
        marker = object()
        try:
            server._pipeline = None
            server._active_model_name = ""
            server._active_model_dataset = ""
            server._load_model_checkpoint = lambda name, path: marker
            server._try_load_model()
            assert server._pipeline is marker
            assert server._active_model_name == "truebones_attnpool"
            assert server._active_model_dataset == "truebones"
        finally:
            server._load_model_checkpoint = old_loader
            server._pipeline = old_pipeline
            server._active_model_name = old_name
            server._active_model_dataset = old_dataset

        os.unlink(preferred_checkpoint)
        assert server._startup_model() == ("", "")

        server._MODEL_PATH = fallback_checkpoint
        assert server._startup_model() == ("other_model", fallback_checkpoint)

    server._MODEL_SAVE_ROOT = old_root
    server._MODEL_PATH = old_path

    runtime_catalog = {}
    server._merge_conditioning_catalogs(runtime_catalog)
    assert "Tyranno" in runtime_catalog
    assert "Michelle" in runtime_catalog

    old_loader = server._load_model_checkpoint
    old_pipeline = server._pipeline
    old_name = server._active_model_name
    old_dataset = server._active_model_dataset
    marker = object()
    try:
        models = {"truebones_attnpool": "/tmp/model.pt"}
        expected = "truebones_attnpool"
        server._pipeline = None
        server._active_model_name = ""
        server._load_model_checkpoint = lambda name, path: marker
        old_available = server._available_models
        server._available_models = lambda: models
        result = asyncio.run(server.select_model(server.ModelSelection(name=expected)))
        assert result["active_model"] == expected
        assert result["active_model_dataset"] == "truebones"
        assert server._pipeline is marker

        server._active_model_dataset = "truebones"
        warnings = server._distribution_warnings({
            "blend_mode": "SINGLE_RETARGET",
            "destination_object_type": "Michelle",
            "destination_dataset": "mixamo",
            "reference": {
                "object_type": "Tyranno",
                "skeleton_dataset": "truebones",
            },
        })
        assert len(warnings) == 1
        assert warnings[0] == {
            "role": "Reference",
            "skeleton": "Michelle",
            "dataset": "mixamo",
            "model_dataset": "truebones",
            "kind": "out_of_distribution",
        }
    finally:
        server._available_models = old_available
        server._load_model_checkpoint = old_loader
        server._pipeline = old_pipeline
        server._active_model_name = old_name
        server._active_model_dataset = old_dataset

    print("Server model catalog smoke test passed")


if __name__ == "__main__":
    main()
