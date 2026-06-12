"""Headless Blender smoke test for add-on registration and in-place upgrades."""

import os
import shutil
import sys
import tempfile

import bpy


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class NeuralNLAPreferences(bpy.types.AddonPreferences):
    bl_idname = "blendanything_client.addon"


class NEURAL_NLA_PT_Panel(bpy.types.Panel):
    bl_idname = "NEURAL_NLA_PT_panel"
    bl_label = "Stale Neural NLA Panel"
    bl_space_type = "NLA_EDITOR"
    bl_region_type = "UI"

    def draw(self, context):
        context.preferences.addons["blendanything_client.addon"]


def main() -> None:
    bpy.utils.register_class(NeuralNLAPreferences)
    bpy.utils.register_class(NEURAL_NLA_PT_Panel)

    from blendanything_client import addon

    addon.register()
    registered_panel = getattr(bpy.types, addon.NEURAL_NLA_PT_Panel.bl_idname)
    assert registered_panel is addon.NEURAL_NLA_PT_Panel
    assert not hasattr(bpy.types, "NeuralNLAPreferences")
    assert addon.bl_info["name"] == "BlendAnything"
    assert addon.bl_info["author"] == "Luca Cazzola"
    assert addon.bl_info["version"] == (1, 0, 0)
    assert addon.bl_info["blender"] == (4, 5, 4)
    assert addon._ADDON_VERSION == "1.0.0"
    assert addon.bl_info["doc_url"] == "https://mmlab-cv.github.io/BlendAnything/"
    assert "SIGGRAPH Posters 2026" in addon.bl_info["description"]
    assert hasattr(bpy.types.Scene, "neural_nla_server_url")
    assert hasattr(bpy.types.Scene, "neural_nla_server_model")
    assert hasattr(bpy.types.Scene, "neural_nla_show_server")
    assert hasattr(bpy.types.Scene, "neural_nla_strength_mode")
    assert hasattr(bpy.types.Scene, "neural_nla_crossfade_shape")
    assert hasattr(bpy.types.Scene, "neural_nla_ddim_inversion_policy")
    assert all(
        cls.__name__ != "NEURAL_NLA_OT_AutoDetectSources"
        for cls in addon._CLASSES
    )
    assert bpy.context.scene.neural_nla_ddim_inversion_policy == "SAME_SKELETON"
    assert bpy.ops.neural_nla.apply_linked_crossfade.poll()
    assert addon._server_url(bpy.context) == "http://localhost:8000"

    obj = bpy.data.objects.new("CrossfadeTest", None)
    bpy.context.scene.collection.objects.link(obj)
    animation_data = obj.animation_data_create()
    track_a = animation_data.nla_tracks.new()
    track_b = animation_data.nla_tracks.new()
    action_a = bpy.data.actions.new("CrossfadeA")
    action_b = bpy.data.actions.new("CrossfadeB")
    strip_a = track_a.strips.new("CrossfadeA", 0, action_a)
    strip_b = track_b.strips.new("CrossfadeB", 10, action_b)
    strip_a.frame_end = 20
    strip_b.frame_end = 30
    strip_a.select = True
    strip_b.select = True
    bpy.context.scene.neural_nla_blend_mode = "RETARGET"
    assert not addon.NEURAL_NLA_PT_StripStrength.poll(bpy.context)
    bpy.context.scene.neural_nla_blend_mode = "BLEND"
    bpy.context.scene.neural_nla_crossfade_shape = "SMOOTH"
    result = bpy.ops.neural_nla.apply_linked_crossfade()
    assert result == {"FINISHED"}
    strength_a = addon._get_strength(bpy.context.scene, strip_a.name)
    strength_b = addon._get_strength(bpy.context.scene, strip_b.name)
    assert strength_a.profile == strength_b.profile == "SMOOTH"
    assert strength_a.fade_in == 0.0 and strength_a.fade_out > 0.0
    assert strength_b.fade_in > 0.0 and strength_b.fade_out == 0.0
    linked_result = addon._resulting_relative_strength(
        strip_a,
        strip_b,
        bpy.context.scene,
        sample_count=32,
    )
    assert linked_result[0] < 0.05
    assert linked_result[-1] > 0.95
    assert all(
        earlier <= later + 1e-6
        for earlier, later in zip(linked_result, linked_result[1:])
    )
    strength_a.profile = "CONSTANT"
    strength_a.value = 0.75
    strength_b.profile = "CONSTANT"
    strength_b.value = 0.25
    resulting = addon._resulting_relative_strength(
        strip_a,
        strip_b,
        bpy.context.scene,
        sample_count=8,
    )
    assert len(resulting) == 8
    assert all(abs(value - 0.25) < 1e-6 for value in resulting)
    image = addon._strength_plot_image(resulting)
    assert tuple(image.size) == (320, 320)
    first_image = image
    strength_a.value = 0.25
    strength_b.value = 0.75
    updated = addon._resulting_relative_strength(
        strip_a,
        strip_b,
        bpy.context.scene,
        sample_count=8,
    )
    assert all(abs(value - 0.75) < 1e-6 for value in updated)
    updated_image = addon._strength_plot_image(updated)
    assert updated_image != first_image

    skeleton = bpy.context.scene.neural_nla_strip_skeletons.add()
    skeleton.strip_name = strip_a.name
    skeleton["profile_id"] = "Coyote"
    skeleton.face_right = "custom_face_joint"
    migrated = addon._ensure_strip_skeleton(bpy.context.scene, strip_a)
    assert migrated.profile_id == "truebones::Coyote"
    assert migrated.face_right == "custom_face_joint"
    migrated.profile_id = "CUSTOM"
    migrated.custom_name = "MySkeleton"
    assert migrated.profile_id == "CUSTOM"
    assert migrated.custom_name == "MySkeleton"
    assert all(
        item[0] != "CUSTOM"
        for item in addon._destination_profile_items(None, bpy.context)
    )

    addon._set_state(status="running", warnings=[{"skeleton": "Michelle"}])
    assert bpy.ops.neural_nla.dismiss_warnings() == {"FINISHED"}
    status = addon._read_state()
    assert status["status"] == "running"
    assert status["warnings"] == []

    with tempfile.TemporaryDirectory() as generated_dir:
        source_result = os.path.join(REPO_ROOT, ".bvh_cache", "Skunk___Walk_892_blended.bvh")
        temporary_result = os.path.join(generated_dir, "downloaded_result.bvh")
        shutil.copy2(source_result, temporary_result)
        old_generated_dir = os.environ.get("BLENDANYTHING_GENERATED_DIR")
        os.environ["BLENDANYTHING_GENERATED_DIR"] = generated_dir
        try:
            persistent_result = addon.import_bvh_as_result(
                temporary_result,
                "truebones::Skunk",
            )
        finally:
            if old_generated_dir is None:
                os.environ.pop("BLENDANYTHING_GENERATED_DIR", None)
            else:
                os.environ["BLENDANYTHING_GENERATED_DIR"] = old_generated_dir
        assert os.path.isfile(persistent_result)
        generated_objects = [
            obj for obj in bpy.data.objects
            if obj.get("blendanything_generated_bvh") == persistent_result
        ]
        assert len(generated_objects) == 1
        generated = generated_objects[0]
        assert generated.animation_data.action is None
        tracks = list(generated.animation_data.nla_tracks)
        assert len(tracks) == 1
        strips = list(tracks[0].strips)
        assert len(strips) == 1
        strip = strips[0]
        action = strip.action
        assert action[addon._ACTION_SOURCE_PROP] == persistent_result
        assert action[addon._ACTION_PROFILE_PROP] == "truebones::Skunk"
        assert action[addon._ACTION_INPUT_MODE_PROP] == "MODEL_PROCESSED"
        skeleton = addon._get_strip_skeleton(bpy.context.scene, strip.name)
        assert skeleton is not None
        assert skeleton.profile_id == "truebones::Skunk"
        strength = addon._get_strength(bpy.context.scene, strip.name)
        assert strength is not None

    addon.unregister()
    assert addon._STRENGTH_PLOT_IMAGE not in bpy.data.images
    assert not hasattr(bpy.types.Scene, "neural_nla_server_url")
    assert not hasattr(bpy.types.Scene, "neural_nla_server_model")
    assert not hasattr(bpy.types.Scene, "neural_nla_show_server")
    assert not hasattr(bpy.types.Scene, "neural_nla_strength_mode")
    assert not hasattr(bpy.types.Scene, "neural_nla_ddim_inversion_policy")
    print("BlendAnything add-on registration smoke test passed")


if __name__ == "__main__":
    main()
