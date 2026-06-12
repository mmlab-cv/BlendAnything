"""Verify combined skeleton catalogs and filename-assisted BVH matching."""

import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blendanything_client.profiles import enum_items, load_profiles, match_source


def main() -> None:
    profiles = load_profiles()
    assert sum(key.startswith("truebones::") for key in profiles) == 70
    assert sum(key.startswith("mixamo::") for key in profiles) == 53

    items = {item[0]: item[1] for item in enum_items()}
    assert items["truebones::Coyote"] == "Coyote"
    assert items["mixamo::Abe"] == "Abe"
    ordered = enum_items()
    truebones_heading = ordered.index(("", "Truebones", "Truebones skeletons"))
    mixamo_heading = ordered.index(("", "Mixamo", "Mixamo skeletons"))
    assert truebones_heading < mixamo_heading
    assert all(
        item[0].startswith("truebones::")
        for item in ordered[truebones_heading + 1:mixamo_heading]
    )

    coyote = os.path.join(
        REPO_ROOT,
        "neural_motion_blending",
        "dataset",
        "truebones",
        "zoo",
        "truebones_processed",
        "bvhs",
        "Coyote___Howling_226.bvh",
    )
    profile_id, score, error = match_source(coyote)
    assert (profile_id, score, error) == ("truebones::Coyote", 1.0, "")
    print("Combined skeleton profile smoke test passed")


if __name__ == "__main__":
    main()
