"""Client-safe skeleton catalogs and BVH skeleton matching."""

import json
import os
import re
from typing import Tuple


_PROFILE_CACHE = None
_ENUM_ITEMS_CACHE = None


def _data_path(filename: str) -> str:
    package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(package_dir, "data", filename)


def load_profiles() -> dict:
    global _PROFILE_CACHE
    if _PROFILE_CACHE is None:
        _PROFILE_CACHE = {}
        for dataset, filename in (
            ("truebones", "truebones_skeletons.json"),
            ("mixamo", "mixamo_skeletons.json"),
        ):
            path = _data_path(filename)
            try:
                with open(path, "r") as handle:
                    profiles = json.load(handle).get("profiles", {})
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[Neural NLA] Could not load skeleton profiles from {path}: {exc}")
                continue
            for character, profile in profiles.items():
                profile = dict(profile)
                profile["dataset"] = dataset
                profile["character"] = character
                _PROFILE_CACHE[f"{dataset}::{character}"] = profile
    return _PROFILE_CACHE


def qualify_profile_id(profile_id: str) -> str:
    """Migrate legacy unqualified Truebones profile identifiers."""
    if not profile_id or profile_id == "NONE" or "::" in profile_id:
        return profile_id
    candidate = f"truebones::{profile_id}"
    return candidate if candidate in load_profiles() else profile_id


def profile_parts(profile_id: str) -> Tuple[str, str]:
    qualified = qualify_profile_id(profile_id)
    if "::" not in qualified:
        return "truebones", qualified
    return tuple(qualified.split("::", 1))


def display_name(profile_id: str) -> str:
    """Return the character-facing label for a qualified profile identifier."""
    if profile_id in {"", "NONE"}:
        return "Auto / Unmatched"
    if profile_id == "CUSTOM":
        return "Custom / User Skeleton"
    return profile_parts(profile_id)[1]


def enum_items() -> list:
    """Return stable enum tuples for Blender's dynamic enum UI."""
    global _ENUM_ITEMS_CACHE
    if _ENUM_ITEMS_CACHE is None:
        utility_items = [
            ("NONE", "Auto / Unmatched", "No model skeleton selected"),
            (
                "CUSTOM",
                "Custom / User Skeleton",
                "Derive approximate conditioning from this strip's source BVH",
            ),
        ]
        profiles = load_profiles()
        grouped_items = []
        for dataset, title in (("truebones", "Truebones"), ("mixamo", "Mixamo")):
            grouped_items.append(("", title, f"{title} skeletons"))
            grouped_items.extend(
                (
                    key,
                    profile["character"],
                    f"Use the {profile['character']} skeleton from {title}",
                )
                for key, profile in sorted(
                    profiles.items(),
                    key=lambda item: item[1]["character"].lower(),
                )
                if profile["dataset"] == dataset
            )
        _ENUM_ITEMS_CACHE = utility_items + grouped_items
    return _ENUM_ITEMS_CACHE


def _source_bvh_hierarchy(path: str) -> Tuple[list, dict]:
    names = []
    parents = {}
    stack = []
    in_end_site = False
    with open(path, "r") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "MOTION":
                break
            match = re.match(r"(?:ROOT|JOINT)\s+(\S+)", line)
            if match:
                name = match.group(1)
                parents[name] = stack[-1] if stack else None
                names.append(name)
                stack.append(name)
                in_end_site = False
                continue
            if line.startswith("End Site"):
                in_end_site = True
                match = re.search(r"#name:\s*(\S+)", line)
                if match:
                    name = match.group(1)
                    parents[name] = stack[-1] if stack else None
                    names.append(name)
                continue
            if line == "}":
                if in_end_site:
                    in_end_site = False
                elif stack:
                    stack.pop()
    return names, parents


def _profile_parent_names(profile: dict) -> dict:
    names = profile.get("joints_names", [])
    parents = profile.get("parents", [])
    return {
        name: (names[int(parents[i])] if int(parents[i]) >= 0 else None)
        for i, name in enumerate(names)
    }


def match_source(path: str) -> Tuple[str, float, str]:
    """Rank known profiles by joint-name and parent-edge agreement."""
    try:
        source_names, source_parents = _source_bvh_hierarchy(path)
    except OSError as exc:
        return "", 0.0, str(exc)
    source_set = set(source_names)
    if not source_set:
        return "", 0.0, "source BVH has no named hierarchy"

    ranked = []
    filename = os.path.basename(path).lower()
    source_index = {name: i for i, name in enumerate(source_names)}
    source_parent_indices = [
        source_index.get(source_parents.get(name), -1) for name in source_names
    ]
    for profile_id, profile in load_profiles().items():
        profile_names = profile.get("joints_names", [])
        profile_set = set(profile_names)
        common = source_set & profile_set
        profile_parent_indices = [int(value) for value in profile.get("parents", [])]
        topology_exact = (
            len(source_parent_indices) == len(profile_parent_indices)
            and source_parent_indices == profile_parent_indices
        )
        if not common and not topology_exact:
            continue
        name_recall = len(common) / max(len(profile_set), 1)
        name_precision = len(common) / max(len(source_set), 1)
        profile_parents = _profile_parent_names(profile)
        comparable = [
            name for name in common
            if source_parents.get(name) in common or source_parents.get(name) is None
        ]
        edge_score = (
            sum(
                source_parents.get(name) == profile_parents.get(name)
                for name in comparable
            )
            / max(len(comparable), 1)
        )
        if topology_exact:
            score = 0.75 + 0.15 * name_recall + 0.10 * name_precision
        else:
            score = 0.45 * name_recall + 0.25 * name_precision + 0.30 * edge_score
        character = profile.get("character", "")
        filename_hint = bool(
            character
            and re.search(
                rf"(^|[^a-z0-9]){re.escape(character.lower())}([^a-z0-9]|$)",
                filename,
            )
        )
        ranked.append((score, filename_hint, profile_id))

    if not ranked:
        return "", 0.0, "no profile shares joint names with this BVH"
    ranked.sort(reverse=True)
    hinted = [candidate for candidate in ranked if candidate[1]]
    if hinted:
        hinted.sort(reverse=True)
        best_score, _, best_id = hinted[0]
        if best_score >= 0.74:
            return best_id, best_score, ""

    best_score, _, best_id = ranked[0]
    next_score = ranked[1][0] if len(ranked) > 1 else 0.0
    if best_score < 0.74:
        return "", best_score, f"best match {best_id} is too weak"
    if best_score - next_score < 0.05:
        return "", best_score, f"ambiguous between {best_id} and {ranked[1][2]}"
    return best_id, best_score, ""
