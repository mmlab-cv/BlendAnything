"""Background server communication and shared progress state."""

import json
import os
import tempfile
import threading
import time

import bpy


_state_lock = threading.Lock()
_state = {
    "status": "idle",
    "result_path": None,
    "result_profile_id": "",
    "generated_path": "",
    "error_msg": "",
    "warnings": [],
    "phase": "",
    "started_at": 0.0,
    "progress": 0.0,
}
_model_items = [("NONE", "No models found", "Refresh models from the server")]
_active_model = ""


def set_state(**kwargs) -> None:
    with _state_lock:
        _state.update(kwargs)


def read_state() -> dict:
    with _state_lock:
        return dict(_state)


def redraw_nla_editors() -> None:
    """Request a redraw so background progress remains visibly animated."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "NLA_EDITOR":
                area.tag_redraw()


def model_enum_items() -> list:
    return _model_items


def active_model() -> str:
    return _active_model


def fetch_models(server_url: str) -> dict:
    """Refresh the server model catalog and return its response."""
    global _model_items, _active_model
    import requests

    response = requests.get(server_url.rstrip("/") + "/models", timeout=10)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("models", [])
    _model_items = (
        [(name, name, f"Load model from save/{name}") for name in models]
        or [("NONE", "No models found", "No valid model folders found on the server")]
    )
    _active_model = payload.get("active_model") or ""
    return payload


def load_model(server_url: str, model_name: str) -> dict:
    """Request an atomic server-side model switch."""
    global _active_model
    import requests

    response = requests.post(
        server_url.rstrip("/") + "/models/load",
        json={"name": model_name},
        timeout=600,
    )
    response.raise_for_status()
    payload = response.json()
    _active_model = payload.get("active_model") or model_name
    return payload


def send_request(
    server_url: str,
    ref_bvh: str,
    target_bvhs: list,
    metadata: dict,
) -> None:
    """Submit a progress-aware blend job and store its result in a temp file."""
    target_handles = []
    try:
        import requests

        set_state(phase="Uploading inputs", progress=0.02)
        with open(ref_bvh, "rb") as f_ref:
            files = [("reference_bvh", ("reference.bvh", f_ref, "text/plain"))]
            for i, path in enumerate(target_bvhs):
                fh = open(path, "rb")
                target_handles.append(fh)
                files.append(("target_bvh", (f"target_{i}.bvh", fh, "text/plain")))

            base_url = server_url.rstrip("/")
            response = requests.post(
                base_url + "/blend/jobs",
                files=files,
                data={"metadata": json.dumps(metadata)},
                timeout=120,
            )
            if response.status_code == 404:
                f_ref.seek(0)
                for fh in target_handles:
                    fh.seek(0)
                set_state(phase="Processing on server (legacy mode)", progress=0.10)
                response = requests.post(
                    base_url + "/blend",
                    files=files,
                    data={"metadata": json.dumps(metadata)},
                    timeout=600,
                )
                response.raise_for_status()
                result_content = response.content
            else:
                response.raise_for_status()
                job_id = response.json()["job_id"]
                deadline = time.monotonic() + 600.0
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server job exceeded the 10 minute timeout.")
                    status_response = requests.get(
                        base_url + f"/blend/jobs/{job_id}",
                        timeout=10,
                    )
                    status_response.raise_for_status()
                    job = status_response.json()
                    set_state(
                        phase=job.get("phase") or "Processing on server",
                        progress=float(job.get("progress", 0.0)),
                        warnings=list(job.get("warnings") or []),
                    )
                    if job.get("status") == "done":
                        break
                    if job.get("status") == "error":
                        raise RuntimeError(job.get("error_msg") or "Server job failed.")
                    time.sleep(1.0)

                set_state(phase="Downloading result", progress=0.98)
                result_response = requests.get(
                    base_url + f"/blend/jobs/{job_id}/result",
                    timeout=120,
                )
                result_response.raise_for_status()
                result_content = result_response.content

        set_state(phase="Saving server response", progress=0.99)
        with tempfile.NamedTemporaryFile(
            suffix=".bvh",
            delete=False,
            prefix="nla_result_",
        ) as result_tmp:
            result_tmp.write(result_content)
            result_path = result_tmp.name

        set_state(
            status="done",
            result_path=result_path,
            error_msg="",
            warnings=read_state().get("warnings", []),
            phase="Importing result",
            progress=1.0,
        )
    except Exception as exc:  # noqa: BLE001
        set_state(
            status="error",
            result_path=None,
            error_msg=str(exc),
            phase="",
            progress=0.0,
        )
    finally:
        for fh in target_handles:
            try:
                fh.close()
            except OSError:
                pass
        for path in [ref_bvh] + list(target_bvhs):
            try:
                os.unlink(path)
            except OSError:
                pass
