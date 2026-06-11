"""Stage 2 — execute candidate plot scripts in the disposable Docker sandbox.

Candidate code is untrusted; it never runs on the host. Each script gets a
fresh container with no network, capped CPU/memory/pids, a read-only root
filesystem, and a hard wall-clock timeout enforced from the host.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

from . import artifacts
from .config import Config

_STD_TRUNC = 100_000
_START_OVERHEAD_S = 15


def docker_available() -> tuple[bool, str]:
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return proc.returncode == 0, proc.stderr.strip()


def image_exists(image: str) -> bool:
    proc = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True)
    return proc.returncode == 0


def run(cfg: Config, only_instance: str | None = None) -> None:
    ok, err = docker_available()
    if not ok:
        raise SystemExit(f"Docker daemon is not running. Start Docker Desktop, then retry.\n{err}")
    if not image_exists(cfg.sandbox.image):
        raise SystemExit(f"Image {cfg.sandbox.image!r} not found. Run `make sandbox-build` first.")
    store = artifacts.Store(cfg.artifacts_dir)
    executed = skipped = 0
    for iid in store.instance_ids():
        if only_instance and iid != only_instance:
            continue
        if store.exec_path(iid).exists():
            skipped += 1
            continue
        if store.load_response(iid) is None:
            continue
        result = execute_instance(cfg, store, iid)
        store.write_json(store.exec_path(iid), result)
        executed += 1
        status = "ok" if result["ran_ok"] else ("timeout" if result["timed_out"] else "failed")
        print(f"  {iid}: {status} ({result['n_figures']} figures)")
    print(f"sandbox: {executed} executed, {skipped} already present")


def execute_instance(cfg: Config, store: artifacts.Store, iid: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "instance_id": iid,
        "ran_ok": False,
        "exit_code": None,
        "timed_out": False,
        "n_figures": 0,
        "stdout": "",
        "stderr": "",
        "figure_paths": [],
        "duration_s": None,
        "manifest": None,
        "detail": None,
    }
    script = store.plot_script_path(iid)
    if not script.exists():
        result["detail"] = (
            "no plot_script extracted (response unparseable, or field missing / not a string)"
        )
        return result

    figdir = store.figures_dir(iid)
    figdir.mkdir(parents=True, exist_ok=True)
    name = f"fb-{iid}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)  # stale-name insurance
    argv = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network=none",
        f"--memory={cfg.sandbox.memory}",
        f"--memory-swap={cfg.sandbox.memory}",
        f"--cpus={cfg.sandbox.cpus}",
        f"--pids-limit={cfg.sandbox.pids}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=256m",
        "-v",
        f"{script.resolve()}:/in/plot_script.py:ro",
        "-v",
        f"{figdir.resolve()}:/out",
        cfg.sandbox.image,
        "python",
        "/runner.py",
        "/in/plot_script.py",
        "/out",
    ]
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=cfg.sandbox.timeout_seconds + _START_OVERHEAD_S,
        )
        result["exit_code"] = proc.returncode
        result["stdout"] = proc.stdout[-_STD_TRUNC:]
        result["stderr"] = proc.stderr[-_STD_TRUNC:]
    except subprocess.TimeoutExpired as exc:
        result["timed_out"] = True
        result["stdout"] = (exc.stdout or "")[-_STD_TRUNC:] if exc.stdout else ""
        result["stderr"] = (exc.stderr or "")[-_STD_TRUNC:] if exc.stderr else ""
        result["detail"] = f"killed after {cfg.sandbox.timeout_seconds + _START_OVERHEAD_S}s wall"
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    result["duration_s"] = round(time.monotonic() - started, 2)

    figures = sorted(p.name for p in figdir.glob("*.png"))
    result["figure_paths"] = [f"figures/{n}" for n in figures]
    result["n_figures"] = len(figures)
    result["manifest"] = store.read_json(figdir / "manifest.json")
    result["ran_ok"] = (not result["timed_out"]) and result["exit_code"] == 0
    return result
