"""Artifact store: path layout and atomic JSON I/O shared by all stages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def instance_id(model: str, sample_idx: int) -> str:
    return f"{model}__s{sample_idx}"


def model_from_instance_id(iid: str) -> str:
    return iid.rsplit("__s", 1)[0]


def _json_default(obj: Any) -> Any:
    # numpy scalars sneak in from scipy/polars computations
    for attr in ("item",):
        if hasattr(obj, attr):
            return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj)}")


class Store:
    """All pipeline reads/writes go through this path layout."""

    def __init__(self, root: Path):
        self.root = Path(root)

    # --- layout -----------------------------------------------------------
    @property
    def instances_dir(self) -> Path:
        return self.root / "instances"

    @property
    def tables_dir(self) -> Path:
        return self.root / "tables"

    @property
    def run_meta_path(self) -> Path:
        return self.root / "run.json"

    def instance_dir(self, iid: str) -> Path:
        return self.instances_dir / iid

    def response_path(self, iid: str) -> Path:
        return self.instance_dir(iid) / "response.json"

    def generation_error_path(self, iid: str) -> Path:
        return self.instance_dir(iid) / "generation_error.json"

    def checks_path(self, iid: str) -> Path:
        return self.instance_dir(iid) / "checks.json"

    # --- pairwise stage ----------------------------------------------------
    @property
    def pairwise_dir(self) -> Path:
        return self.root / "pairwise"

    @property
    def pairwise_manifest_path(self) -> Path:
        return self.pairwise_dir / "manifest.json"

    @property
    def pairwise_batches_path(self) -> Path:
        return self.pairwise_dir / "batches.json"

    @property
    def pairwise_attempts_path(self) -> Path:
        return self.pairwise_dir / "attempts.json"

    @property
    def pairwise_raw_dir(self) -> Path:
        return self.pairwise_dir / "raw"

    def pairwise_verdict_path(self, judge: str, custom_id: str) -> Path:
        return self.pairwise_dir / "verdicts" / judge / f"{custom_id}.json"

    def load_pairwise_verdict(self, judge: str, custom_id: str) -> dict[str, Any] | None:
        return self.read_json(self.pairwise_verdict_path(judge, custom_id))

    def table_path(self, name: str) -> Path:
        return self.tables_dir / f"{name}.json"

    def instance_ids(self) -> list[str]:
        if not self.instances_dir.exists():
            return []
        return sorted(p.name for p in self.instances_dir.iterdir() if p.is_dir())

    # --- I/O ----------------------------------------------------------------
    def write_json(self, path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
        tmp.write_text(text + "\n")
        os.replace(tmp, path)

    def read_json(self, path: Path) -> Any | None:
        if not path.exists():
            return None
        return json.loads(path.read_text())

    # --- typed convenience loaders -------------------------------------------
    def load_response(self, iid: str) -> dict[str, Any] | None:
        return self.read_json(self.response_path(iid))

    def load_checks(self, iid: str) -> dict[str, Any] | None:
        return self.read_json(self.checks_path(iid))
