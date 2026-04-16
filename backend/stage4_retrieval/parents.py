"""Parent lookup helpers for stage-4 retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_parent_lookup(parents_json_path: Path | None) -> dict[str, dict[str, Any]]:
    """parents.json을 parent_id 기준 lookup dict로 변환한다."""
    if parents_json_path is None or not parents_json_path.exists():
        return {}

    payload = json.loads(parents_json_path.read_text())
    if not isinstance(payload, dict):
        return {}

    parent_lookup: dict[str, dict[str, Any]] = {}
    for parent in payload.get("parents") or []:
        if not isinstance(parent, dict):
            continue
        parent_id = str(parent.get("parent_id") or "").strip()
        if not parent_id:
            continue
        parent_lookup[parent_id] = parent
    return parent_lookup
