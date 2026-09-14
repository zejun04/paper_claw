from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .models import ExistingRecord


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "last_success_at": None, "processed": {}}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"状态文件必须是 JSON 对象: {path}")
    value.setdefault("schema_version", 1)
    value.setdefault("last_success_at", None)
    value.setdefault("processed", {})
    return value


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def discover_existing(root: Path, directories: list[str]) -> dict[str, ExistingRecord]:
    records: dict[str, ExistingRecord] = {}
    pattern = re.compile(r"^arxiv_id:\s*[\"']?([^\"'\r\n]+)", re.MULTILINE)
    status_pattern = re.compile(r"^analysis_status:\s*[\"']?([^\"'\r\n]+)", re.MULTILINE)
    for directory in directories:
        base = root / directory
        if not base.exists():
            continue
        for path in base.rglob("*.md"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            id_match = pattern.search(text)
            if not id_match:
                continue
            status_match = status_pattern.search(text)
            status = status_match.group(1).strip() if status_match else "complete"
            records.setdefault(id_match.group(1).strip(), ExistingRecord(str(path), status))
    return records


def mark_success(state: dict, processed: dict[str, dict]) -> None:
    state["last_success_at"] = datetime.now(timezone.utc).isoformat()
    state.setdefault("processed", {}).update(processed)
