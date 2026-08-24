from __future__ import annotations

from pathlib import Path
from typing import Any, cast

# pyrefly: ignore [missing-source-for-stubs]
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return cast(dict[str, Any], data if isinstance(data, dict) else {})
