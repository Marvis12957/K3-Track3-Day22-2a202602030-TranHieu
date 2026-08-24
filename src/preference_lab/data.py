from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from .schemas import PreferenceExample

logger = logging.getLogger(__name__)


def load_jsonl(path: str | Path) -> list[PreferenceExample]:
    """Load preference examples from JSONL.

    Features:
    - Line-numbered error messages for invalid JSON or validation failures.
    - Duplicate prompt detection with warnings.
    """
    examples: list[PreferenceExample] = []
    seen_prompts: dict[str, int] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num}: invalid JSON — {e}") from e
            try:
                ex = PreferenceExample.model_validate(obj)
            except Exception as e:
                raise ValueError(f"Line {line_num}: validation error — {e}") from e
            prompt_key = ex.prompt.strip().lower()
            if prompt_key in seen_prompts:
                logger.warning(
                    "Line %d: duplicate prompt (first seen line %d)",
                    line_num,
                    seen_prompts[prompt_key],
                )
            else:
                seen_prompts[prompt_key] = line_num
            examples.append(ex)
    return examples


def split_by_prompt(
    examples: list[PreferenceExample], validation_ratio: float = 0.2
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid leakage.

    Groups all examples sharing the same prompt together, then assigns
    entire groups to either train or validation using deterministic
    shuffling (seed=42).
    """
    groups: defaultdict[str, list[PreferenceExample]] = defaultdict(list)
    for ex in examples:
        groups[ex.prompt].append(ex)
    prompts = sorted(groups.keys())
    rng = random.Random(42)
    rng.shuffle(prompts)
    val_count = max(1, int(len(prompts) * validation_ratio))
    val_prompts = set(prompts[:val_count])
    train = [ex for p in prompts if p not in val_prompts for ex in groups[p]]
    val = [ex for p in val_prompts for ex in groups[p]]
    return train, val
