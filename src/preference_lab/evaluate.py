from __future__ import annotations

import json
from pathlib import Path

from .schemas import PreferenceExample


def pairwise_accuracy(
    examples: list[PreferenceExample],
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> float:
    """Return fraction where chosen score is greater than rejected score.

    Ties (equal scores) count as half a win.
    """
    if not examples:
        return 0.0
    if len(chosen_scores) != len(examples) or len(rejected_scores) != len(examples):
        raise ValueError(
            f"Score length mismatch: {len(chosen_scores)} chosen, "
            f"{len(rejected_scores)} rejected, {len(examples)} examples"
        )
    wins = 0.0
    for c, r in zip(chosen_scores, rejected_scores):
        if c > r:
            wins += 1.0
        elif c == r:
            wins += 0.5  # ties count as half-win
    return wins / len(examples)


def write_metrics(metrics: dict[str, float], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
