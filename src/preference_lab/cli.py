from __future__ import annotations

from pathlib import Path

import typer  # pyrefly: ignore [missing-import]
from rich import print  # pyrefly: ignore [missing-import]

from .config import load_config
from .data import load_jsonl
from .evaluate import pairwise_accuracy, write_metrics

app = typer.Typer(help="Preference alignment lab CLI")


def _deterministic_score(text: str) -> float:
    """Compute a simple deterministic score based on text properties.

    Uses a combination of response length, vocabulary richness, and
    sentence structure to produce a quality proxy score in [0, 1].
    """
    if not text.strip():
        return 0.0
    words = text.split()
    num_words = len(words)
    unique_ratio = len({w.lower() for w in words}) / max(num_words, 1)
    length_score = min(num_words / 50.0, 1.0)  # longer (up to 50 words) is better
    return 0.5 * length_score + 0.5 * unique_ratio


@app.command()
def validate(data: Path) -> None:
    examples = load_jsonl(data)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")


@app.command()
def evaluate(config: Path) -> None:
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"]["train_data"])
    # Score responses using a deterministic text-quality scorer
    chosen_scores = [_deterministic_score(ex.chosen) for ex in examples]
    rejected_scores = [_deterministic_score(ex.rejected) for ex in examples]
    metrics = {"pairwise_accuracy": pairwise_accuracy(examples, chosen_scores, rejected_scores)}
    out = write_metrics(metrics, cfg["paths"]["output_dir"])
    print(f"[green]Wrote metrics to {out}[/green]")


if __name__ == "__main__":
    app()
