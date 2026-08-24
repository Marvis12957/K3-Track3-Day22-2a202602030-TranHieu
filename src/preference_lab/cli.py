from __future__ import annotations

from pathlib import Path
from typing import Annotated

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
def evaluate(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to YAML config file")] = Path(
        "configs/local.yaml"
    ),
) -> None:
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"]["train_data"])
    # Score responses using a deterministic text-quality scorer
    chosen_scores = [_deterministic_score(ex.chosen) for ex in examples]
    rejected_scores = [_deterministic_score(ex.rejected) for ex in examples]
    metrics = {"pairwise_accuracy": pairwise_accuracy(examples, chosen_scores, rejected_scores)}
    out = write_metrics(metrics, cfg["paths"]["output_dir"])
    print(f"[green]Wrote metrics to {out}[/green]")


from .trainers import PreferenceTrainer, TrainingConfig


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to YAML config file")] = Path(
        "configs/local.yaml"
    ),
) -> None:
    cfg = load_config(config)
    train_cfg = cfg.get("training", {})
    training_config = TrainingConfig(
        method=train_cfg.get("method", "dpo"),
        beta=train_cfg.get("beta", 0.1),
        lambda_orpo=train_cfg.get("lambda_orpo", 0.1),
        desirable_weight=train_cfg.get("desirable_weight", 1.0),
        undesirable_weight=train_cfg.get("undesirable_weight", 1.0),
        max_length=train_cfg.get("max_length", 512),
        batch_size=train_cfg.get("batch_size", 2),
    )
    trainer = PreferenceTrainer(training_config)
    print(f"[bold blue]Starting training with method: {training_config.method}...[/bold blue]")
    metrics = trainer.train()
    out_dir = Path(cfg["paths"]["output_dir"])
    out = write_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, out_dir)
    print(
        f"[green]Training finished! Final loss: {metrics['final_loss']:.4f}, Mean loss: {metrics['mean_loss']:.4f}[/green]"
    )
    print(f"[green]Saved metrics to {out}[/green]")


if __name__ == "__main__":
    app()
