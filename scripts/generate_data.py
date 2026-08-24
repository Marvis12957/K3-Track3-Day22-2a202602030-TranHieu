from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from openai import OpenAI
from rich import print

app = typer.Typer(help="Synthetic Data Generation for Preference Alignment")

SYSTEM_PROMPT = """You are an AI data engineer specializing in preference alignment (DPO/ORPO).
Your task is to generate high-quality preference pairs in JSONL format.
Each pair must have:
1. 'prompt': A clear instruction or question.
2. 'chosen': A high-quality, accurate, and helpful response.
3. 'rejected': A plausible but lower-quality response (e.g., contains a subtle error, hallucination, or poor formatting).
4. 'metadata': A dictionary with 'domain' and 'rubric'.

Output ONLY the JSONL lines, one per line. Do not include markdown formatting or extra text."""

USER_PROMPT_TEMPLATE = """Generate {count} new preference pairs about {domain}.
Use the following examples as a style guide:
{examples}

Focus on: {focus}"""


def _load_env_file(env_path: Path = Path(".env")) -> None:
    """Load simple key=value pairs from .env into os.environ if not already set."""
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v


@app.command()
def generate(
    count: int = 5,
    domain: str = "machine learning",
    focus: str = "technical accuracy and safety",
    output_file: Path = Path("data/synthetic_preferences.jsonl"),
    seed_file: Path = Path("data/sample_preferences.jsonl"),
    model: str | None = None,
    mock: bool = False,
) -> None:
    """Generate synthetic preference pairs using Gemini API (or OpenAI / offline mock)."""
    _load_env_file()

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not mock and gemini_key:
        print("[cyan]Using Gemini API endpoint with GEMINI_API_KEY...[/cyan]")
        client = OpenAI(
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        model_name = model or "gemini-2.5-flash"
    elif not mock and openai_key:
        print("[cyan]Using OpenAI API endpoint with OPENAI_API_KEY...[/cyan]")
        client = OpenAI(api_key=openai_key)
        model_name = model or "gpt-4o"
    else:
        if not mock:
            print(
                "[yellow]Notice: GEMINI_API_KEY or OPENAI_API_KEY not found in .env / environment. Running in offline mock generation mode.[/yellow]"
            )
        print(f"Generating [blue]{count}[/blue] mock pairs for domain: [green]{domain}[/green]...")

        mock_templates = [
            (
                f"Explain the role of gradient clipping in {domain}.",
                "Gradient clipping caps gradients during backpropagation to prevent exploding gradients in deep networks.",
                "Gradient clipping sets all negative gradients to zero to speed up training.",
            ),
            (
                f"What is early stopping in {domain}?",
                "Early stopping halts training when validation loss stops improving, preventing the model from overfitting.",
                "Early stopping terminates training after a fixed 10 epochs regardless of performance.",
            ),
            (
                f"Why use learning rate warmup in {domain}?",
                "Learning rate warmup gradually increases the learning rate initially, stabilizing early training steps.",
                "Learning rate warmup makes the model train on the CPU before moving to the GPU.",
            ),
            (
                f"Describe weight decay in {domain} training.",
                "Weight decay adds an L2 penalty to the loss function, encouraging smaller weights and reducing overfitting.",
                "Weight decay removes unused weights from the network architecture permanently.",
            ),
            (
                f"What is the difference between batch size and epoch in {domain}?",
                "Batch size is the number of samples processed per step, while an epoch is one full pass through the dataset.",
                "Batch size and epoch are identical terms describing the total number of training iterations.",
            ),
        ]

        valid_lines = []
        for i in range(count):
            prompt, chosen, rejected = mock_templates[i % len(mock_templates)]
            item = {
                "prompt": f"[{i + 1}] {prompt}",
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {"domain": domain, "rubric": "accuracy", "generator": "synthetic_mock"},
            }
            valid_lines.append(json.dumps(item))

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("a", encoding="utf-8") as f:
            for line in valid_lines:
                f.write(line + "\n")
        print(f"[green]Successfully added {len(valid_lines)} pairs to {output_file}[/green]")
        return

    # Load some examples from seed file
    examples_str = ""
    if seed_file.exists():
        with seed_file.open("r") as f:
            lines = [line.strip() for line in f if line.strip()][:3]
            examples_str = "\n".join(lines)

    print(
        f"Generating [blue]{count}[/blue] pairs for domain: [green]{domain}[/green] using model [blue]{model_name}[/blue]..."
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    count=count, domain=domain, examples=examples_str, focus=focus
                ),
            },
        ],
        temperature=0.7,
    )

    content = response.choices[0].message.content
    if not content:
        print("[red]Error: Received empty response from API.[/red]")
        raise typer.Exit(1)

    # Simple validation and write
    valid_lines = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip markdown code blocks if the model included them
        if line.startswith("```"):
            continue
        try:
            json.loads(line)
            valid_lines.append(line)
        except json.JSONDecodeError:
            print(f"[yellow]Skipping invalid JSON line: {line[:50]}...[/yellow]")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f:
        for line in valid_lines:
            f.write(line + "\n")

    print(f"[green]Successfully added {len(valid_lines)} pairs to {output_file}[/green]")


if __name__ == "__main__":
    app()
