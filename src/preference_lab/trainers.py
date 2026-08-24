from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .losses import dpo_loss, orpo_loss

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2


class PreferenceTrainer:
    """Interface for DPO/ORPO training implementations."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self) -> dict[str, Any]:
        """Train the policy using a mock CPU trainer.

        Simulates one epoch of training by generating random log-probabilities
        and computing the configured loss. Returns final metrics.
        """
        rng = np.random.default_rng(seed=42)
        num_batches = 5
        losses: list[float] = []

        for batch_idx in range(num_batches):
            # Simulate log-probabilities (log probs are <= 0)
            policy_chosen_logps = rng.uniform(-1.0, -0.1, size=self.config.batch_size)
            policy_rejected_logps = rng.uniform(-2.0, -0.5, size=self.config.batch_size)

            if self.config.method == "dpo":
                ref_chosen_logps = rng.uniform(-1.5, -0.2, size=self.config.batch_size)
                ref_rejected_logps = rng.uniform(-2.0, -0.5, size=self.config.batch_size)
                loss = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=self.config.beta,
                )
            elif self.config.method == "orpo":
                sft_nll = rng.uniform(0.5, 2.0, size=self.config.batch_size)
                loss = orpo_loss(
                    sft_nll,
                    policy_chosen_logps,
                    policy_rejected_logps,
                    lambda_orpo=self.config.lambda_orpo,
                )
            else:
                # Mock method: return a dummy decreasing loss
                loss = 1.0 / (batch_idx + 1)

            losses.append(loss)
            logger.info(
                "Batch %d/%d — %s loss: %.4f", batch_idx + 1, num_batches, self.config.method, loss
            )

        metrics = {
            "final_loss": losses[-1],
            "mean_loss": float(np.mean(losses)),
            "method": self.config.method,
        }
        logger.info("Training complete. Mean loss: %.4f", metrics["mean_loss"])
        return metrics
