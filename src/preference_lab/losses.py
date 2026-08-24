from __future__ import annotations

import numpy as np


def dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float,
) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    L_DPO = -log σ(β · [(log π(y_w|x) - log π(y_l|x)) - (log π_ref(y_w|x) - log π_ref(y_l|x))])

    Uses numerically stable log-sigmoid: log σ(x) = -softplus(-x) = -log(1 + exp(-x)).
    """
    policy_ratio = policy_chosen_logps - policy_rejected_logps
    ref_ratio = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_ratio - ref_ratio)
    # -log_sigmoid(logits) = log(1 + exp(-logits)) = softplus(-logits)
    losses = np.logaddexp(0.0, -logits)
    return float(np.mean(losses))


def orpo_loss(
    sft_nll: np.ndarray,
    chosen_logps: np.ndarray,
    rejected_logps: np.ndarray,
    lambda_orpo: float,
) -> float:
    """Compute a simplified ORPO-style objective.

    L_ORPO = L_SFT + λ · L_OR
    where L_OR = -log σ(log(odds(y_w) / odds(y_l)))
    and odds(y) = P(y) / (1 - P(y)), P(y) = exp(logp).

    A small epsilon is added for numerical stability.
    """
    eps = 1e-10
    chosen_probs = np.exp(chosen_logps)
    rejected_probs = np.exp(rejected_logps)
    # odds = p / (1 - p)
    chosen_odds = chosen_probs / (1.0 - chosen_probs + eps)
    rejected_odds = rejected_probs / (1.0 - rejected_probs + eps)
    # log odds ratio
    log_or = np.log(chosen_odds + eps) - np.log(rejected_odds + eps)
    # -log_sigmoid(log_or) = softplus(-log_or)
    or_loss = np.logaddexp(0.0, -log_or)
    return float(np.mean(sft_nll) + lambda_orpo * np.mean(or_loss))
