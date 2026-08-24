import numpy as np

from preference_lab.losses import dpo_loss, kto_loss, orpo_loss


def test_dpo_loss_positive() -> None:
    """When policy strongly prefers chosen over ref, loss should be low."""
    loss = dpo_loss(
        np.array([-0.5]),  # policy chosen (high)
        np.array([-1.5]),  # policy rejected (low)
        np.array([-0.6]),  # ref chosen
        np.array([-1.0]),  # ref rejected
        beta=0.1,
    )
    assert isinstance(loss, float)
    assert loss > 0.0  # loss is always positive


def test_dpo_loss_increases_when_policy_disagrees() -> None:
    """Loss should be higher when policy prefers rejected over chosen."""
    # Policy agrees with preference (chosen > rejected)
    loss_good = dpo_loss(
        np.array([-0.2]),  # policy chosen (high)
        np.array([-2.0]),  # policy rejected (low)
        np.array([-1.0]),
        np.array([-1.0]),
        beta=0.5,
    )
    # Policy disagrees (chosen < rejected)
    loss_bad = dpo_loss(
        np.array([-2.0]),  # policy chosen (low)
        np.array([-0.2]),  # policy rejected (high)
        np.array([-1.0]),
        np.array([-1.0]),
        beta=0.5,
    )
    assert loss_bad > loss_good


def test_dpo_loss_batch() -> None:
    """DPO loss should work with batch inputs."""
    loss = dpo_loss(
        np.array([-0.5, -0.3]),
        np.array([-1.5, -1.0]),
        np.array([-0.6, -0.4]),
        np.array([-1.0, -0.8]),
        beta=0.1,
    )
    assert isinstance(loss, float)
    assert loss > 0.0


def test_orpo_loss_positive() -> None:
    """ORPO loss should return a positive float."""
    loss = orpo_loss(
        np.array([1.0]),  # sft_nll
        np.array([-0.5]),  # chosen logps
        np.array([-1.5]),  # rejected logps
        lambda_orpo=0.1,
    )
    assert isinstance(loss, float)
    assert loss > 0.0


def test_orpo_loss_prefers_chosen() -> None:
    """ORPO penalty should decrease when chosen logps are higher than rejected."""
    loss_good = orpo_loss(
        np.array([1.0]),
        np.array([-0.2]),  # chosen much higher
        np.array([-2.0]),  # rejected much lower
        lambda_orpo=0.5,
    )
    loss_bad = orpo_loss(
        np.array([1.0]),
        np.array([-2.0]),  # chosen lower
        np.array([-0.2]),  # rejected higher
        lambda_orpo=0.5,
    )
    assert loss_bad > loss_good


def test_kto_loss_positive() -> None:
    """KTO loss should return a positive float."""
    loss = kto_loss(
        policy_chosen_logps=np.array([-0.5]),
        policy_rejected_logps=np.array([-1.5]),
        ref_chosen_logps=np.array([-0.6]),
        ref_rejected_logps=np.array([-1.0]),
        beta=0.1,
    )
    assert isinstance(loss, float)
    assert loss > 0.0


def test_kto_loss_prefers_chosen() -> None:
    """KTO loss should be lower when policy strongly prefers chosen over reference."""
    loss_good = kto_loss(
        policy_chosen_logps=np.array([-0.2]),
        policy_rejected_logps=np.array([-2.0]),
        ref_chosen_logps=np.array([-1.0]),
        ref_rejected_logps=np.array([-1.0]),
        beta=0.5,
    )
    loss_bad = kto_loss(
        policy_chosen_logps=np.array([-2.0]),
        policy_rejected_logps=np.array([-0.2]),
        ref_chosen_logps=np.array([-1.0]),
        ref_rejected_logps=np.array([-1.0]),
        beta=0.5,
    )
    assert loss_bad > loss_good
