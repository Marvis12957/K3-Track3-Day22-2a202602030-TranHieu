from preference_lab.data import load_jsonl, split_by_prompt


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected


def test_split_returns_all_examples() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5)
    assert len(train) + len(val) == len(examples)


def test_split_no_prompt_leakage() -> None:
    """Ensure no prompt appears in both train and val splits."""
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.3)
    train_prompts = {ex.prompt for ex in train}
    val_prompts = {ex.prompt for ex in val}
    assert train_prompts.isdisjoint(val_prompts), "Prompt leakage detected between train and val"
