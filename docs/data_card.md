# Data Card

- **Dataset name**: Sample Preference Pairs for ML Education
- **Source**: Manually curated for the Preference Alignment Lab (K3-Track3-Day22)
- **License/permission**: Internal lab use only — not for redistribution
- **Schema**: JSONL format with fields: `prompt` (str, min 1 char), `chosen` (str, min 1 char), `rejected` (str, min 1 char), `metadata` (dict with `domain` and `rubric` keys). Validated via Pydantic `PreferenceExample` model.
- **Labeling rubric**: `accuracy` — chosen responses are factually correct explanations of ML concepts; rejected responses contain subtle errors, oversimplifications, or factual inaccuracies.
- **Known biases**:
  - All 24 examples are in the `education` domain with `accuracy` rubric — no diversity in domain or rubric type.
  - Chosen responses are consistently longer than rejected responses, which means a simple length-based scorer can achieve 100% accuracy. Real-world preference data has more nuance.
  - All prompts are in English; no multilingual coverage.
  - No adversarial or safety-critical examples included.
- **Safety/PII checks**:
  - No PII (personally identifiable information) detected in the dataset.
  - No toxic, harmful, or offensive content present.
  - Dataset does not include medical, legal, or financial advice prompts.
  - Regression prompts (medical advice, uncertainty, missing context) are documented separately in `docs/regression_prompts.md` but not represented in the training data.
- **Train/validation/test split method**: Split by unique prompt using deterministic shuffle (seed=42). All examples sharing the same prompt are kept in the same split to prevent data leakage. Default ratio: 80% train / 20% validation. Verified via `test_split_no_prompt_leakage` with `set.isdisjoint()`.
