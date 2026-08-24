# Lab Guide

## Task 1: Data loader
Implement robust JSONL loading. Add line-numbered error messages and duplicate checks.

## Task 1.5: (Optional) Synthetic Data Generation
Use an LLM (Gemini or OpenAI) to expand your dataset. This teaches you how to create high-quality alignment data at scale.
```bash
# Option 1: Gemini API
export GEMINI_API_KEY=your_key
python scripts/generate_data.py --count 10 --domain "machine learning"

# Option 2: OpenAI API
export OPENAI_API_KEY=your_key
python scripts/generate_data.py --count 10 --domain "python coding"
```

## Task 2: Loss function
Choose DPO or ORPO. Implement the TODO in `src/preference_lab/losses.py`.

## Task 3: Evaluation
Replace mock scores with model-derived scores or a deterministic scorer for CPU mode.

## Task 4: Report
Write a short report with dataset notes, config, metrics, and failure modes.
