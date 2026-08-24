from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PreferenceExample(BaseModel):
    """One preference pair for DPO/ORPO-style alignment."""

    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, rejected: str, info: Any) -> str:
        chosen = info.data.get("chosen")
        if chosen is not None:
            norm_chosen = chosen.strip().lower()
            norm_rejected = rejected.strip().lower()
            if norm_chosen == norm_rejected:
                raise ValueError("chosen and rejected must differ")
            from difflib import SequenceMatcher

            if SequenceMatcher(None, norm_chosen, norm_rejected).ratio() > 0.95:
                raise ValueError("chosen and rejected are near-duplicates")
        return rejected
