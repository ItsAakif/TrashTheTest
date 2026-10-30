"""Tests for notes robustness: coerce_enum_errors and TopicNotes validation."""

from enum import Enum

import pytest
from pydantic import BaseModel, ValidationError

from services.academic.types import TopicNotes, TopicShape
from services.strategies.llm.structured import coerce_enum_errors


def _minimal_valid_payload():
    """Build a minimal valid TopicNotes payload."""
    return {
        "unit_title": "Unit 1",
        "topic_title": "Functions",
        "heading": "Functions",
        "body_paragraphs": ["A function is a block of code."],
        "key_terms": ["function"],
        "shape": "concept",
        "citations": [{"chunk_id": "chunk-1", "span": "function"}],
        "no_source": False,
        "sufficiency": "SUFFICIENT",
        "source_gap_reason": None,
        "provenance_note": "test",
    }


def test_coerce_enum_errors_repairs_code_in_shape():
    """A payload with code in 'shape' coerces to a valid TopicNotes."""
    payload = _minimal_valid_payload()
    payload["shape"] = "def function_name(parameters): body"

    payload, changed = coerce_enum_errors(TopicNotes, payload)
    assert changed is True

    notes = TopicNotes.model_validate(payload)
    assert notes.shape == TopicShape.CONCEPT


def test_coerce_enum_errors_still_raises_for_other_errors():
    """A payload with bad shape AND missing required field still raises."""
    payload = _minimal_valid_payload()
    payload["shape"] = "def function_name(parameters): body"
    del payload["unit_title"]  # missing required field

    payload, changed = coerce_enum_errors(TopicNotes, payload)
    assert changed is True

    with pytest.raises(ValidationError):
        TopicNotes.model_validate(payload)


class _Color(str, Enum):
    RED = "red"
    BLUE = "blue"


class _Inner(BaseModel):
    color: _Color = _Color.RED


class _Outer(BaseModel):
    items: list[_Inner] = []


def test_coerce_enum_errors_handles_nested_models():
    """A list of dicts containing an enum field is coerced too."""
    payload = {
        "items": [
            {"color": "def something(): pass"},
            {"color": "blue"},
        ]
    }

    payload, changed = coerce_enum_errors(_Outer, payload)
    assert changed is True

    outer = _Outer.model_validate(payload)
    assert outer.items[0].color == _Color.RED
    assert outer.items[1].color == _Color.BLUE
