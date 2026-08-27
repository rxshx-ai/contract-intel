"""Provider layer: schema hardening and honest failure. No network."""

import json
import os

import pytest

from api.llm import (
    MAX_INPUT_CHARS,
    MODEL,
    STRICT_SCHEMA_MODELS,
    ExtractionUnavailable,
    client,
    complete_json,
    strict_schema,
)
from api.extract import RawClause, RawExtraction


def test_default_model_supports_strict_schema():
    assert MODEL in STRICT_SCHEMA_MODELS or os.environ.get("GROQ_MODEL")


def test_every_object_forbids_extra_properties():
    schema = strict_schema(RawExtraction)

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node.get("properties", {}))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)


def test_defaults_are_stripped():
    """Groq strict mode rejects `default`; Pydantic emits it for every optional."""
    assert "default" not in json.dumps(strict_schema(RawClause))


def test_optional_fields_become_nullable_unions():
    props = strict_schema(RawClause)["properties"]
    assert {"type": "null"} in props["amount"]["anyOf"]
    assert props["clause_type"] == {"type": "string", "title": "Clause Type"}


def test_enums_survive_hardening():
    schema = strict_schema(RawExtraction)
    role = schema["properties"]["our_role"]
    assert set(role["enum"]) == {"buyer", "seller", "mutual"}


def test_missing_key_fails_with_an_actionable_message(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ExtractionUnavailable) as exc:
        client()
    assert "GROQ_API_KEY is not set" in str(exc.value)


def test_oversized_document_is_rejected_not_truncated(monkeypatch):
    """Silent truncation would corrupt every offset downstream."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    with pytest.raises(ExtractionUnavailable) as exc:
        complete_json("sys", "x" * (MAX_INPUT_CHARS + 1), RawExtraction)
    assert "NOT silently truncated" in str(exc.value)
