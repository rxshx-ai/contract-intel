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


def test_every_wire_field_is_nullable():
    """Strict mode marks all fields required; a model with nothing to say emits
    null. A non-nullable field with a default therefore costs the ENTIRE chunk.
    """
    props = strict_schema(RawClause)["properties"]
    for name, prop in props.items():
        options = prop.get("anyOf", [prop])
        assert {"type": "null"} in options, f"{name} is not nullable"


def test_wire_schema_has_no_nested_enums():
    """Deliberate: Groq strict mode rejects the WHOLE response when one nested
    enum is violated, and models write "Customer" where we asked for "us".
    Enums are normalized in Python instead (see extract.normalize_party)."""
    schema = strict_schema(RawExtraction)

    def walk(node):
        if isinstance(node, dict):
            assert "enum" not in node, f"enum survived at {node}"
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)


def test_class_docstrings_are_not_shipped_to_the_model():
    """Docstrings are implementation notes, not prompt material."""
    schema = strict_schema(RawExtraction)
    assert "ON THE WIRE" not in json.dumps(schema)
    # per-field guidance is kept
    assert "Verbatim" in json.dumps(schema)


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
