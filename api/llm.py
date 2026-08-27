"""Groq client and JSON-schema plumbing.

Isolated from extract.py so the provider is one small, replaceable file. The
rest of the system knows nothing about who serves the model.

Groq's strict structured-output mode has requirements Pydantic does not emit by
default: every property must be listed in `required`, and every object needs
`additionalProperties: false`. `strict_schema()` performs that rewrite.
"""

from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# 131k context, 65k max output, strict json_schema support, ~500 tok/s.
DEFAULT_MODEL = "openai/gpt-oss-120b"
MODEL = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

# Models Groq documents as supporting strict json_schema. Anything else falls
# back to json_object mode with the schema embedded in the prompt.
STRICT_SCHEMA_MODELS = {
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3.8-27b",
}

MAX_INPUT_CHARS = 380_000   # ~120k tokens, under the 131k window


class ExtractionUnavailable(RuntimeError):
    """No credentials, no cache, or the model could not produce valid output."""


def client():
    import groq

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise ExtractionUnavailable(
            "GROQ_API_KEY is not set. Export it, or run against seeded demo "
            "contracts (see eval/make_fixtures.py --seed-cache)."
        )
    return groq.Groq(api_key=key, max_retries=3, timeout=120.0)


# --------------------------------------------------------------------------
# schema rewriting
# --------------------------------------------------------------------------

def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema -> Groq strict-mode schema."""
    return _harden(model.model_json_schema())


_DROP_KEYS = {"default", "format", "examples", "$comment"}


def _harden(node: Any) -> Any:
    if isinstance(node, list):
        return [_harden(v) for v in node]
    if not isinstance(node, dict):
        return node

    out = {k: _harden(v) for k, v in node.items() if k not in _DROP_KEYS}

    if out.get("type") == "object" or "properties" in out:
        properties = out.get("properties", {})
        out["additionalProperties"] = False
        # Strict mode requires EVERY property in `required`. Optionality is
        # expressed by the union-with-null that Pydantic already emits.
        out["required"] = list(properties.keys())
    return out


def schema_hint(schema: dict[str, Any]) -> str:
    """Compact schema text for models without strict mode."""
    return json.dumps(schema, separators=(",", ":"))


# --------------------------------------------------------------------------
# the call
# --------------------------------------------------------------------------

def complete_json(
    system: str,
    user: str,
    output_model: type[T],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 32_000,
    schema_name: str = "extraction",
) -> T:
    """One structured completion, validated into `output_model`.

    Temperature is 0 so a given document yields the same extraction twice --
    the reasoning layer downstream is deterministic and it would be odd for its
    input not to be.
    """
    model = model or MODEL
    if len(user) > MAX_INPUT_CHARS:
        raise ExtractionUnavailable(
            f"Document is {len(user):,} characters, above the {MAX_INPUT_CHARS:,} "
            f"limit for {model}. Chunked extraction is not implemented; the "
            f"document is NOT silently truncated."
        )

    schema = strict_schema(output_model)
    supports_strict = model in STRICT_SCHEMA_MODELS

    if supports_strict:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
    else:
        response_format = {"type": "json_object"}
        messages = [
            {"role": "system",
             "content": f"{system}\n\nReply with JSON matching this schema "
                        f"exactly:\n{schema_hint(schema)}"},
            {"role": "user", "content": user},
        ]

    completion = client().chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = completion.choices[0].message.content or ""
    finish = completion.choices[0].finish_reason
    if finish == "length":
        raise ExtractionUnavailable(
            f"Model hit the {max_tokens:,}-token output cap mid-extraction. The "
            f"result would be truncated, so it is rejected rather than used."
        )
    try:
        return output_model.model_validate_json(content)
    except ValidationError as exc:
        raise ExtractionUnavailable(
            f"{model} returned JSON that does not match the extraction schema "
            f"({exc.error_count()} errors). First: {exc.errors()[0].get('msg')}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ExtractionUnavailable(f"{model} returned malformed JSON: {exc}") from exc


def usage_of(completion) -> dict[str, Any]:
    u = getattr(completion, "usage", None)
    return {} if u is None else {
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_time": getattr(u, "total_time", None),
    }
