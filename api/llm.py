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

# Free tier is 8,000 TPM across every strict-schema model. That ceiling covers
# prompt AND the max_tokens reservation, so a large reservation alone can 413 a
# tiny request. Override once on a paid tier.
TPM_LIMIT = int(os.environ.get("GROQ_TPM_LIMIT", "8000"))
TPM_HEADROOM = 0.92         # leave slack for tokenizer estimate error
MAX_OUTPUT_TOKENS = int(os.environ.get("GROQ_MAX_OUTPUT", "5500"))


class ExtractionUnavailable(RuntimeError):
    """No credentials, no cache, or the model could not produce valid output."""


class OutputTruncated(ExtractionUnavailable):
    """The extraction did not fit in the output budget.

    Recoverable: the caller can split the chunk and retry. Distinct from a
    generic failure precisely so that recovery is possible.
    """


class TokenBudget:
    """Client-side TPM throttle over a rolling 60-second window.

    Groq counts `max_tokens` against the per-minute budget at request time, not
    at completion, so the reservation must be included here too. Without this a
    multi-chunk contract 413s partway through and leaves a half-extraction.
    """

    def __init__(self, limit: int = TPM_LIMIT):
        self.limit = int(limit * TPM_HEADROOM)
        self._events: list[tuple[float, int]] = []

    def _spent(self, now: float) -> int:
        self._events = [(t, n) for t, n in self._events if now - t < 60.0]
        return sum(n for _, n in self._events)

    def reserve(self, tokens: int, verbose: bool = False) -> float:
        """Block until `tokens` fit in the window. Returns seconds waited."""
        import time

        waited = 0.0
        while True:
            now = time.monotonic()
            spent = self._spent(now)
            if spent + tokens <= self.limit or not self._events:
                self._events.append((now, tokens))
                return waited
            oldest = min(t for t, _ in self._events)
            sleep_for = max(0.5, 60.0 - (now - oldest) + 0.5)
            if verbose:
                print(f"    [throttle] {spent:,}/{self.limit:,} tokens used this "
                      f"minute; waiting {sleep_for:.0f}s", flush=True)
            time.sleep(sleep_for)
            waited += sleep_for


BUDGET = TokenBudget()


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
        # Pydantic turns the class docstring into `description`. That is internal
        # commentary, not guidance for the model -- ship neither the tokens nor
        # the confusion. Per-field descriptions are kept.
        out.pop("description", None)
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
    max_tokens: int | None = None,
    schema_name: str = "extraction",
    verbose: bool = False,
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

    from api.chunking import estimate_tokens

    prompt_tokens = estimate_tokens(system) + estimate_tokens(user) + 64
    if max_tokens is None:
        # Fit the whole request inside one minute's budget.
        available = int(TPM_LIMIT * TPM_HEADROOM) - prompt_tokens
        max_tokens = max(512, min(MAX_OUTPUT_TOKENS, available))
    if prompt_tokens + max_tokens > TPM_LIMIT:
        raise ExtractionUnavailable(
            f"Request needs ~{prompt_tokens + max_tokens:,} tokens but the "
            f"per-minute limit is {TPM_LIMIT:,}. Reduce chunk size "
            f"(api/chunking.DEFAULT_MAX_CHARS) or raise GROQ_TPM_LIMIT on a paid tier."
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

    BUDGET.reserve(prompt_tokens + max_tokens, verbose=verbose)
    completion = _create_with_retry(
        model=model, messages=messages, response_format=response_format,
        temperature=temperature, max_tokens=max_tokens, verbose=verbose,
    )
    content = completion.choices[0].message.content or ""
    finish = completion.choices[0].finish_reason
    if finish == "length":
        raise OutputTruncated(
            f"Extraction exceeded the {max_tokens:,}-token output budget. "
            f"Truncated output is rejected, never used."
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


def _create_with_retry(*, model, messages, response_format, temperature,
                       max_tokens, verbose=False, attempts: int = 4):
    """Retry on rate limiting, honouring the server's own retry-after."""
    import time

    import groq

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return client().chat.completions.create(
                model=model, messages=messages, response_format=response_format,
                temperature=temperature, max_tokens=max_tokens,
            )
        except groq.RateLimitError as exc:
            last = exc
            delay = _retry_after(exc) or min(60.0, 5.0 * (2 ** attempt))
            if verbose:
                print(f"    [rate limit] retrying in {delay:.0f}s "
                      f"(attempt {attempt + 1}/{attempts})", flush=True)
            time.sleep(delay)
        except groq.BadRequestError as exc:
            if "json_validate_failed" in str(exc):
                # Constrained decoding that runs out of room fails validation
                # rather than reporting finish_reason=length.
                raise OutputTruncated(
                    "Model output did not match the schema, usually because the "
                    "extraction ran out of output budget mid-object."
                ) from exc
            raise
        except groq.APIStatusError as exc:
            if exc.status_code and exc.status_code >= 500:
                last = exc
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    raise ExtractionUnavailable(f"rate limited after {attempts} attempts: {last}")


def _retry_after(exc) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key in ("retry-after", "x-ratelimit-reset-tokens"):
        value = headers.get(key)
        if not value:
            continue
        try:
            return float(str(value).rstrip("s")) + 1.0
        except ValueError:
            continue
    return None


def usage_of(completion) -> dict[str, Any]:
    u = getattr(completion, "usage", None)
    return {} if u is None else {
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_time": getattr(u, "total_time", None),
    }
