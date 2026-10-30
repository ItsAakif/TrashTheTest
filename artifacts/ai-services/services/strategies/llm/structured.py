"""Schema-enforced structured generation with bounded repair (T306).

Pydantic v2 models are the single source of truth. Every call routes
through generate_structured, which:

- requests the provider's native strict JSON-schema mode when available
  (response_format json_schema), JSON mode otherwise
- validates against the model; on failure performs at most two repairs
  that include the validation error
- recovers a TRUNCATED JSON payload by closing the object the provider cut
  off (at most two recoveries) and validating the repaired payload against
  the same model, so a length-limited answer becomes usable output instead
  of a dead job
- asks for a SHORTER answer when the provider reported a length stop,
  instead of blindly re-requesting the same oversized payload
- fails with a typed StructuredGenerationError after that, never loosely
  parsed text

No json.loads-on-unvalidated-output in the generation path; no "return
JSON" prompt instruction without a validator behind it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from ..base import GenerationConfig, GenerationResult

# Shared output budget for every schema-enforced call that does not ask
# for something else. Callers that generate a bounded unit of work per call
# (e.g. one academic topic) may raise it via ``max_tokens``; raising it for
# an unbounded payload only moves the truncation, it does not remove it.
DEFAULT_MAX_OUTPUT_TOKENS = 4096

# Bounded truncated-JSON recoveries. A length-limited answer is repaired by
# closing the JSON object; if that does not validate, the repair loop below
# re-asks. Both loops are bounded, so a broken provider can never spin.
MAX_TRUNCATION_RECOVERIES = 2

# Provider finish reasons that mean "the output hit the token ceiling".
_TRUNCATION_FINISH_REASONS = frozenset(
    {"length", "max_tokens", "max_output_tokens", "token_limit", "truncated"}
)

# Characters that end a bare (unquoted) JSON value or a string/object/array.
_JSON_STRUCTURAL = frozenset('{}[],:"')

# A bare value is only recoverable at a cut when the token is complete.
_COMPLETE_SCALAR_RE = re.compile(
    r"\A(?:true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\Z"
)


class StructuredGenerationError(RuntimeError):
    """Typed failure: schema validation + bounded repair exhausted."""

    def __init__(
        self,
        message: str,
        raw: str = "",
        errors: Any = None,
        truncated: bool = False,
    ):
        super().__init__(message)
        self.raw = raw
        self.errors = errors
        # True when the provider's answer stopped mid-JSON (length stop or
        # an object that never closed). Callers use it to report an honest,
        # specific failure instead of a generic "invalid JSON".
        self.truncated = bool(truncated)


def schema_for(model: Type[BaseModel]) -> Dict[str, Any]:
    """JSON schema for the provider's strict json_schema mode."""
    try:
        return model.model_json_schema()
    except Exception:
        return {"type": "object"}


def _strict_config(
    schema: Dict[str, Any],
    name: str,
    max_tokens: Optional[int] = None,
) -> GenerationConfig:
    return GenerationConfig(
        temperature=0.2,
        # 4096 output budget: reasoning models spend tokens before the JSON;
        # 2048 truncated 10-card sets live (Phase B find). Still bounded.
        # Callers generating a bounded unit per call may raise it.
        max_tokens=int(max_tokens or DEFAULT_MAX_OUTPUT_TOKENS),
        response_format={
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema},
        },
    )


def _json_mode_config(max_tokens: Optional[int] = None) -> GenerationConfig:
    return GenerationConfig(
        temperature=0.2,
        max_tokens=int(max_tokens or DEFAULT_MAX_OUTPUT_TOKENS),
        response_format={"type": "json_object"},
    )


def _balanced_end(text: str, start: int) -> int:
    """Index of the '}' closing the object that opens at `start`, else -1."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _drop_key_boundary(out: List[str], key_start: int) -> int:
    """Index to cut at when a key (and its missing value) must be dropped."""
    index = key_start
    while index > 0 and out[index - 1].isspace():
        index -= 1
    if index > 0 and out[index - 1] == ",":
        index -= 1
    return index


def _close_truncated_json(text: str) -> str:
    """Close a JSON object the provider cut off; '' when unsalvageable.

    Drops the dangling partial value (or the key whose value never arrived),
    terminates a string left open mid-paragraph, then appends the missing
    closers in order. The candidate still has to parse AND validate, so a
    repair can never invent accepted content - it only recovers what the
    model already wrote before the output ceiling.
    """
    start = text.find("{")
    if start < 0:
        return ""
    out: List[str] = []
    stack: List[str] = []                 # closers still owed, innermost last
    value_end: Optional[int] = None       # out index just past the last value
    scalar_start: Optional[int] = None    # out index of a bare value in flight
    string_start: Optional[int] = None
    last_string_start: Optional[int] = None
    in_string = False
    escape = False
    for ch in text[start:]:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                last_string_start = string_start
                # +1 accounts for the closing quote appended below.
                value_end = len(out) + 1
            out.append(ch)
            continue
        if (ch.isspace() or ch in _JSON_STRUCTURAL) and scalar_start is not None:
            value_end = len(out)
            scalar_start = None
        if ch == '"':
            in_string = True
            string_start = len(out)
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                break                       # unbalanced: stop, close what is open
            stack.pop()
            out.append(ch)
            value_end = len(out)
            continue
        elif ch == ":" and last_string_start is not None:
            # The value for this key never arrived: drop the key with it.
            value_end = _drop_key_boundary(out, last_string_start)
        out.append(ch)
        if scalar_start is None and not ch.isspace() and ch not in _JSON_STRUCTURAL:
            scalar_start = len(out) - 1
    if scalar_start is not None:
        # A bare value ended at EOF. Keep it only when the token is complete:
        # "nul"/"12."/"tru" are cut keywords/numbers, so the previous value is
        # the last safe boundary.
        token = "".join(out[scalar_start:])
        if _COMPLETE_SCALAR_RE.match(token):
            value_end = len(out)
    if in_string and string_start is not None:
        previous = "".join(out[:string_start]).rstrip()
        # Strings are values inside an array; inside an object a string
        # followed by ':' is a key, and a key without its value is dropped.
        in_object = bool(stack) and stack[-1] == "}"
        is_value = previous.endswith(":") or not in_object
        if is_value:
            # A truncated value: keep the text, close the string.
            if out and out[-1] == "\\":
                out.pop()                    # dangling escape is not valid JSON
            out.append('"')
            value_end = len(out)
        else:
            value_end = _drop_key_boundary(out, string_start)
    if value_end is None or value_end <= 0:
        return ""
    candidate = "".join(out[:value_end]) + "".join(reversed(stack))
    try:
        json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return ""
    return candidate


def _extract_json(text: str, allow_truncation_repair: bool = True) -> Any:
    """Parse one JSON object; validation ALWAYS happens afterwards.

    Generation-path JSON goes through this helper plus model_validate,
    never bare json.loads in service code (T306).

    With ``allow_truncation_repair`` a length-limited answer (the object
    never closes) is closed back up once; the caller still validates the
    result. Repair never runs twice on the same text.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Bounded salvage: first balanced {...} block, validated afterwards.
    start = text.find("{")
    if start < 0:
        raise StructuredGenerationError("LLM output contained no JSON object", raw=text)
    end = _balanced_end(text, start)
    if end < 0:
        if allow_truncation_repair:
            candidate = _close_truncated_json(text)
            if candidate:
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    pass
        raise StructuredGenerationError(
            "LLM output JSON was truncated", raw=text, truncated=True
        )
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError) as exc:
        raise StructuredGenerationError(f"LLM output was not valid JSON: {exc}", raw=text)


def _is_truncated(result: GenerationResult) -> bool:
    return str(getattr(result, "finish_reason", "") or "").lower() in _TRUNCATION_FINISH_REASONS


def coerce_enum_errors(model, payload):
    """Repair enum/literal fields Pydantic rejected, deterministically.

    Weak models write prose or code into enum fields.  Mapping the value
    back to a legal member costs nothing and must never burn an LLM repair
    call or fail the request.  Returns (payload, changed).
    """
    import types as _types
    from enum import Enum as _Enum
    changed = False
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            payload[index], item_changed = coerce_enum_errors(model, item)
            changed = changed or item_changed
        return payload, changed
    if not isinstance(payload, dict):
        return payload, False
    fields = getattr(model, "model_fields", {}) or {}
    for name, field in fields.items():
        annotation = field.annotation
        # Unwrap generic aliases like list[Model] to get the inner type.
        if isinstance(annotation, _types.GenericAlias):
            args = getattr(annotation, "__args__", ())
            if args and isinstance(args[0], type) and hasattr(args[0], "model_fields"):
                inner = args[0]
                if name in payload and isinstance(payload[name], list):
                    payload[name], nested_changed = coerce_enum_errors(inner, payload[name])
                    changed = changed or nested_changed
                continue
        if isinstance(annotation, type) and issubclass(annotation, _Enum):
            if name in payload and payload[name] is not None:
                raw = payload[name]
                if not isinstance(raw, annotation):
                    target = None
                    text = str(raw).strip()
                    for member in annotation:
                        if text.lower() in (str(member.value).lower(),
                                            str(member.name).lower()):
                            target = member
                            break
                    if target is None:
                        default = field.default
                        target = (default if isinstance(default, annotation)
                                  else next(iter(annotation)))
                    payload[name] = target.value if isinstance(target, _Enum) else target
                    changed = True
        elif isinstance(annotation, type) and hasattr(annotation, "model_fields"):
            if name in payload and isinstance(payload[name], (dict, list)):
                payload[name], nested_changed = coerce_enum_errors(annotation, payload[name])
                changed = changed or nested_changed
    return payload, changed


async def generate_structured(
    provider: Any,
    model: Type[BaseModel],
    prompt: str,
    max_repairs: int = 2,
    config: Optional[GenerationConfig] = None,
    max_tokens: Optional[int] = None,
    extra_validator: Optional[Any] = None,
) -> BaseModel:
    """Generate + validate against `model`, repairing at most twice.

    ``max_tokens`` raises the per-call output budget for callers whose payload
    is a bounded unit of work (one topic, one card set). Omit it to keep the
    shared 4096 default.

    ``extra_validator`` runs after schema validation and before the payload is
    returned (e.g. the engine's fabricated-citation gate). A failure there is
    treated exactly like a schema failure: the model gets the bounded repair
    attempts with a corrective prompt that carries the validator's message,
    and the ORIGINAL typed error is re-raised once the bound is exhausted --
    so a caller matching on "Fabricated ..." still sees that error, never a
    generic wrap.
    """
    schema = schema_for(model)
    name = getattr(model, "__name__", "output")
    first = config or _strict_config(schema, name, max_tokens)

    raw_text = ""
    result: Optional[GenerationResult] = None
    last_errors: Any = None
    recoveries = 0
    current_prompt = prompt
    for attempt in range(max_repairs + 1):
        if attempt == 1 and config is None:
            # Strict mode failed once: fall back to plain JSON mode.
            first = _json_mode_config(max_tokens)
        try:
            if hasattr(provider, "chat"):
                from ..base import ChatMessage

                result = await provider.chat(
                    [ChatMessage(role="user", content=current_prompt)], first
                )
            else:
                result = await provider.generate(current_prompt, first)
            raw_text = result.text or ""
        except Exception as exc:
            if attempt >= max_repairs:
                raise StructuredGenerationError(
                    f"LLM call failed during structured generation: {exc}"
                )
            continue
        length_stop = _is_truncated(result)
        data: Any = None
        parse_failed = False
        truncated = False
        try:
            data = _extract_json(raw_text, allow_truncation_repair=False)
        except StructuredGenerationError as exc:
            parse_failed = True
            truncated = bool(getattr(exc, "truncated", False)) or length_stop
            if truncated and recoveries < MAX_TRUNCATION_RECOVERIES:
                # Recover the payload the provider cut off, then validate it
                # against the same model below. Bounded: two recoveries.
                recoveries += 1
                try:
                    data = _extract_json(raw_text, allow_truncation_repair=True)
                    parse_failed = False
                except StructuredGenerationError:
                    pass
        if parse_failed:
            if attempt >= max_repairs:
                message = (
                    "LLM output JSON was truncated after "
                    f"{recoveries} recovery attempt(s)"
                    if truncated
                    else "LLM output was not valid JSON"
                )
                raise StructuredGenerationError(
                    message, raw=raw_text, truncated=truncated
                )
            if length_stop:
                # Honest retry: the answer hit the output ceiling, so asking
                # for the same size again only truncates again.
                current_prompt = (
                    f"{prompt}\n\nYour previous answer was cut off because it "
                    "exceeded the output limit. Answer again with at most half "
                    "as many paragraphs, shorter sentences, and the same "
                    "citations. Output ONLY the JSON object."
                )
            else:
                current_prompt = (
                    f"{prompt}\n\nYour previous response was not valid JSON. "
                    f"Respond with ONLY a JSON object matching the schema."
                )
            continue
        payload, coerced = coerce_enum_errors(model, data)
        if coerced:
            try:
                return model.model_validate(payload)
            except ValidationError:
                pass
        try:
            validated = model.model_validate(data)
        except ValidationError as exc:
            last_errors = exc.errors()
            if attempt >= max_repairs:
                break
            current_prompt = (
                f"{prompt}\n\nYour previous response failed schema validation "
                f"with these errors:\n{json.dumps(last_errors)[:2000]}\n"
                f"Respond with ONLY a corrected JSON object matching the schema."
            )
            continue
        if extra_validator is None:
            return validated
        try:
            extra_validator(validated)
        except Exception as exc:
            # Grounding failure (fabricated citation ids): it belongs to the
            # SAME bounded repair loop, so the model may correct itself. When
            # the bound is exhausted the ORIGINAL typed error propagates
            # untouched, so callers matching on "Fabricated ..." keep working.
            last_errors = str(exc)
            if attempt >= max_repairs:
                raise
            current_prompt = (
                f"{prompt}\n\nYour previous response was rejected by the "
                f"grounding check:\n{str(exc)[:2000]}\n"
                "Cite ONLY chunk ids that appear verbatim in the source "
                "passages above; never invent, shorten, or paraphrase an id. "
                "Respond with ONLY a corrected JSON object matching the schema."
            )
            continue
        return validated
    raise StructuredGenerationError(
        f"Schema validation failed after {max_repairs} repairs", raw=raw_text,
        errors=last_errors,
    )


__all__ = [
    "StructuredGenerationError",
    "coerce_enum_errors",
    "generate_structured",
    "schema_for",
]
