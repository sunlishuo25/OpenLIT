"""Schema guard type validation and postflight integration regressions."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from openlit import GuardAction, GuardDeniedError, Pipeline, Schema
from openlit.guard._integration import (
    _extract_openai_input,
    _extract_openai_output,
    _make_async_guard_wrapper,
    _make_sync_guard_wrapper,
)


@pytest.mark.parametrize("schema_type", ["integer", "number"])
@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("location", ["root", "property", "item"])
def test_numeric_schema_rejects_booleans(schema_type, value, location):
    """JSON booleans are not numbers, including inside objects and arrays."""
    schema = {"type": schema_type}
    data = value
    path = "root"
    if location == "property":
        schema = {"type": "object", "properties": {"count": schema}}
        data = {"count": value}
        path = ".count"
    elif location == "item":
        schema = {"type": "array", "items": schema}
        data = [0, value]
        path = "[1]"

    result = Schema(schema=schema).evaluate(json.dumps(data))

    assert result.action == GuardAction.DENY
    assert result.classification == "schema_mismatch"
    assert result.explanation == f"Expected {schema_type} at {path}, got bool"


@pytest.mark.parametrize(
    "schema_type, value",
    [
        ("integer", 0),
        ("integer", 1),
        ("integer", -1),
        ("number", 0),
        ("number", 1),
        ("number", -1),
        ("number", 1.0),
        ("number", 0.5),
        ("number", -0.5),
        ("boolean", True),
        ("boolean", False),
    ],
)
def test_schema_accepts_matching_numeric_and_boolean_values(schema_type, value):
    """Zero, negative numbers, decimals and explicitly boolean fields pass."""
    result = Schema(schema={"type": schema_type}).evaluate(json.dumps(value))
    assert result.action == GuardAction.ALLOW


@pytest.mark.parametrize("schema_type", ["integer", "number"])
@pytest.mark.parametrize("value", [None, "1"])
def test_numeric_schema_still_rejects_other_types(schema_type, value):
    """The boolean correction preserves null and string type mismatches."""
    result = Schema(schema={"type": schema_type}).evaluate(json.dumps(value))
    assert result.action == GuardAction.DENY
    assert result.classification == "schema_mismatch"


@pytest.mark.parametrize("schema_type", ["integer", "number"])
@pytest.mark.parametrize("action", ["deny", "warn"])
def test_postflight_pipeline_preserves_configured_action(schema_type, action):
    """A schema violation follows the configured action even with fail-open."""
    pipeline = Pipeline(guards=[Schema(action=action, schema={"type": schema_type})])
    result = pipeline.evaluate("false", phase="postflight")

    assert result.action == GuardAction(action)
    assert result.results[0].classification == "schema_mismatch"
    assert pipeline.evaluate("false", phase="preflight").action == GuardAction.ALLOW


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("schema_type", ["integer", "number"])
@pytest.mark.parametrize("content", ["true", "false", "0"])
def test_provider_wrapper_checks_schema_output(is_async, schema_type, content):
    """Sync and async provider responses deny booleans and return valid numbers."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    pipeline = Pipeline(guards=[Schema(schema={"type": schema_type})])
    wrapper_factory = _make_async_guard_wrapper if is_async else _make_sync_guard_wrapper
    wrapper = wrapper_factory(pipeline, _extract_openai_input, _extract_openai_output)

    def provider(**kwargs):
        return response

    async def async_provider(**kwargs):
        return response

    def call():
        kwargs = {"messages": [{"role": "user", "content": "Return a count"}]}
        if is_async:
            return asyncio.run(wrapper(async_provider, None, (), kwargs))
        return wrapper(provider, None, (), kwargs)

    if content == "0":
        assert call() is response
    else:
        with pytest.raises(GuardDeniedError, match=f"Expected {schema_type} at root, got bool"):
            call()
