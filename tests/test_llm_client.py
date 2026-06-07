import json
from unittest.mock import MagicMock

import pytest

from robomind.robomind.llm_client import (
    PERFORM_ACTION_TOOL,
    SYSTEM_PROMPT,
    AnthropicLLMClient,
    OpenAILLMClient,
    create_llm_client,
)
from robomind.robomind.petoi_commands import PETOI_COMMANDS


# ── tool schema ──────────────────────────────────────────────────────────────


def test_perform_action_tool_enum_covers_all_commands():
    enum_values = PERFORM_ACTION_TOOL["function"]["parameters"]["properties"]["action"][
        "enum"
    ]
    assert set(enum_values) == set(PETOI_COMMANDS.keys())


def test_perform_action_tool_action_is_required():
    required = PERFORM_ACTION_TOOL["function"]["parameters"]["required"]
    assert "action" in required


def test_system_prompt_mentions_perform_action():
    assert "perform_action" in SYSTEM_PROMPT


def test_system_prompt_lists_commands():
    # At least a sample of commands should appear in the prompt
    for key in ["wkF", "sit", "hi"]:
        assert key in SYSTEM_PROMPT


# ── factory ──────────────────────────────────────────────────────────────────


def test_create_llm_client_defaults_to_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = create_llm_client()
    assert isinstance(client, OpenAILLMClient)
    assert client.model == "gpt-5.4-nano"


def test_create_llm_client_respects_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4-mini")
    client = create_llm_client()
    assert client.model == "gpt-5.4-mini"


def test_create_llm_client_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm_client(provider="banana", model="x", api_key="x")


def test_create_llm_client_anthropic_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        create_llm_client(provider="anthropic", model="x")


# ── OpenAILLMClient ───────────────────────────────────────────────────────────


def test_openai_client_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        OpenAILLMClient(model="gpt-5.4-nano", api_key=None)


def _fake_client(content: str, tool_action: str | None = None) -> OpenAILLMClient:
    """Build an OpenAILLMClient whose underlying API call is mocked."""
    message = MagicMock()
    message.content = content
    if tool_action:
        tc = MagicMock()
        tc.function.name = "perform_action"
        tc.function.arguments = json.dumps({"action": tool_action})
        message.tool_calls = [tc]
    else:
        message.tool_calls = None

    response = MagicMock()
    response.choices = [MagicMock(message=message)]

    client = OpenAILLMClient.__new__(OpenAILLMClient)
    client.model = "gpt-5.4-nano"
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = response
    return client


def test_complete_text_only():
    client = _fake_client("Hello there!")
    text, action = client.complete([{"role": "user", "content": "hi"}])
    assert text == "Hello there!"
    assert action is None


def test_complete_with_action():
    client = _fake_client("Sitting down now!", tool_action="sit")
    text, action = client.complete(
        [{"role": "user", "content": "sit"}], tools=[PERFORM_ACTION_TOOL]
    )
    assert text == "Sitting down now!"
    assert action == "sit"


def test_complete_passes_tools_and_tool_choice_to_api():
    client = _fake_client("ok")
    messages = [{"role": "user", "content": "go"}]
    client.complete(messages, tools=[PERFORM_ACTION_TOOL])
    kwargs = client.client.chat.completions.create.call_args[1]
    assert kwargs["tools"] == [PERFORM_ACTION_TOOL]
    assert kwargs["tool_choice"] == "auto"


def test_complete_no_tools_omits_tool_choice():
    client = _fake_client("ok")
    client.complete([{"role": "user", "content": "hi"}], tools=None)
    kwargs = client.client.chat.completions.create.call_args[1]
    assert "tool_choice" not in kwargs


def test_complete_ignores_unknown_tool_calls():
    """A tool_call with a different name should not set action_key."""
    message = MagicMock()
    message.content = "text"
    tc = MagicMock()
    tc.function.name = "some_other_tool"
    tc.function.arguments = json.dumps({})
    message.tool_calls = [tc]
    response = MagicMock()
    response.choices = [MagicMock(message=message)]

    client = OpenAILLMClient.__new__(OpenAILLMClient)
    client.model = "gpt-5.4-nano"
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = response

    text, action = client.complete([{"role": "user", "content": "x"}])
    assert action is None
