import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.gemini_cli_client import GeminiCLIClient


def test_gemini_cli_client_maps_non_streaming_response():
    payload = {
        "id": "resp-1",
        "model": "gemini-3.1-pro-preview",
        "content": "hello from gemini",
        "reasoning_content": "thinking",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
            }
        ],
        "finish_reason": "tool_calls",
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }

    completed = SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    with patch("agent.gemini_cli_client.subprocess.run", return_value=completed):
        client = GeminiCLIClient(command="/usr/bin/node", args=["/tmp/gemini_cli_bridge.mjs"])
        response = client.chat.completions.create(model="gemini-3.1-pro-preview", messages=[{"role": "user", "content": "hi"}])

    assert response.id == "resp-1"
    assert response.model == "gemini-3.1-pro-preview"
    assert response.choices[0].message.content == "hello from gemini"
    assert response.choices[0].message.reasoning_content == "thinking"
    assert response.choices[0].message.tool_calls[0].function.name == "read_file"
    assert response.usage.total_tokens == 18


def test_gemini_cli_client_synthesizes_stream_chunks():
    payload = {
        "id": "resp-2",
        "model": "gemini-3-flash-preview",
        "content": "final content",
        "reasoning_content": "reasoning",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "grep", "arguments": '{"pattern":"Gemini"}'},
            }
        ],
        "finish_reason": "tool_calls",
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }

    completed = SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    with patch("agent.gemini_cli_client.subprocess.run", return_value=completed):
        client = GeminiCLIClient(command="/usr/bin/node", args=["/tmp/gemini_cli_bridge.mjs"])
        chunks = list(
            client.chat.completions.create(
                model="gemini-3-flash-preview",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
        )

    assert chunks[0].choices[0].delta.reasoning_content == "reasoning"
    assert chunks[1].choices[0].delta.tool_calls[0].function.name == "grep"
    assert chunks[2].choices[0].delta.content == "final content"
    assert chunks[3].choices[0].finish_reason == "tool_calls"
    assert chunks[4].usage.total_tokens == 8
