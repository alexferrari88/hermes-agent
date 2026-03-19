"""OpenAI-compatible shim for Hermes' Gemini CLI bridge."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

GEMINI_CLI_MARKER_BASE_URL = "gemini-cli://oauth"
_DEFAULT_TIMEOUT_SECONDS = 900.0


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_bridge_args() -> list[str]:
    return [str(_project_root() / "scripts" / "gemini_cli_bridge.mjs")]


def _resolve_command() -> str:
    return os.getenv("HERMES_GEMINI_BRIDGE_COMMAND", "").strip() or "node"


def _resolve_args() -> list[str]:
    raw_args = os.getenv("HERMES_GEMINI_BRIDGE_ARGS", "").strip()
    return shlex.split(raw_args) if raw_args else _default_bridge_args()


def _resolve_oauth_file() -> str:
    raw = os.getenv("HERMES_GEMINI_OAUTH_FILE", "").strip()
    if raw:
        return str(Path(raw).expanduser().resolve())
    return str((Path.home() / ".gemini" / "oauth_creds.json").resolve())


class _GeminiChatCompletions:
    def __init__(self, client: "GeminiCLIClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _GeminiChatNamespace:
    def __init__(self, client: "GeminiCLIClient"):
        self.completions = _GeminiChatCompletions(client)


class GeminiCLIClient:
    """Minimal client exposing chat.completions.create()."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        oauth_file: str | None = None,
        default_headers: dict[str, str] | None = None,
        cwd: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "gemini-cli"
        self.base_url = base_url or GEMINI_CLI_MARKER_BASE_URL
        self.command = command or _resolve_command()
        self.args = list(args or _resolve_args())
        self.oauth_file = oauth_file or _resolve_oauth_file()
        self._default_headers = dict(default_headers or {})
        self._cwd = str(Path(cwd or os.getcwd()).resolve())
        self.chat = _GeminiChatNamespace(self)
        self.is_closed = False

    def close(self) -> None:
        self.is_closed = True

    def _create_chat_completion(self, **kwargs: Any) -> Any:
        stream = bool(kwargs.pop("stream", False))
        timeout = float(kwargs.pop("timeout", _DEFAULT_TIMEOUT_SECONDS) or _DEFAULT_TIMEOUT_SECONDS)
        response_payload = self._invoke_bridge(kwargs, timeout_seconds=timeout)
        if stream:
            return self._stream_response(response_payload)
        return self._normalize_response(response_payload)

    def _invoke_bridge(self, payload: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
        request_payload = {
            **payload,
            "oauth_file": self.oauth_file,
        }
        try:
            proc = subprocess.run(
                [self.command] + self.args,
                input=json.dumps(request_payload),
                capture_output=True,
                text=True,
                cwd=self._cwd,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Gemini bridge command '{self.command}'. "
                "Install Node.js or set HERMES_GEMINI_BRIDGE_COMMAND."
            ) from exc

        stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
        stdout_text = stdout_lines[-1] if stdout_lines else ""
        stderr_text = (proc.stderr or "").strip()

        if proc.returncode != 0:
            detail = stderr_text or stdout_text or f"exit code {proc.returncode}"
            raise RuntimeError(f"Gemini CLI bridge failed: {detail}")

        if not stdout_text:
            raise RuntimeError("Gemini CLI bridge returned no output.")

        try:
            payload = json.loads(stdout_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gemini CLI bridge returned invalid JSON: {stdout_text[:200]}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Gemini CLI bridge returned an invalid response payload.")
        return payload

    def _normalize_response(self, payload: dict[str, Any]) -> SimpleNamespace:
        assistant_message = SimpleNamespace(
            role="assistant",
            content=(payload.get("content") or None),
            tool_calls=self._tool_calls(payload),
            reasoning=(payload.get("reasoning_content") or None),
            reasoning_content=(payload.get("reasoning_content") or None),
            reasoning_details=None,
        )
        choice = SimpleNamespace(
            index=0,
            message=assistant_message,
            finish_reason=payload.get("finish_reason") or "stop",
        )
        return SimpleNamespace(
            id=payload.get("id") or "gemini-cli",
            model=payload.get("model") or "gemini-cli",
            choices=[choice],
            usage=self._usage(payload.get("usage") or {}),
        )

    def _stream_response(self, payload: dict[str, Any]) -> Iterator[SimpleNamespace]:
        usage = self._usage(payload.get("usage") or {})
        tool_calls = self._tool_calls(payload)
        content = payload.get("content") or ""
        reasoning = payload.get("reasoning_content") or ""
        finish_reason = payload.get("finish_reason") or "stop"
        model = payload.get("model") or "gemini-cli"

        def _iter() -> Iterator[SimpleNamespace]:
            if reasoning:
                yield self._chunk(
                    model=model,
                    delta=SimpleNamespace(reasoning_content=reasoning, content=None, tool_calls=None),
                )
            if tool_calls:
                yield self._chunk(
                    model=model,
                    delta=SimpleNamespace(content=None, tool_calls=self._tool_call_deltas(tool_calls)),
                )
            if content:
                yield self._chunk(
                    model=model,
                    delta=SimpleNamespace(content=content, tool_calls=None),
                )
            yield self._chunk(
                model=model,
                delta=SimpleNamespace(content=None, tool_calls=None),
                finish_reason=finish_reason,
            )
            yield SimpleNamespace(choices=[], model=model, usage=usage)

        return _iter()

    def _tool_calls(self, payload: dict[str, Any]) -> list[SimpleNamespace]:
        tool_calls = []
        raw_tool_calls = payload.get("tool_calls") or []
        for tool_call in raw_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") or {}
            tool_calls.append(
                SimpleNamespace(
                    id=tool_call.get("id") or "",
                    type=tool_call.get("type") or "function",
                    function=SimpleNamespace(
                        name=function.get("name") or "",
                        arguments=function.get("arguments") or "{}",
                    ),
                )
            )
        return tool_calls

    def _tool_call_deltas(self, tool_calls: list[SimpleNamespace]) -> list[SimpleNamespace]:
        deltas = []
        for index, tool_call in enumerate(tool_calls):
            deltas.append(
                SimpleNamespace(
                    index=index,
                    id=tool_call.id,
                    function=SimpleNamespace(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                )
            )
        return deltas

    def _usage(self, usage: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            ),
        )

    def _chunk(
        self,
        *,
        model: str,
        delta: SimpleNamespace,
        finish_reason: str | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            model=model,
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=delta,
                    finish_reason=finish_reason,
                )
            ],
        )


class _AsyncGeminiChatCompletions:
    def __init__(self, client: GeminiCLIClient):
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self._client.chat.completions.create, **kwargs)


class _AsyncGeminiChatNamespace:
    def __init__(self, client: GeminiCLIClient):
        self.completions = _AsyncGeminiChatCompletions(client)


class AsyncGeminiCLIClient:
    def __init__(self, client: GeminiCLIClient):
        self.api_key = client.api_key
        self.base_url = client.base_url
        self._sync_client = client
        self.chat = _AsyncGeminiChatNamespace(client)

    async def close(self) -> None:
        await asyncio.to_thread(self._sync_client.close)
