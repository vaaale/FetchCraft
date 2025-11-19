from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)


def openai_history_to_pydantic_ai(
    openai_messages: List[Dict[str, Any]],
    *,
    include_system: bool = False,
) -> List[ModelMessage]:
    """
    Convert OpenAI chat messages into pydantic-ai ModelMessage objects.

    Parameters
    ----------
    openai_messages:
        List of messages in OpenAI chat format, e.g.:
        {
          "role": "user" | "assistant" | "system" | "tool" | "function",
          "content": "...",
          # optional:
          "tool_calls": [...],
          "name": "...",
          "tool_call_id": "...",
          "created_at": int|float|str,
          "timestamp": str,
          ...
        }

    include_system:
        If True, include system messages as SystemPromptPart inside ModelRequest.
        Typically False if you're already passing `system_prompt` to the Agent.
    """
    result: List[ModelMessage] = []

    for msg in openai_messages:
        role = msg.get("role")
        content = msg.get("content")
        ts = _parse_timestamp(msg)

        # --- SYSTEM ---
        if role == "system":
            if not include_system:
                continue
            if not content:
                continue
            text = _content_to_text(content)
            result.append(
                ModelRequest(
                    parts=[
                        SystemPromptPart(
                            content=text,
                            timestamp=ts,
                        )
                    ]
                )
            )
            continue

        # --- USER ---
        if role == "user":
            if not content:
                continue
            text = _content_to_text(content)
            result.append(
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=text,
                            timestamp=ts,
                        )
                    ]
                )
            )
            continue

        # --- ASSISTANT ---
        if role == "assistant":
            tool_calls = msg.get("tool_calls") or msg.get("tool_calls", [])
            parts = []

            # 1) tool calls (if any)
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {}) or {}
                    name = func.get("name") or tc.get("name")
                    args_raw = func.get("arguments")

                    # pydantic-ai is happy with str or dict for args :contentReference[oaicite:1]{index=1}
                    if isinstance(args_raw, (dict, list)):
                        args = args_raw
                    else:
                        args = args_raw  # likely a JSON string from OpenAI

                    parts.append(
                        ToolCallPart(
                            tool_name=name or "unknown_tool",
                            args=args,
                            tool_call_id=tc.get("id"),
                        )
                    )

            # 2) assistant text (if any)
            if content:
                text = _content_to_text(content)
                if text:
                    parts.append(TextPart(content=text))

            if not parts:
                # nothing meaningful, skip
                continue

            result.append(
                ModelResponse(
                    parts=parts,
                    timestamp=ts,
                )
            )
            continue

        # --- TOOL / FUNCTION (tool return) ---
        if role in ("tool", "function"):
            name = msg.get("name") or msg.get("tool_name") or "unknown_tool"
            tool_call_id = msg.get("tool_call_id")
            tool_content = msg.get("content")

            parsed_content: Any = tool_content
            if isinstance(tool_content, str):
                # Often we store tool returns as JSON strings in OpenAI format
                try:
                    parsed_content = json.loads(tool_content)
                except Exception:
                    parsed_content = tool_content

            result.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name=name,
                            content=parsed_content,
                            tool_call_id=tool_call_id or "",
                            timestamp=ts,
                        )
                    ]
                )
            )
            continue

        # --- UNKNOWN ROLE ---
        # You might want to log this instead of raising
        raise ValueError(f"Unsupported OpenAI message role: {role!r}")

    return result


def _content_to_text(content: Any) -> str:
    """
    Normalize OpenAI 'content' into a plain string.

    Supports:
    - str
    - list of {\"type\": \"text\", \"text\": ...} (Responses API / new Chat format)
    """
    if isinstance(content, str):
        return content

    # list of parts
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_val = part.get("text")
                # can be just a string or an object with 'value'
                if isinstance(text_val, str):
                    texts.append(text_val)
                elif isinstance(text_val, dict) and "value" in text_val:
                    texts.append(str(text_val["value"]))
        return "".join(texts)

    # fallback – try some common patterns
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        return json.dumps(content, ensure_ascii=False)

    return str(content)


def _parse_timestamp(msg: Dict[str, Any]) -> datetime:
    """
    Try to extract a timestamp from the OpenAI message for nicer history;
    fall back to 'now' in UTC if nothing usable is found.
    """
    raw = msg.get("timestamp") or msg.get("created_at")

    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)

    if isinstance(raw, str):
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                dt = datetime.strptime(raw, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue

    # default
    return datetime.now(timezone.utc)
