from typing import List

from pydantic_ai import ModelMessage, ModelRequest, UserPromptPart, ModelResponse, TextPart, UnexpectedModelBehavior
from pydantic_core import to_jsonable_python

from fetchcraft.agents.base import ChatMessage


def to_chat_messages(messages: List[ModelMessage]) -> List[ChatMessage]:
    return [to_chat_message(m) for m in messages]


def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            assert isinstance(first_part.content, str)
            msg = {
                'role': 'user',
                'timestamp': first_part.timestamp.isoformat(),
                'content': first_part.content,
            }
            return ChatMessage.model_validate(msg)
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            msg = {
                'role': 'model',
                'timestamp': m.timestamp.isoformat(),
                'content': first_part.content,
            }
            return ChatMessage.model_validate(msg)
    raise UnexpectedModelBehavior(f'Unexpected message type for chat app: {m}')


def to_pydantic_ai_messages(chat_messages: List[ChatMessage]):
    for m in chat_messages:
        if m.role == "user":
            yield ModelRequest(parts=[UserPromptPart(content=m.content)])
        elif m.role == "assistant":
            yield ModelResponse(parts=[TextPart(content=m.content)])
        elif m.role == "tool":
            yield ModelResponse(parts=[TextPart(content=m.content)])
        elif m.role == "system":
            yield ModelResponse(parts=[TextPart(content=m.content)])
        else:
            raise ValueError(f'Unexpected role: {m.role}')


