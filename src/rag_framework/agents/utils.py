from typing import List

from pydantic_ai import ModelMessage, ModelRequest, UserPromptPart, ModelResponse, TextPart, UnexpectedModelBehavior

from rag_framework.agents.base import ChatMessage


def to_chat_messages(messages: List[ModelMessage]) -> List[ChatMessage]:
    return [to_chat_message(m) for m in messages]


def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            assert isinstance(first_part.content, str)
            return {
                'role': 'user',
                'timestamp': first_part.timestamp.isoformat(),
                'content': first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            return {
                'role': 'model',
                'timestamp': m.timestamp.isoformat(),
                'content': first_part.content,
            }
    raise UnexpectedModelBehavior(f'Unexpected message type for chat app: {m}')
