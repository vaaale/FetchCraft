from typing import Sequence, Any

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, CompletionResponseAsyncGen, CompletionResponse, ChatResponseAsyncGen, CompletionResponseGen, ChatResponseGen
from llama_index.llms.openai import OpenAI as LOpenAI

from fetchcraft.llama_index.llm.outputparser import ThinkingOutputParser


class OpenAI(LOpenAI):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        parser = ThinkingOutputParser()
        response = super().chat(messages, **kwargs)
        new_content = parser.parse(response.message.content)
        response.message.content = new_content
        return response

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        parser = ThinkingOutputParser()
        response = await super().achat(messages, **kwargs)
        new_content = parser.parse(response.message.content)
        response.message.content = new_content
        return response

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        super_generator = await super().astream_chat(messages, **kwargs)

        async def gen() -> ChatResponseAsyncGen:
            parser = ThinkingOutputParser()
            async for item in super_generator:
                yield item
            item.message.content = parser.parse(item.message.content)
        return gen()

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        super_generator = super().stream_chat(messages, **kwargs)

        parser = ThinkingOutputParser()
        for item in super_generator:
            yield item
        item.message.content = parser.parse(item.message.content)




