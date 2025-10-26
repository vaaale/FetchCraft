"""
ReAct agent implementation using pydantic-ai.
"""
from datetime import datetime
from typing import Optional, Any, List, Union, Dict, AsyncIterable

from pydantic_ai import AgentRunResultEvent, AgentStreamEvent, PartStartEvent, PartDeltaEvent, TextPartDelta, \
    ThinkingPartDelta, ToolCallPartDelta, FunctionToolCallEvent, FunctionToolResultEvent, FinalResultEvent

from ..logging import configure_logging

try:
    from pydantic_ai import Agent, RunContext, Tool
    from pydantic_ai.models import Model

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    Agent = None
    RunContext = None
    Model = None

from pydantic import ConfigDict, PrivateAttr
import re
from .base import BaseAgent, AgentResponse, CitationContainer

configure_logging("root")

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on retrieved information.
Knowledge cutoff: 31.12.2023
Today's date: {today}

When answering:
1. Understand the task the user wants you to do
2. Create a plan for solving the task step-by-step
3. Use the provided tools to find relevant information or execute actions. You can call the tools as often as you need.
4. Base your answer on the retrieved documents ONLY! Never use prior knowledge!
5. If the provided documents do not contain the information refine your query and try again.
5. When reciting facts, always include a citation to the source of the information. Always add citations to your answer using a Markdown link. Example: 'See [<document_title>](<document_number>)'
(The document_number must always be an integer and is the number following 'Document ' in the citations)
"""

#3. If the information is not in the retrieved documents, say so

output_messages: list[str] = []


async def handle_event(event: AgentStreamEvent):
    if isinstance(event, PartStartEvent):
        output_messages.append(f'[Request] Starting part {event.index}: {event.part!r}')
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, TextPartDelta):
            output_messages.append(f'[Request] Part {event.index} text delta: {event.delta.content_delta!r}')
        elif isinstance(event.delta, ThinkingPartDelta):
            output_messages.append(f'[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}')
        elif isinstance(event.delta, ToolCallPartDelta):
            output_messages.append(f'[Request] Part {event.index} args delta: {event.delta.args_delta}')
    elif isinstance(event, FunctionToolCallEvent):
        output_messages.append(
            f'[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})'
        )
    elif isinstance(event, FunctionToolResultEvent):
        output_messages.append(f'[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}')
    elif isinstance(event, FinalResultEvent):
        output_messages.append(f'[Result] The model starting producing a final result (tool_name={event.tool_name})')


async def event_stream_handler(
        ctx: RunContext,
        event_stream: AsyncIterable[AgentStreamEvent],
):
    async for event in event_stream:
        await handle_event(event)


class ReActAgent(BaseAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _tools: List[Tool] = PrivateAttr(default=None)
    model: Union[str, Any] = "openai:gpt-4"
    agent_kwargs: Dict = {}

    """
    ReAct (Reasoning and Acting) agent that uses a retriever to answer questions.
    
    The agent can search for relevant information using the retriever
    and then reason about the results to provide an answer.
    """

    def __init__(
            self,
            model: Union[str, Any] = "openai:gpt-4",
            tools: List[Tool] = None,
            **agent_kwargs
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            retriever_tool: RetrieverTool to use for searching information
            model: LLM model to use (default: "openai:gpt-4")
            system_prompt: Custom system prompt
            **agent_kwargs: Additional arguments for pydantic-ai Agent
        """
        if tools is None:
            tools = []

        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError(
                "pydantic-ai is required for ReActAgent. "
                "Install it with: pip install pydantic-ai"
            )

        # Register the retriever tool
        # Create the pydantic-ai agent
        super().__init__(
            model=model,
            **agent_kwargs,
        )
        self._tools = tools

    @classmethod
    def create(
            cls,
            model: Union[str, Any] = "openai:gpt-4",
            tools: List[Tool] | None = None,
            system_prompt: Optional[str] = None,
            **agent_kwargs
    ) -> 'ReActAgent':
        """
        Create a ReActAgent instance.
        
        Args:
            model: LLM model to use (default: "openai:gpt-4")
            system_prompt: Custom system prompt
            **agent_kwargs: Additional arguments for pydantic-ai Agent
            
        Returns:
            ReActAgent instance
            :param tools: List of tools to use for searching information
        """
        return cls(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            agent_kwargs=agent_kwargs,
        )

    async def query(self, question: str, **kwargs) -> AgentResponse:
        """
        Query the agent with a question.
        
        Args:
            question: The question to ask
            **kwargs: Additional parameters for the agent
            
        Returns:
            The agent's response
        """

        agent = Agent(
            model=self.model,
            system_prompt=SYSTEM_PROMPT.format(today=datetime.now().strftime("%d.%m.%Y")),
            tools=self._tools,
            deps_type=CitationContainer,
            **self.agent_kwargs,
        )


        def final_output(ctx: RunContext, answer: str) -> str:
            """Call this function when to return your final answer.
            The input should be your final answer."""
            return answer

        chat_history = []
        citations = CitationContainer()
        result = await agent.run(
            user_prompt=question,
            message_history=chat_history,
            output_type=final_output,
            deps=citations,
            **kwargs
        )
        response = result.output
        answer = self._post_process_citations(response, citations)
        return AgentResponse(response=answer, citations=citations.citations, all_citations=citations.all_citations)

    async def aquery(self, question: str, **kwargs) -> AgentResponse:
        """
        Async version of query (alias for consistency).
        
        Args:
            question: The question to ask
            **kwargs: Additional parameters
            
        Returns:
            The agent's response
        """
        return await self.query(question, **kwargs)

    def __repr__(self) -> str:
        return f"ReActAgent(model={self.model})"
