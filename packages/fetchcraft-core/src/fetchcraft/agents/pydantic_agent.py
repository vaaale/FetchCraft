"""
ReAct agent implementation using pydantic-ai.
"""
from datetime import datetime
from typing import Optional, Any, List, Union, Dict, AsyncIterable

from pydantic_ai import AgentStreamEvent, PartStartEvent, PartDeltaEvent, TextPartDelta, \
    ThinkingPartDelta, ToolCallPartDelta, FunctionToolCallEvent, FunctionToolResultEvent, FinalResultEvent, ModelSettings, ModelMessagesTypeAdapter, ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel

from .output_formatters import DefaultOutputFormatter
from .base import BaseAgent
from .memory import Memory
from .model import AgentResponse, ChatMessage, CitationContainer
from .output_formatters import OutputFormatter
from .utils import to_chat_message, to_pydantic_ai_messages
from ..base_logging import configure_logging
from ..mixins import ObjectNodeMixin
import json
from pydantic_core import to_jsonable_python

from ..pydantic_ai_utils import openai_history_to_pydantic_ai

try:
    from pydantic_ai import Agent, RunContext, Tool
    from pydantic_ai.models import Model

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    Agent = None
    RunContext = None
    Model = None

from pydantic import ConfigDict, PrivateAttr, TypeAdapter

configure_logging("root")

# SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on retrieved information.
# SYSTEM_PROMPT = """You are a helpful AI assistant that uses reasoning and multi-step thinking to answer questions based on retrieved information and conversation history.
# Knowledge cutoff: 31.12.2023
# Today's date: {today}
#
# When answering:
# 1. Read the conversation history and the users query carefully to understand what the user wants
# 2. Create a plan for solving the task step-by-step
# 3. Use the provided tools to find relevant information or execute actions. You can call the tools as often as you need.
# 4. Base your answer on the retrieved documents ONLY! Never use prior knowledge!
# 5. If the provided documents do not contain the information refine your query and try again.
# 5. When reciting facts, always include a citation to the parsing of the information. Always add citations to your answer using a Markdown link. Example: 'See [<document_title>](<document_number>)'
# (The document_number must always be an integer and is the number following 'Document ' in the citations)
# """

# SYSTEM_PROMPT = """You are a helpful AI assistant that uses reasoning and multi-step thinking to answer questions based on retrieved information and conversation history.
#
# Lets think Step-By-Step:
# 1. Break the problem down into manageable steps using the provided tools
# 2. Read the conversation history and the users query carefully to understand what the user wants
# 3. Read the provided context and take note of the document numbers
# 4. Write your answer, using the provided context ONLY.
# 5. Always add citations to your answer.
# 6. Always write citations using Markdown format. Example: 'See [<document_title>](<document_number>)'
#
# (The document_number must always be an integer and is the number following 'Document ' in the context)
# """

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions **only** using the retrieved documents (“context”) and the conversation history. You must always include properly formatted citations pointing to the documents you used.

---

## High-level behavior

- Use reasoning and multi-step thinking **internally** to understand the query and the context.
- Base your answer **only** on the provided documents and conversation history.  
  - **Do not** use external knowledge or make up facts.
- If the documents do not contain enough information to answer, say so explicitly.

---

## Workflow (internal, not shown to the user)

1. **Understand the request**  
   - Read the entire conversation and the latest user query carefully.
   - Identify what the user is asking and what type of answer is expected (definition, explanation, summary, comparison, etc.).

2. **Inspect the context**
   - Read all provided documents.
   - Identify any information that can be used to enrich the answer.
   - Note each document’s:
     - **Document number**: the integer after the word `Document` (e.g., `Document 3` → number = `3`).
     - **Document title**: the title/text given for that document.

3. **Plan your answer**
   - Decide which documents are relevant to the question.
   - Collect the specific parts of those documents that support your answer.

4. **Write the answer**
   - Answer the user’s question clearly and directly.
   - Elaborate on your answer if you deem it valuable to the user.
   - Use your own words to summarize and synthesize the information from the documents.
   - **Do not** copy large chunks of text verbatim unless absolutely necessary.

5. **Add citations (mandatory)**
   - Every time you use information from a document, add a citation.
   - At minimum, **each paragraph that contains factual information must include at least one citation**.
   - Place citations at the end of the sentence or paragraph they support.

---

## Citation rules (very important)

1. **Citation format (Markdown)**
   - Always write citations in this exact format (including brackets and parentheses):
     - `See [<document_title>](<document_number>)`
   - Example:
     - `See `  
   - If you cite multiple documents for the same sentence or paragraph, separate them with spaces or commas:
     - `See , `

2. **Document number**
   - Must **always** be an integer.
   - It is the number that follows the word `Document` in the context.
     - Example: `Document 7: Installation Manual` → document number is `7`.

3. **Document title**
   - Use the title given in the context for that document.
   - If no clear title is provided, use a generic label like `Document 7` as the title:
     - `See `

4. **No fabricated citations**
   - Never invent document numbers or titles.
   - Only cite documents that actually appear in the provided context.
   - If you cannot find a relevant document to support a claim, **do not make the claim**.

5. **Consistency**
   - Do not mix other citation styles (such as `[3]`, `(Doc 3)`, etc.).
   - Always use the Markdown link style described above.

---

## Insufficient information

- If the context does not contain enough information to answer the question:
  - Say that you cannot fully answer based on the provided documents, and
  - Optionally summarize any **partially relevant** information you did find, with citations.

- Example:
  - “I cannot fully answer your question based on the provided documents. They do not discuss X in detail. However, they do mention Y. See ”

---

## Never forget citations

- **Do not provide any final answer without citations.**
- If you realize an answer has no citations, you must add them before finishing.
- It is better to give a short answer with correct citations than a long answer without them.

---

Only output the final answer to the user (with citations). Do **not** describe your internal steps unless the user explicitly asks for them.
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


class PydanticAgent(BaseAgent, ObjectNodeMixin):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _tools: List[Tool] = PrivateAttr(default=None)
    model: Union[str, Model] = "openai:gpt-4-turbo"
    agent_kwargs: Dict = {}
    _memory: Memory = PrivateAttr(default=None)
    system_prompt: Optional[str] = SYSTEM_PROMPT

    """
    ReAct (Reasoning and Acting) agent that uses a retriever to answer questions.
    
    The agent can search for relevant information using the retriever
    and then reason about the results to provide an answer.
    """

    def __init__(
            self,
            model: Union[str, Model] = "openai:gpt-4",
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

    def to_json(self) -> Dict[str, Any]:
        return {
            "agent_kwargs": self.agent_kwargs,
        }

    @classmethod
    def create(
            cls,
            model: Union[str, Any] = "openai:gpt-4",
            tools: List[Tool] | None = None,
            system_prompt: Optional[str] = SYSTEM_PROMPT,
            output_formatter: Optional[OutputFormatter] = None,
            **agent_kwargs
    ) -> 'PydanticAgent':
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
            output_formatter=output_formatter or DefaultOutputFormatter(),
            agent_kwargs=agent_kwargs,
        )

    async def query(self, question: str, messages: Optional[List[ChatMessage]] = None, **kwargs) -> AgentResponse:
        """
        Query the agent with a question.
        
        Args:
            question: The question to ask
            **kwargs: Additional parameters for the agent
            
        Returns:
            The agent's response
            :param question:
            :param messages:
        """
        model = OpenAIChatModel(
            self.model,
            settings=ModelSettings(
                temperature=0.8,
                top_p=0.95,
                extra_body={"top_k": 20, "min_p": 0}
            ),
        )
        agent = Agent(
            # model=self.model,
            model=model,
            system_prompt=self.system_prompt.format(today=datetime.now().strftime("%d.%m.%Y")),
            tools=self._tools,
            deps_type=CitationContainer,
            **self.agent_kwargs,
        )


        def answer(ctx: RunContext, answer: str) -> str:
            """Call this function when to return your final answer.
            The input should be your final answer."""
            return answer

        # chat_history = messages if messages else []
        citations = CitationContainer()
        self._memory = Memory()

        chat_history = []
        if messages:
            message_dicts = [m.model_dump() for m in messages]
            chat_history = openai_history_to_pydantic_ai(message_dicts)

        result = await agent.run(
            user_prompt=question,
            message_history=chat_history,
            output_type=answer,
            deps=citations,
            **kwargs
        )
        response = result.output

        answer = self.output_formatter.format(response, citations)

        self._memory.add_memory(
            ChatMessage.user_message(question),
            ChatMessage.assistant_message(answer),
            citations.citations
        )

        used_citations = citations.citations
        return AgentResponse(
            query=ChatMessage.user_message(question),
            response=ChatMessage.assistant_message(answer),
            citations=used_citations,
            all_citations=citations.all_citations
        )

    def __repr__(self) -> str:
        return f"ReActAgent(model={self.model})"
