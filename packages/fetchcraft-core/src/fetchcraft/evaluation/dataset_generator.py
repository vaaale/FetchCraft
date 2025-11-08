"""
Dataset generator for creating evaluation datasets from documents.
"""

import random
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from pydantic_ai import Agent
from pydantic_ai.models import Model
from tqdm import tqdm
import logging

from ..document_store.base import DocumentStore
from ..vector_store.base import VectorStore
from ..node import Node, DocumentNode, Chunk

logger = logging.getLogger(__name__)

# USER_PROMPT = (
#     "Given the following text, generate {num_questions} diverse questions that can be answered using ONLY the information in this text.\n"
#     "\n"
#     "The questions should:\n"
#     "1. Be natural and varied in style\n"
#     "2. Range from simple factual to more complex analytical questions\n"
#     "3. Be answerable using only the provided text\n"
#     "4. Cover different aspects of the text\n"
#     "5. Be clear and unambiguous\n"
#     "\n"
#     "Text:\n"
#     "{text}\n"
#     "\n"
#     "Generate exactly {num_questions} questions, one per line, without numbering or bullet points."
# )

SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality questions for evaluation datasets."

USER_PROMPT = """You are helping evaluate a retrieval system.
Given a parsing passage and a desired count N, write N user-style questions that a person might ask without knowing the passage exists, yet which are answerable using only the passage.

## Requirements

1) Standalone & generic
   - Questions must make sense out of context and must not imply the existence of a specific document.
   - Avoid: “According to the passage…”, “In this document…”, “the article”, “the text”, “the author”, “this report”.
   - Do: include necessary entities/timeframes (e.g., “on 9/11”, “in the Mars 2020 mission”), so the question is unambiguous.

2) Answerable from the passage
   - Every question must have a specific, factual answer supported by the passage (dates, names, causes, definitions, comparisons, counts, processes, caveats).
   - Do not ask for opinions or information not present.

3) Sufficient specificity
   - Include key entities, qualifiers, and constraints needed to retrieve the right chunk (event name, location, time period, actor, metric).

4) Natural user phrasing
   - Write like a real user query (concise, varied).
   - Avoid copying full sentences; prefer paraphrases and natural wording.

5) Diversity
   - Mix question types: what/when/where/who/why/how, comparisons, enumerations, cause/effect, definitions, numeric specifics.
   - Vary difficulty and length (short to medium).

6) No ambiguity or pronouns without antecedents
   - Replace “it/they/this” with the concrete entity.
   
SOURCE PASSAGE:
{text}

Generate {num_questions} questions.
"""

class QuestionContextPair(BaseModel):
    """A question-answer pair for evaluation."""
    question: str = Field(description="The generated question")
    node_id: str = Field(description="ID of the node that should answer this question")
    context: str = Field(description="The text context from which the question was generated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Questions(BaseModel):
    """Questions that can be answered using ONLY the information in this text."""
    questions: List[str] = Field(description="List of questions")


class EvaluationDataset(BaseModel):
    """A dataset for evaluating retriever performance."""
    qa_pairs: List[QuestionContextPair] = Field(description="List of question-answer pairs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")

    def __len__(self) -> int:
        return len(self.qa_pairs)

    def save(self, filepath: str) -> None:
        """Save dataset to a JSON file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'EvaluationDataset':
        """Load dataset from a JSON file."""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


class DatasetGenerator(BaseModel):
    """
    Generates evaluation datasets for testing retriever performance.
    
    The generator samples documents from a DocumentStore and VectorStore,
    then uses an LLM to generate questions that can be answered by each node.
    
    Example:
        ```python
        from openai import AsyncOpenAI
        from fetchcraft.evaluation import DatasetGenerator
        
        model = "openai:gpt-5"
        generator = DatasetGenerator(
            model=model,
            document_store=doc_store,
            vector_store=vector_store
        )
        
        dataset = await generator.generate_dataset(
            num_documents=10,
            questions_per_node=3
        )
        dataset.save("eval_dataset.json")
        ```
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _model: str | Model = PrivateAttr()

    def __init__(
            self,
            model: str | Model,
    ):
        """
        Initialize the dataset generator.
        
        Args:
            client: OpenAI AsyncClient instance for generating questions
            document_store: DocumentStore to sample documents from
            vector_store: VectorStore to retrieve nodes from
            model: OpenAI model to use for question generation
            index_id: Optional index identifier to filter documents
        """
        super().__init__()
        self._model = model

    async def _generate_questions_for_node(
            self,
            node: Node,
            num_questions: int
    ) -> List[str]:
        """
        Generate questions that can be answered by the given node.
        
        Args:
            node: The node to generate questions for
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        agent = Agent(
            model=self._model,
            system_prompt=SYSTEM_PROMPT,
            output_type=Questions,
            retries=3
        )

        response = await agent.run(
            user_prompt=USER_PROMPT.format(
                num_questions=num_questions,
                text=node.text
            )
        )

        questions = response.output.questions
        return questions

    async def _get_top_level_nodes_for_document(
        self,
        doc_id: str,
        document_store: DocumentStore,
        vector_store: VectorStore,
        index_id: Optional[str] = None
    ) -> List[Node]:
        """
        Get all top-level nodes (direct children) for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of top-level nodes
        """
        # Get the document
        document = await document_store.get_document(doc_id)
        if not document:
            logger.warning(f"Document {doc_id} not found in document store")
            return []

        # Get all children IDs
        if not document.children_ids:
            logger.warning(f"Document {doc_id} has no children")
            return []

        # Fetch all child nodes from vector store
        nodes = []
        for child_id in document.children_ids:
            node = await vector_store.get_node(child_id, index_id=index_id)
            if node:
                nodes.append(node)

        return nodes

    async def generate_dataset(
            self,
            num_documents: int,
            document_store: DocumentStore,
            vector_store: VectorStore,
            index_id: Optional[str] = None,
            questions_per_node: int = 3,
            max_nodes_per_document: int = 5,
            show_progress: bool = True,
    ) -> EvaluationDataset:
        """
        Generate an evaluation dataset.
        
        Args:
            num_documents: Number of documents to sample
            questions_per_node: Number of questions to generate per node
            max_nodes_per_document: Maximum nodes to use per document (None = all)
            show_progress: Whether to show progress bars
            
        Returns:
            EvaluationDataset containing question-answer pairs
        """
        logger.info(f"Generating evaluation dataset from {num_documents} documents")

        # Sample documents from document store
        all_documents = await document_store.list_documents(
            limit=1000,  # Get a large pool to sample from
            # filters={'document': True} if hasattr(self.document_store, 'list_documents') else None
        )

        if len(all_documents) < num_documents:
            logger.warning(
                f"Only {len(all_documents)} documents available, "
                f"requested {num_documents}"
            )
            num_documents = len(all_documents)

        # Randomly sample documents
        sampled_documents = random.sample(all_documents, num_documents)

        qa_pairs: List[QuestionContextPair] = []

        # Process each document
        doc_iterator = tqdm(sampled_documents, desc="Processing documents") if show_progress else sampled_documents

        for document in doc_iterator:
            # Get top-level nodes for this document
            nodes = await self._get_top_level_nodes_for_document(document.id, document_store, vector_store, index_id)

            if not nodes:
                logger.warning(f"No nodes found for document {document.id[:8]}")
                continue

            # Limit nodes if specified
            if max_nodes_per_document and len(nodes) > max_nodes_per_document:
                nodes = random.sample(nodes, max_nodes_per_document)

            # Generate questions for each node
            node_iterator = tqdm(nodes, desc=f"  Nodes", leave=False) if show_progress else nodes

            for node in node_iterator:
                questions = await self._generate_questions_for_node(node, questions_per_node)

                for question in questions:
                    qa_pair = QuestionContextPair(
                        question=question,
                        node_id=node.id,
                        context=node.text[:500] + "..." if len(node.text) > 500 else node.text,
                        metadata={
                            'doc_id': document.id,
                            'parsing': node.metadata.get('parsing', 'unknown')
                        }
                    )
                    qa_pairs.append(qa_pair)

        logger.info(f"Generated {len(qa_pairs)} question-answer pairs")

        return EvaluationDataset(
            qa_pairs=qa_pairs,
            metadata={
                'num_documents': num_documents,
                'questions_per_node': questions_per_node,
                'total_pairs': len(qa_pairs),
            }
        )

    async def generate_from_specific_nodes(
            self,
            node_ids: List[str],
            vector_store: VectorStore,
            questions_per_node: int = 3,
            index_id: Optional[str] = None,
            show_progress: bool = True
    ) -> EvaluationDataset:
        """
        Generate dataset from specific node IDs.
        
        Args:
            node_ids: List of node IDs to generate questions for
            vector_store: VectorStore to retrieve nodes from
            questions_per_node: Number of questions per node
            index_id: Optional index identifier
            show_progress: Whether to show progress
            
        Returns:
            EvaluationDataset containing question-answer pairs
        """
        logger.info(f"Generating questions for {len(node_ids)} specific nodes")

        qa_pairs: List[QuestionContextPair] = []
        node_iterator = tqdm(node_ids, desc="Processing nodes") if show_progress else node_ids

        for node_id in node_iterator:
            # Get node from vector store
            node = await vector_store.get_node(node_id, index_id=index_id)
            if not node:
                logger.warning(f"Node {node_id} not found")
                continue

            # Generate questions
            questions = await self._generate_questions_for_node(node, questions_per_node)

            for question in questions:
                qa_pair = QuestionContextPair(
                    question=question,
                    node_id=node.id,
                    context=node.text[:500] + "..." if len(node.text) > 500 else node.text,
                    metadata=node.metadata.copy()
                )
                qa_pairs.append(qa_pair)

        logger.info(f"Generated {len(qa_pairs)} question-answer pairs")

        return EvaluationDataset(
            qa_pairs=qa_pairs,
            metadata={
                'num_nodes': len(node_ids),
                'questions_per_node': questions_per_node,
                'total_pairs': len(qa_pairs),
                'model': self._model
            }
        )
