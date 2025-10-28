"""
Dataset generator for creating evaluation datasets from documents.
"""

import random
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from tqdm import tqdm
import logging

from ..document_store.base import DocumentStore
from ..vector_store.base import VectorStore
from ..node import Node, DocumentNode, Chunk

logger = logging.getLogger(__name__)


class QuestionAnswerPair(BaseModel):
    """A question-answer pair for evaluation."""
    question: str = Field(description="The generated question")
    node_id: str = Field(description="ID of the node that should answer this question")
    context: str = Field(description="The text context from which the question was generated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EvaluationDataset(BaseModel):
    """A dataset for evaluating retriever performance."""
    qa_pairs: List[QuestionAnswerPair] = Field(description="List of question-answer pairs")
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


class DatasetGenerator:
    """
    Generates evaluation datasets for testing retriever performance.
    
    The generator samples documents from a DocumentStore and VectorStore,
    then uses an LLM to generate questions that can be answered by each node.
    
    Example:
        ```python
        from openai import AsyncOpenAI
        from fetchcraft.evaluation import DatasetGenerator
        
        client = AsyncOpenAI(api_key="your-key")
        generator = DatasetGenerator(
            client=client,
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
    
    def __init__(
        self,
        client: Any,
        document_store: DocumentStore,
        vector_store: VectorStore,
        model: str = "gpt-4",
        index_id: Optional[str] = None
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
        self.client = client
        self.document_store = document_store
        self.vector_store = vector_store
        self.model = model
        self.index_id = index_id
    
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
        prompt = f"""Given the following text, generate {num_questions} diverse questions that can be answered using ONLY the information in this text.

The questions should:
1. Be natural and varied in style
2. Range from simple factual to more complex analytical questions
3. Be answerable using only the provided text
4. Cover different aspects of the text
5. Be clear and unambiguous

Text:
{node.text}

Generate exactly {num_questions} questions, one per line, without numbering or bullet points."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates high-quality questions for evaluation datasets."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            questions_text = response.choices[0].message.content
            if questions_text is None:
                logger.warning(f"No questions generated for node {node.id[:8]}")
                return []
            
            # Parse questions (one per line)
            questions = [
                q.strip() 
                for q in questions_text.strip().split('\n') 
                if q.strip() and not q.strip().startswith('#')
            ]
            
            # Filter out empty questions and numbering
            questions = [
                q.lstrip('0123456789.-) ').strip() 
                for q in questions 
                if len(q.strip()) > 10
            ]
            
            return questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Error generating questions for node {node.id[:8]}: {e}")
            return []
    
    async def _get_top_level_nodes_for_document(
        self,
        doc_id: str
    ) -> List[Node]:
        """
        Get all top-level nodes (direct children) for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of top-level nodes
        """
        # Get the document
        document = await self.document_store.get_document(doc_id)
        if not document:
            logger.warning(f"Document {doc_id} not found in document store")
            return []
        
        # Get all children IDs
        if not hasattr(document, 'children_ids') or not document.children_ids:
            logger.warning(f"Document {doc_id} has no children")
            return []
        
        # Fetch all child nodes from vector store
        nodes = []
        for child_id in document.children_ids:
            node = await self.vector_store.get_document(child_id, index_id=self.index_id)
            if node:
                nodes.append(node)
        
        return nodes
    
    async def generate_dataset(
        self,
        num_documents: int,
        questions_per_node: int = 3,
        max_nodes_per_document: Optional[int] = None,
        show_progress: bool = True
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
        all_documents = await self.document_store.list_documents(
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
        
        qa_pairs: List[QuestionAnswerPair] = []
        
        # Process each document
        doc_iterator = tqdm(sampled_documents, desc="Processing documents") if show_progress else sampled_documents
        
        for document in doc_iterator:
            # Get top-level nodes for this document
            nodes = await self._get_top_level_nodes_for_document(document.id)
            
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
                    qa_pair = QuestionAnswerPair(
                        question=question,
                        node_id=node.id,
                        context=node.text[:500] + "..." if len(node.text) > 500 else node.text,
                        metadata={
                            'doc_id': document.id,
                            'source': node.metadata.get('source', 'unknown')
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
                'model': self.model
            }
        )
    
    async def generate_from_specific_nodes(
        self,
        node_ids: List[str],
        questions_per_node: int = 3,
        show_progress: bool = True
    ) -> EvaluationDataset:
        """
        Generate dataset from specific node IDs.
        
        Args:
            node_ids: List of node IDs to generate questions for
            questions_per_node: Number of questions per node
            show_progress: Whether to show progress
            
        Returns:
            EvaluationDataset containing question-answer pairs
        """
        logger.info(f"Generating questions for {len(node_ids)} specific nodes")
        
        qa_pairs: List[QuestionAnswerPair] = []
        node_iterator = tqdm(node_ids, desc="Processing nodes") if show_progress else node_ids
        
        for node_id in node_iterator:
            # Get node from vector store
            node = await self.vector_store.get_document(node_id, index_id=self.index_id)
            if not node:
                logger.warning(f"Node {node_id} not found")
                continue
            
            # Generate questions
            questions = await self._generate_questions_for_node(node, questions_per_node)
            
            for question in questions:
                qa_pair = QuestionAnswerPair(
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
                'model': self.model
            }
        )
