"""
Tests for evaluation module.
"""
import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from fetchcraft import (
    DatasetGenerator,
    EvaluationDataset,
    RetrieverEvaluator,
    EvaluationMetrics,
    Node,
    Chunk,
    DocumentNode,
    NodeWithScore,
)
from fetchcraft.evaluation.dataset_generator import QuestionContextPair
from fetchcraft.evaluation.evaluator import QueryResult


class TestQuestionAnswerPair:
    """Test QuestionAnswerPair model."""
    
    def test_create_qa_pair(self):
        qa = QuestionContextPair(
            question="What is Python?",
            node_id="node-123",
            context="Python is a programming language..."
        )
        
        assert qa.question == "What is Python?"
        assert qa.node_id == "node-123"
        assert qa.metadata == {}


class TestEvaluationDataset:
    """Test EvaluationDataset model."""
    
    def test_create_dataset(self):
        qa_pairs = [
            QuestionContextPair(
                question="Q1",
                node_id="n1",
                context="C1"
            ),
            QuestionContextPair(
                question="Q2",
                node_id="n2",
                context="C2"
            )
        ]
        
        dataset = EvaluationDataset(qa_pairs=qa_pairs)
        assert len(dataset) == 2
        assert dataset.qa_pairs[0].question == "Q1"
    
    def test_save_and_load(self, tmp_path):
        qa_pairs = [
            QuestionContextPair(
                question="What is AI?",
                node_id="node-1",
                context="AI is artificial intelligence"
            )
        ]
        
        dataset = EvaluationDataset(
            qa_pairs=qa_pairs,
            metadata={"version": "1.0"}
        )
        
        # Save
        file_path = tmp_path / "dataset.json"
        dataset.save(str(file_path))
        
        # Load
        loaded = EvaluationDataset.load(str(file_path))
        assert len(loaded) == 1
        assert loaded.qa_pairs[0].question == "What is AI?"
        assert loaded.metadata["version"] == "1.0"


class TestDatasetGenerator:
    """Test DatasetGenerator."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock OpenAI client."""
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
        from openai.types.completion_usage import CompletionUsage
        
        client = AsyncMock()
        # Create a proper ChatCompletion response with tool calls
        tool_call = ChatCompletionMessageToolCall(
            id="call_123",
            type="function",
            function=Function(
                name="final_result",
                arguments='{"questions": ["Question 1?", "Question 2?", "Question 3?"]}'
            )
        )
        message = ChatCompletionMessage(
            role="assistant",
            content=None,
            tool_calls=[tool_call]
        )
        choice = Choice(
            index=0,
            message=message,
            finish_reason="tool_calls"
        )
        response = ChatCompletion(
            id="chatcmpl-123",
            model="gpt-4",
            object="chat.completion",
            created=1234567890,
            choices=[choice],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )
        client.chat.completions.create.return_value = response
        return client
    
    @pytest.fixture
    def mock_doc_store(self):
        """Mock document store."""
        store = AsyncMock()
        
        # Mock document with children
        doc = DocumentNode(
            id="doc-1",
            text="Document text",
            children_ids=["node-1", "node-2"]
        )
        store.get_document.return_value = doc
        store.list_documents.return_value = [doc]
        
        return store
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store."""
        store = AsyncMock()
        
        # Mock nodes
        node1 = Chunk(
            id="node-1",
            text="Node 1 text for testing question generation",
            metadata={"source": "test.txt"}
        )
        node2 = Chunk(
            id="node-2",
            text="Node 2 text for testing question generation",
            metadata={"source": "test.txt"}
        )
        
        store.get_document.side_effect = lambda node_id, **kwargs: {
            "node-1": node1,
            "node-2": node2
        }.get(node_id)
        
        return store
    
    @pytest.mark.asyncio
    async def test_generate_questions_for_node(
        self,
        mock_client,
        mock_doc_store,
        mock_vector_store
    ):
        # Create model with mock client
        model = OpenAIChatModel(
            "gpt-4",
            provider=OpenAIProvider(
                openai_client=mock_client
            )
        )
        generator = DatasetGenerator(model=model)
        
        node = Chunk(id="test", text="Test text" * 10)
        questions = await generator._generate_questions_for_node(node, 3)
        
        assert len(questions) == 3
        assert all(isinstance(q, str) for q in questions)
    
    @pytest.mark.asyncio
    async def test_get_top_level_nodes(
        self,
        mock_client,
        mock_doc_store,
        mock_vector_store
    ):
        model = OpenAIChatModel(
            os.environ.get("OPENAI_MODEL", "gpt-4-turbo"),
            provider=OpenAIProvider(
                openai_client=mock_client
            )
        )
        generator = DatasetGenerator(
            model=model,
        )
        
        nodes = await generator._get_top_level_nodes_for_document("doc-1", mock_doc_store, mock_vector_store)
        
        assert len(nodes) == 2
        assert nodes[0].id == "node-1"
        assert nodes[1].id == "node-2"
    
    @pytest.mark.asyncio
    async def test_generate_from_specific_nodes(
        self,
        mock_client,
        mock_doc_store,
        mock_vector_store
    ):
        # Create model with mock client
        model = OpenAIChatModel(
            "gpt-4",
            provider=OpenAIProvider(
                openai_client=mock_client
            )
        )
        generator = DatasetGenerator(model=model)
        
        dataset = await generator.generate_from_specific_nodes(
            node_ids=["node-1"],
            vector_store=mock_vector_store,
            questions_per_node=2,
            show_progress=False
        )
        
        assert len(dataset) > 0
        assert all(isinstance(qa, QuestionContextPair) for qa in dataset.qa_pairs)


class TestRetrieverEvaluator:
    """Test RetrieverEvaluator."""
    
    @pytest.fixture
    def mock_retriever(self):
        """Mock retriever."""
        retriever = AsyncMock()
        retriever.top_k = 5
        return retriever
    
    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset for testing."""
        qa_pairs = [
            QuestionContextPair(
                question="Q1",
                node_id="node-1",
                context="Context 1"
            ),
            QuestionContextPair(
                question="Q2",
                node_id="node-2",
                context="Context 2"
            ),
            QuestionContextPair(
                question="Q3",
                node_id="node-3",
                context="Context 3"
            )
        ]
        return EvaluationDataset(qa_pairs=qa_pairs)
    
    @pytest.mark.asyncio
    async def test_evaluate_single_query_hit(self, mock_retriever):
        """Test evaluation when correct node is retrieved."""
        # Setup mock retriever response
        mock_retriever.aretrieve.return_value = [
            NodeWithScore(
                node=Chunk(id="node-1", text="Text 1"),
                score=0.95
            ),
            NodeWithScore(
                node=Chunk(id="node-2", text="Text 2"),
                score=0.85
            )
        ]
        
        evaluator = RetrieverEvaluator(retriever=mock_retriever)
        qa_pair = QuestionContextPair(
            question="Test question",
            node_id="node-1",
            context="Test context"
        )
        
        result = await evaluator._evaluate_single_query(qa_pair)
        
        assert result.hit is True
        assert result.rank == 1
        assert result.reciprocal_rank == 1.0
    
    @pytest.mark.asyncio
    async def test_evaluate_single_query_miss(self, mock_retriever):
        """Test evaluation when correct node is not retrieved."""
        mock_retriever.aretrieve.return_value = [
            NodeWithScore(
                node=Chunk(id="node-2", text="Text 2"),
                score=0.85
            )
        ]
        
        evaluator = RetrieverEvaluator(retriever=mock_retriever)
        qa_pair = QuestionContextPair(
            question="Test question",
            node_id="node-1",
            context="Test context"
        )
        
        result = await evaluator._evaluate_single_query(qa_pair)
        
        assert result.hit is False
        assert result.rank is None
        assert result.reciprocal_rank == 0.0
    
    @pytest.mark.asyncio
    async def test_evaluate_dataset(self, mock_retriever, sample_dataset):
        """Test full dataset evaluation."""
        # Mock retriever to return different results
        mock_retriever.aretrieve.side_effect = [
            # Q1: Hit at rank 1
            [
                NodeWithScore(node=Chunk(id="node-1", text="T1"), score=0.9),
                NodeWithScore(node=Chunk(id="other", text="T2"), score=0.8)
            ],
            # Q2: Hit at rank 2
            [
                NodeWithScore(node=Chunk(id="other", text="T1"), score=0.9),
                NodeWithScore(node=Chunk(id="node-2", text="T2"), score=0.8)
            ],
            # Q3: Miss
            [
                NodeWithScore(node=Chunk(id="other1", text="T1"), score=0.9),
                NodeWithScore(node=Chunk(id="other2", text="T2"), score=0.8)
            ]
        ]
        
        evaluator = RetrieverEvaluator(retriever=mock_retriever)
        metrics = await evaluator.evaluate(
            dataset=sample_dataset,
            show_progress=False
        )
        
        assert metrics.total_queries == 3
        assert metrics.hits == 2
        assert metrics.hit_rate == 2/3
        assert metrics.recall_at_k == 2/3
        
        # MRR = (1/1 + 1/2 + 0) / 3 = 0.5
        assert abs(metrics.mrr - 0.5) < 0.001
    
    def test_calculate_precision_at_k(self, mock_retriever):
        """Test precision@k calculation."""
        evaluator = RetrieverEvaluator(retriever=mock_retriever)
        
        # Setup mock results
        evaluator.results = [
            QueryResult(
                question="Q1",
                expected_node_id="n1",
                retrieved_node_ids=["n1", "n2"],
                retrieved_scores=[0.9, 0.8],
                hit=True,
                rank=1,
                reciprocal_rank=1.0
            ),
            QueryResult(
                question="Q2",
                expected_node_id="n2",
                retrieved_node_ids=["n3", "n4"],
                retrieved_scores=[0.9, 0.8],
                hit=False,
                rank=None,
                reciprocal_rank=0.0
            )
        ]
        
        # Precision@1 = 1/2 (only first query has hit at rank 1)
        precision = evaluator._calculate_precision_at_k(1)
        assert abs(precision - 0.5) < 0.001
    
    def test_get_failed_queries(self, mock_retriever):
        """Test getting failed queries."""
        evaluator = RetrieverEvaluator(retriever=mock_retriever)
        
        evaluator.results = [
            QueryResult(
                question="Q1",
                expected_node_id="n1",
                retrieved_node_ids=["n1"],
                retrieved_scores=[0.9],
                hit=True,
                rank=1,
                reciprocal_rank=1.0
            ),
            QueryResult(
                question="Q2",
                expected_node_id="n2",
                retrieved_node_ids=["n3"],
                retrieved_scores=[0.9],
                hit=False,
                rank=None,
                reciprocal_rank=0.0
            )
        ]
        
        failed = evaluator.get_failed_queries()
        assert len(failed) == 1
        assert failed[0].question == "Q2"
    
    def test_save_results(self, mock_retriever, tmp_path):
        """Test saving results to file."""
        evaluator = RetrieverEvaluator(retriever=mock_retriever)
        
        evaluator.results = [
            QueryResult(
                question="Q1",
                expected_node_id="n1",
                retrieved_node_ids=["n1"],
                retrieved_scores=[0.9],
                hit=True,
                rank=1,
                reciprocal_rank=1.0
            )
        ]
        
        file_path = tmp_path / "results.json"
        evaluator.save_results(str(file_path))
        
        assert file_path.exists()
        
        import json
        with open(file_path) as f:
            data = json.load(f)
        
        assert len(data['results']) == 1
        assert data['summary']['total_queries'] == 1
        assert data['summary']['hits'] == 1


class TestEvaluationMetrics:
    """Test EvaluationMetrics model."""
    
    def test_create_metrics(self):
        metrics = EvaluationMetrics(
            total_queries=100,
            hits=80,
            hit_rate=0.8,
            mrr=0.75,
            precision_at_1=0.6,
            precision_at_3=0.7,
            precision_at_5=0.75,
            precision_at_k=0.8,
            recall_at_1=0.6,
            recall_at_3=0.7,
            recall_at_5=0.75,
            recall_at_k=0.8,
            ndcg_at_k=0.78,
            average_rank=1.5,
            rank_distribution={1: 60, 2: 15, 3: 5},
            k=5
        )
        
        assert metrics.total_queries == 100
        assert metrics.hit_rate == 0.8
        assert metrics.mrr == 0.75
    
    def test_metrics_string_representation(self):
        metrics = EvaluationMetrics(
            total_queries=10,
            hits=8,
            hit_rate=0.8,
            mrr=0.75,
            precision_at_1=0.6,
            precision_at_3=0.7,
            precision_at_5=0.75,
            precision_at_k=0.8,
            recall_at_1=0.6,
            recall_at_3=0.7,
            recall_at_5=0.75,
            recall_at_k=0.8,
            ndcg_at_k=0.78,
            average_rank=1.5,
            rank_distribution={1: 6, 2: 2},
            k=5
        )
        
        metrics_str = str(metrics)
        assert "Total Queries: 10" in metrics_str
        assert "Hit Rate@5: 0.8000" in metrics_str
        assert "MRR" in metrics_str
    
    def test_metrics_to_dict(self):
        metrics = EvaluationMetrics(
            total_queries=10,
            hits=8,
            hit_rate=0.8,
            mrr=0.75,
            precision_at_1=0.6,
            precision_at_3=0.7,
            precision_at_5=0.75,
            precision_at_k=0.8,
            recall_at_1=0.6,
            recall_at_3=0.7,
            recall_at_5=0.75,
            recall_at_k=0.8,
            ndcg_at_k=0.78,
            average_rank=1.5,
            rank_distribution={1: 6, 2: 2},
            k=5
        )
        
        metrics_dict = metrics.to_dict()
        assert metrics_dict['total_queries'] == 10
        assert metrics_dict['hit_rate'] == 0.8
        assert metrics_dict['mrr'] == 0.75
