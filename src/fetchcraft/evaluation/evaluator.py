"""
Evaluator for testing retriever performance using evaluation datasets.
"""

import asyncio
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from tqdm import tqdm
import logging

from ..retriever.base import Retriever
from ..node import NodeWithScore
from .dataset_generator import EvaluationDataset, QuestionContextPair

logger = logging.getLogger(__name__)


class QueryResult(BaseModel):
    """Result of a single query evaluation."""
    question: str
    expected_node_id: str
    retrieved_node_ids: List[str]
    retrieved_scores: List[float]
    hit: bool = Field(description="Whether the expected node was retrieved")
    rank: Optional[int] = Field(default=None, description="Rank of the expected node (1-indexed), None if not found")
    reciprocal_rank: float = Field(default=0.0, description="Reciprocal rank (1/rank), 0 if not found")


class EvaluationMetrics(BaseModel):
    """Comprehensive metrics for retriever evaluation."""
    
    # Basic metrics
    total_queries: int = Field(description="Total number of queries evaluated")
    
    # Hit Rate / Recall@k
    hit_rate: float = Field(description="Proportion of queries where the correct node was retrieved")
    hits: int = Field(description="Number of queries with hits")
    
    # Mean Reciprocal Rank (MRR)
    mrr: float = Field(description="Mean Reciprocal Rank across all queries")
    
    # Precision@k
    precision_at_1: float = Field(description="Precision at rank 1")
    precision_at_3: float = Field(description="Precision at rank 3")
    precision_at_5: float = Field(description="Precision at rank 5")
    precision_at_k: float = Field(description="Precision at k (where k is the retriever's top_k)")
    
    # Recall@k
    recall_at_1: float = Field(description="Recall at rank 1")
    recall_at_3: float = Field(description="Recall at rank 3") 
    recall_at_5: float = Field(description="Recall at rank 5")
    recall_at_k: float = Field(description="Recall at k (where k is the retriever's top_k)")
    
    # NDCG (Normalized Discounted Cumulative Gain)
    ndcg_at_k: float = Field(description="NDCG at k")
    
    # Average metrics
    average_rank: float = Field(description="Average rank of correct nodes (for hits only)")
    
    # Distribution
    rank_distribution: Dict[int, int] = Field(
        default_factory=dict,
        description="Distribution of ranks where correct nodes were found"
    )
    
    # Additional info
    k: int = Field(description="Value of k used for retrieval")
    
    def __str__(self) -> str:
        """Pretty print the metrics."""
        return f"""
Retrieval Evaluation Metrics (k={self.k})
{'='*50}
Total Queries: {self.total_queries}

Hit Rate & Recall:
  Hit Rate@{self.k}: {self.hit_rate:.4f} ({self.hits}/{self.total_queries})
  Recall@1:  {self.recall_at_1:.4f}
  Recall@3:  {self.recall_at_3:.4f}
  Recall@5:  {self.recall_at_5:.4f}
  Recall@{self.k}:  {self.recall_at_k:.4f}

Precision:
  Precision@1: {self.precision_at_1:.4f}
  Precision@3: {self.precision_at_3:.4f}
  Precision@5: {self.precision_at_5:.4f}
  Precision@{self.k}: {self.precision_at_k:.4f}

Ranking Quality:
  MRR (Mean Reciprocal Rank): {self.mrr:.4f}
  NDCG@{self.k}: {self.ndcg_at_k:.4f}
  Average Rank (when found): {self.average_rank:.2f}

Rank Distribution:
{self._format_rank_distribution()}
{'='*50}
"""
    
    def _format_rank_distribution(self) -> str:
        """Format the rank distribution for display."""
        if not self.rank_distribution:
            return "  No hits"
        
        lines = []
        for rank in sorted(self.rank_distribution.keys()):
            count = self.rank_distribution[rank]
            pct = (count / self.total_queries) * 100
            bar = '█' * int(pct / 2)  # Scale bar to reasonable size
            lines.append(f"  Rank {rank}: {count:4d} ({pct:5.1f}%) {bar}")
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return self.model_dump()


class RetrieverEvaluator:
    """
    Evaluates retriever performance using an evaluation dataset.
    
    Calculates comprehensive metrics including:
    - Hit Rate / Recall@k
    - Mean Reciprocal Rank (MRR)
    - Precision@k
    - NDCG (Normalized Discounted Cumulative Gain)
    - Average rank
    
    Example:
        ```python
        from fetchcraft.evaluation import RetrieverEvaluator, EvaluationDataset
        
        dataset = EvaluationDataset.load("eval_dataset.json")
        evaluator = RetrieverEvaluator(retriever=my_retriever)
        
        metrics = await evaluator.evaluate(dataset)
        print(metrics)
        
        # Save detailed results
        evaluator.save_results("results.json")
        ```
    """
    
    def __init__(self, retriever: Retriever):
        """
        Initialize the evaluator.
        
        Args:
            retriever: The retriever to evaluate
        """
        self.retriever = retriever
        self.results: List[QueryResult] = []
    
    async def _evaluate_single_query(
        self,
        qa_pair: QuestionContextPair
    ) -> QueryResult:
        """
        Evaluate a single query.
        
        Args:
            qa_pair: Question-answer pair to evaluate
            
        Returns:
            QueryResult with evaluation details
        """
        # Retrieve using the question
        retrieved: List[NodeWithScore] = await self.retriever.aretrieve(qa_pair.question)
        
        # Extract node IDs and scores
        retrieved_node_ids = [node.node.id for node in retrieved]
        retrieved_scores = [node.score for node in retrieved]
        
        # Check if expected node was retrieved
        hit = qa_pair.node_id in retrieved_node_ids
        rank = None
        reciprocal_rank = 0.0
        
        if hit:
            # Find the rank (1-indexed)
            rank = retrieved_node_ids.index(qa_pair.node_id) + 1
            reciprocal_rank = 1.0 / rank
        
        return QueryResult(
            question=qa_pair.question,
            expected_node_id=qa_pair.node_id,
            retrieved_node_ids=retrieved_node_ids,
            retrieved_scores=retrieved_scores,
            hit=hit,
            rank=rank,
            reciprocal_rank=reciprocal_rank
        )
    
    def _calculate_precision_at_k(self, k: int) -> float:
        """Calculate Precision@k."""
        if not self.results:
            return 0.0
        
        precision_sum = 0.0
        for result in self.results:
            # For each query, precision@k is 1/k if the correct doc is in top-k, else 0
            if result.rank is not None and result.rank <= k:
                precision_sum += 1.0 / k
        
        return precision_sum / len(self.results)
    
    def _calculate_recall_at_k(self, k: int) -> float:
        """Calculate Recall@k."""
        if not self.results:
            return 0.0
        
        # For single relevant document per query, recall@k = hit_rate@k
        hits_at_k = sum(
            1 for result in self.results 
            if result.rank is not None and result.rank <= k
        )
        return hits_at_k / len(self.results)
    
    def _calculate_ndcg_at_k(self, k: int) -> float:
        """
        Calculate NDCG@k (Normalized Discounted Cumulative Gain).
        
        For binary relevance (0 or 1), this simplifies to:
        DCG@k = sum(rel_i / log2(i + 1)) for i in 1..k
        IDCG@k = 1 / log2(2) = 1 (for single relevant document)
        """
        if not self.results:
            return 0.0
        
        ndcg_sum = 0.0
        for result in self.results:
            if result.rank is not None and result.rank <= k:
                # Relevance is 1 for the correct document, 0 for others
                import math
                dcg = 1.0 / math.log2(result.rank + 1)
                idcg = 1.0 / math.log2(2)  # Perfect ranking (rel doc at position 1)
                ndcg = dcg / idcg
                ndcg_sum += ndcg
            # If not found in top-k, NDCG = 0
        
        return ndcg_sum / len(self.results)
    
    def _calculate_rank_distribution(self) -> Dict[int, int]:
        """Calculate the distribution of ranks."""
        distribution: Dict[int, int] = {}
        for result in self.results:
            if result.rank is not None:
                distribution[result.rank] = distribution.get(result.rank, 0) + 1
        return distribution
    
    async def evaluate(
        self,
        dataset: EvaluationDataset,
        show_progress: bool = True,
        max_queries: Optional[int] = None
    ) -> EvaluationMetrics:
        """
        Evaluate the retriever on the dataset.
        
        Args:
            dataset: EvaluationDataset to evaluate on
            show_progress: Whether to show progress bar
            max_queries: Maximum number of queries to evaluate (None = all)
            
        Returns:
            EvaluationMetrics with comprehensive performance metrics
        """
        logger.info(f"Evaluating retriever on {len(dataset)} queries")
        
        qa_pairs = dataset.qa_pairs
        if max_queries:
            qa_pairs = qa_pairs[:max_queries]
        
        # Evaluate all queries
        iterator = tqdm(qa_pairs, desc="Evaluating queries") if show_progress else qa_pairs
        
        self.results = []
        for qa_pair in iterator:
            result = await self._evaluate_single_query(qa_pair)
            self.results.append(result)
        
        # Calculate metrics
        k = self.retriever.top_k
        total = len(self.results)
        hits = sum(1 for r in self.results if r.hit)
        
        # MRR
        mrr = sum(r.reciprocal_rank for r in self.results) / total if total > 0 else 0.0
        
        # Average rank (for hits only)
        ranks = [r.rank for r in self.results if r.rank is not None]
        avg_rank = sum(ranks) / len(ranks) if ranks else 0.0
        
        # Calculate precision and recall at various k values
        metrics = EvaluationMetrics(
            total_queries=total,
            hits=hits,
            hit_rate=hits / total if total > 0 else 0.0,
            mrr=mrr,
            precision_at_1=self._calculate_precision_at_k(1),
            precision_at_3=self._calculate_precision_at_k(3),
            precision_at_5=self._calculate_precision_at_k(5),
            precision_at_k=self._calculate_precision_at_k(k),
            recall_at_1=self._calculate_recall_at_k(1),
            recall_at_3=self._calculate_recall_at_k(3),
            recall_at_5=self._calculate_recall_at_k(5),
            recall_at_k=self._calculate_recall_at_k(k),
            ndcg_at_k=self._calculate_ndcg_at_k(k),
            average_rank=avg_rank,
            rank_distribution=self._calculate_rank_distribution(),
            k=k
        )
        
        logger.info(f"Evaluation complete: Hit Rate = {metrics.hit_rate:.4f}, MRR = {metrics.mrr:.4f}")
        
        return metrics
    
    def save_results(self, filepath: str) -> None:
        """
        Save detailed evaluation results to a JSON file.
        
        Args:
            filepath: Path to save the results
        """
        import json
        
        data = {
            'results': [r.model_dump() for r in self.results],
            'summary': {
                'total_queries': len(self.results),
                'hits': sum(1 for r in self.results if r.hit),
                'hit_rate': sum(1 for r in self.results if r.hit) / len(self.results) if self.results else 0.0
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {filepath}")
    
    def get_failed_queries(self) -> List[QueryResult]:
        """Get all queries where retrieval failed (no hit)."""
        return [r for r in self.results if not r.hit]
    
    def get_queries_by_rank(self, rank: int) -> List[QueryResult]:
        """Get all queries where the correct node was found at a specific rank."""
        return [r for r in self.results if r.rank == rank]
    
    async def evaluate_with_different_k(
        self,
        dataset: EvaluationDataset,
        k_values: List[int],
        show_progress: bool = True
    ) -> Dict[int, EvaluationMetrics]:
        """
        Evaluate the retriever with different k values.
        
        Args:
            dataset: EvaluationDataset to evaluate on
            k_values: List of k values to test
            show_progress: Whether to show progress
            
        Returns:
            Dictionary mapping k values to their metrics
        """
        results = {}
        
        for k in k_values:
            logger.info(f"Evaluating with k={k}")
            original_k = self.retriever.top_k
            self.retriever.top_k = k
            
            metrics = await self.evaluate(dataset, show_progress=show_progress)
            results[k] = metrics
            
            # Restore original k
            self.retriever.top_k = original_k
        
        return results
