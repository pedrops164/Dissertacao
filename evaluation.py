"""
evaluation.py - Framework for evaluating LLM and RAG systems
This file provides functions and classes to evaluate and compare different RAG implementations:
- Basic LLM
- Self-RAG
- Fusion RAG
- CRAG

It includes automated metrics and LLM-as-judge evaluation approaches.
"""

import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from llm_system import LLMSystem
from llm_client import call_llm_assessment

# Define evaluation result data structure
@dataclass
class EvaluationEntry:
    """A single evaluation entry for a specific question"""
    question: str
    ground_truth: Optional[str]
    response: str
    metrics: Dict[str, float]
    
    def to_dict(self):
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "response": self.response,
            "metrics": self.metrics
        }

@dataclass
class EvaluationResult:
    """Collection of evaluation entries for a system"""
    system_name: str
    entries: List[EvaluationEntry]
    avg_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        """Calculate average metrics across all entries if not provided"""
        if self.avg_metrics is None:
            self.calculate_avg_metrics()
    
    def calculate_avg_metrics(self):
        """Calculate average metrics across all entries"""
        if not self.entries:
            self.avg_metrics = {}
            return
            
        # Get all metric keys
        all_metrics = set()
        for entry in self.entries:
            all_metrics.update(entry.metrics.keys())
            
        # Calculate average for each metric
        self.avg_metrics = {}
        for metric in all_metrics:
            values = [entry.metrics.get(metric, 0.0) for entry in self.entries if metric in entry.metrics]
            if values:
                self.avg_metrics[metric] = sum(values) / len(values)
            else:
                self.avg_metrics[metric] = 0.0
            
    def to_dict(self):
        """Convert the result to a dictionary for serialization"""
        return {
            "system_name": self.system_name,
            "entries": [entry.to_dict() for entry in self.entries],
            "avg_metrics": self.avg_metrics
        }

class RAGEvaluator:
    """Framework for evaluating and comparing RAG systems"""
    
    def __init__(self, use_llm_judge=False):
        """
        Initialize the evaluator
        
        Args:
            judge_model: Model to use for LLM-as-judge evaluations
        """
        self.use_llm_judge = use_llm_judge
        self.results = {}
        
    def evaluate_system(self, system: LLMSystem, queries: List[Any], content_metrics: bool = True) -> List[EvaluationResult]:
        """
        Evaluate a RAG system on a set of queries
        
        Args:
            system_name: Name of the system being evaluated
            system: The actual system object with query method
            queries: List of query strings
            content_metrics: Whether to calculate content quality metrics
            
        Returns:
            List of EvaluationResult objects
        """
        system_name = system.system_name
        entries = []

        if queries is None or len(queries) == 0:
            print(f"No queries provided for evaluation of {system_name}.")
            return entries
        
        if isinstance(queries[0], str):
            # If queries are strings, convert to list of tuples with ground truth
            queries = [(q, None) for q in queries]
        
        for query, ground_truth in queries:
            # Measure efficiency metrics
            start_time = time.time()
            
            # Get response from system
            response, token_count = self._get_system_response(system, query)
            
            # Calculate efficiency metrics
            latency = time.time() - start_time
            
            # Initialize metrics dictionary
            metrics = {
                "latency": latency,
                "token_count": token_count
            }
                
            # Calculate content quality metrics if applicable
            if content_metrics and self.use_llm_judge:
                content_scores = self._calculate_content_metrics(query, ground_truth, response)
                metrics.update(content_scores)
            
            entry = EvaluationEntry(
                question=query,
                ground_truth=ground_truth,
                response=response,
                metrics=metrics
            )
            
            entries.append(entry)
           
        # Create result with all entries
        result = EvaluationResult(
            system_name=system_name,
            entries=entries
        )
         
        self.results[system_name] = result
        return result

    def _get_system_response(self, system, query: str) -> Tuple[str, List[str]]:
        """
        Get response from a system, handling different implementations
        
        Returns:
            Tuple of (response_text, retrieved_documents)
        """
        try:
            # Make a single call to the system
            response, tokens_used = system.query(query)

            return response, tokens_used
                
        except Exception as e:
            print(f"Error getting response from system: {e}")
            return "", 0
        
    def _calculate_content_metrics(self, query: str, ground_truth: str, response: str) -> Dict[str, float]:
        """Calculate content quality metrics using LLM-as-judge"""
        scores = self._llm_judge_evaluation(query, ground_truth, response)
        return scores
        
    def _llm_judge_evaluation(self, query: str, ground_truth: str, response: str) -> Dict[str, float]:
        """
        Use an LLM to evaluate the quality of the response
        
        Returns:
            Dictionary of scores for different quality dimensions
        """
        # Create prompt for LLM judge
        prompt = self._create_judge_prompt(query, ground_truth, response)
        
        # Get evaluation from judge model
        # This implementation depends on your specific judge model
        judge_response = self._get_judge_response(prompt)
        
        # Parse scores from judge response
        try:
            scores = self._parse_judge_scores(judge_response)
            return scores
        except Exception as e:
            print(f"Error parsing judge scores: {e}")
            return {
                "factual_correctness": 0.0,
                "answer_relevance": 0.0,
                "hallucination_score": 0.0,
                "completeness": 0.0,
                "coherence": 0.0
            }
    
    def _create_judge_prompt(self, query: str, ground_truth: str, response: str) -> str:
        """Create evaluation prompt for judge model"""
        # Construct context from retrieved documents
        
        prompt = f"""Evaluate the following response to a query. Rate each aspect on a scale of 1-10.

Query: {query}

Ground Truth: {ground_truth}

Response to Evaluate:
{response}

Please evaluate the response on the following criteria:
1. Factual Correctness (1-10): Is the information in the response factually accurate according to the retrieved context?
2. Answer Relevance (1-10): How relevant is the response to the query?
3. Hallucination (1-10): Does the response contain information not supported by the retrieved context? (10 = no hallucination, 1 = completely hallucinated)
4. Completeness (1-10): Does the response address all aspects of the query?
5. Coherence (1-10): Is the response well-structured, logical, and easy to understand?

Provide your ratings in the following JSON format:
```json
{{
  "factual_correctness": 0,
  "answer_relevance": 0,
  "hallucination_score": 0,
  "completeness": 0,
  "coherence": 0
}}
```
"""
        return prompt
    
    def _get_judge_response(self, prompt: str) -> str:
        """Get evaluation from judge model"""
        return call_llm_assessment(prompt=prompt, max_tokens=100, temperature=0.0)

    
    def _parse_judge_scores(self, judge_response: str) -> Dict[str, float]:
        """Parse scores from judge response"""
        # Extract JSON from response
        json_str = judge_response.split("```json")[1].split("```")[0].strip()
        scores = json.loads(json_str)
        return scores
    
    def _count_tokens(self, query: str, response: str) -> int:
        """Count tokens in query and response"""
        # This is a simplified implementation
        # For real implementation, use tokenizer from your LLM
        return len(query.split()) + len(response.split())
    
    def compare_systems(self, system_names: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple systems across all metrics
        
        Args:
            system_names: List of system names to compare (None for all)
            
        Returns:
            Dictionary with average scores for each system and metric
        """
        if not self.results:
            return {}
            
        if system_names is None:
            system_names = list(self.results.keys())
            
        comparison = {}
        
        for system_name in system_names:
            system_result = self.results.get(system_name, None)
            
            if not system_result:
                continue
                
            comparison[system_name] = system_result.avg_metrics
            
        return comparison
    
    def save_results(self, output_file: str):
        """Save evaluation results to file"""
        with open(output_file, 'w') as f:
            json.dump([r.to_dict() for r in list(self.results.values())], f, indent=2)
    
    def load_results(self, input_file: str):
        """Load evaluation results from file"""
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        self.results = []
        for item in data:
            result = EvaluationResult(
                system_name=item["system_name"],
                question=item["question"],
                response=item["response"],
                retrieved_docs=item["retrieved_docs"],
                metrics=item["metrics"]
            )
            self.results.append(result)

if __name__ == "__main__":
    # Initialize evaluator
    evaluator = RAGEvaluator(use_llm_judge=True)
    # no judge model for now
    #evaluator = RAGEvaluator(judge_model=your_judge_model)

    # Import queries
    from queries import factual_queries_en, reasoning_queries_en
    from llm_system import NoRAGSystem, SelfRAGSystem, FusionRAGSystem, CRAGRAGSystem

    no_rag_system = NoRAGSystem("No-RAG System")
    self_rag_system = SelfRAGSystem("Self-RAG System")
    #fusion_rag_system = FusionRAGSystem("Fusion-RAG System")
    #crag_rag_system = CRAGRAGSystem("CRAG-RAG System")

    print("Evaluating systems...")

    # Evaluate basic LLM
    basic_llm_results = evaluator.evaluate_system(
        no_rag_system,
        factual_queries_en
    )

    print("Basic LLM evaluation complete.")

    # Evaluate Self-RAG
    self_rag_results = evaluator.evaluate_system(
        self_rag_system,
        factual_queries_en
    )

    print("Self-RAG evaluation complete.")

    # Compare systems
    comparison = evaluator.compare_systems()
    print(json.dumps(comparison, indent=2))

    # Save results
    evaluator.save_results("rag_evaluation_results.json")