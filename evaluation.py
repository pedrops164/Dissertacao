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
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from llm_system import LLMSystem
from llm_client import call_llm_assessment
from abc import ABC, abstractmethod # Import ABC and abstractmethod
from bioasq import load_bioasq_yesno_questions
import concurrent.futures # Import for ThreadPoolExecutor

# Define evaluation result data structure
@dataclass
class EvaluationEntry:
    """A single evaluation entry for a specific question"""
    question: str
    ground_truth: str # Can be string for open-domain, or specific type for others
    response: str
    # Metrics specific to this single entry (e.g., latency for this query, or if it was correct for binary)
    metrics: Dict[str, float]
    
    def to_dict(self):
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "response": self.response,
            "metrics": self.metrics
        }

# --- Abstract Base Class for Evaluation Results ---
@dataclass
class BaseEvaluationResult(ABC):
    """Abstract base class for collecting evaluation entries for a system."""
    system_name: str
    entries: List[EvaluationEntry]
    # This dictionary will store the aggregated/summary metrics calculated by subclasses.
    summary_metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure summary metrics are calculated after initialization if entries exist."""
        if self.entries and not self.summary_metrics: # Calculate if not pre-filled and entries exist
            self.calculate_summary_metrics()

    @abstractmethod
    def calculate_summary_metrics(self) -> None:
        """
        Abstract method to calculate specific summary metrics for the evaluation type.
        Subclasses must implement this method to populate self.summary_metrics.
        """
        pass
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary for serialization."""
        return {
            "system_name": self.system_name,
            "evaluation_type": self.__class__.__name__, # Add type for clarity when loading
            "entries": [entry.to_dict() for entry in self.entries],
            "summary_metrics": self.summary_metrics
        }

# --- Concrete Subclass for Open-Domain Question Answering ---
@dataclass
class OpenDomainEvaluationResult(BaseEvaluationResult):
    """Evaluation results for open-domain question answering systems."""

    def calculate_summary_metrics(self) -> None:
        """Calculate average metrics across all entries for open-domain QA."""
        if not self.entries:
            self.summary_metrics = {}
            return
            
        # Gather all unique metric keys from individual entries' metrics dictionaries
        # These are expected to be things like latency, token_count, factual_correctness, etc.
        all_individual_metric_keys = set()
        for entry in self.entries:
            all_individual_metric_keys.update(entry.metrics.keys())
            
        calculated_summary_metrics = {}
        for metric_key in all_individual_metric_keys:
            values = [entry.metrics.get(metric_key) for entry in self.entries if entry.metrics.get(metric_key) is not None]
            if values:
                # Ensure all values are numeric for averaging
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    calculated_summary_metrics[f"avg_{metric_key}"] = sum(numeric_values) / len(numeric_values)
                else:
                    # Handle cases where a metric key exists but has no numeric values (e.g. all None or non-numeric)
                    calculated_summary_metrics[f"avg_{metric_key}"] = 0.0 # Or None, or skip
            else:
                calculated_summary_metrics[f"avg_{metric_key}"] = 0.0 # Or None, or skip
        
        self.summary_metrics = calculated_summary_metrics

# --- Concrete Subclass for Binary Classification (Yes/No) Questions ---
@dataclass
class BinaryClassificationEvaluationResult(BaseEvaluationResult):
    """
    Evaluation results for systems answering binary (e.g., Yes/No) questions.
    Assumes `ground_truth` in EvaluationEntry is the expected binary label (e.g., "yes", "no").
    Assumes `response` in EvaluationEntry is the system's predicted binary label.
    The `metrics` in each EvaluationEntry should contain an `is_correct` field (1.0 for correct, 0.0 for incorrect).
    """

    def _normalize_label(self, label: str) -> Optional[str]:
        """Helper to normalize labels to lower case for comparison."""
        if isinstance(label, str):
            return label.strip(' .').lower()
        return None

    def calculate_summary_metrics(self) -> None:
        """Calculate accuracy and other relevant metrics for binary classification."""
        if not self.entries:
            self.summary_metrics = {}
            return

        correct_predictions = 0
        total_predictions = len(self.entries)
        
        for entry in self.entries:
            # Ensure ground_truth and response are strings for normalization
            gt_normalized = self._normalize_label(str(entry.ground_truth))
            resp_normalized = self._normalize_label(entry.response)

            # Primary metric: Accuracy based on 'is_correct' if available in entry.metrics
            # This allows the per-entry metric calculation to handle complex correctness logic
            if gt_normalized == resp_normalized:
                correct_predictions += 1
        
        accuracy = (correct_predictions / total_predictions) if total_predictions > 0 else -1
        
        self.summary_metrics = {
            "accuracy": accuracy,
            "total_questions": float(total_predictions),
            "correct_predictions": float(correct_predictions),
        }
    
# Make RAGEvaluator an Abstract Base Class
class RAGEvaluator(ABC):
    """
    Abstract base class for evaluating and comparing RAG systems.
    Concrete evaluators must implement evaluate_system and compare_systems.
    """

    def __init__(self, n_workers):
        self.results = {}
        self.n_workers = n_workers

    @abstractmethod
    def evaluate_system(self, system: LLMSystem, queries: List[Any]) -> BaseEvaluationResult: # Return single EvaluationResult
        """
        Evaluate a RAG system on a set of queries.
        This method must be implemented by subclasses.
        
        Args:
            system: The LLMSystem object to evaluate.
            queries: List of queries. Can be strings or tuples (query, ground_truth).
            
        Returns:
            An EvaluationResult object for the evaluated system.
        """
        pass

    def compare_systems(self) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple systems across all metrics
        
        Args:
            system_names: List of system names to compare (None for all)
            
        Returns:
            Dictionary with average scores for each system and metric
        """
        if not self.results:
            return {}
            
        system_names = list(self.results.keys())
            
        comparison = {}
        
        for system_name in system_names:
            system_result = self.results.get(system_name, None)
            
            if not system_result:
                continue
                
            comparison[system_name] = system_result.summary_metrics
            
        return comparison
    
    def save_results(self, output_file: str):
        """Save evaluation results to file"""
        with open(output_file, 'w') as f:
            json.dump([r.to_dict() for r in list(self.results.values())], f, indent=2)

class RAGEvaluatorYesNoQuestion(RAGEvaluator):
    """Framework for evaluating and comparing RAG systems on yes/no questions"""
        
    def evaluate_system(self, system: LLMSystem, queries: List[Tuple[str, str]]) -> BinaryClassificationEvaluationResult:
        """
        Evaluate a RAG system on a set of yes/no queries in parallel.
        
        Args:
            system: The LLMSystem object to evaluate.
            queries: List of (query_string, ground_truth_string) tuples. 
                     Ground truth should be 'yes' or 'no'.
                     
        Returns:
            A BinaryClassificationEvaluationResult object.
        """
        system_name = system.system_name
        entries: List[EvaluationEntry] = []

        if not queries:
            raise ValueError("No queries provided to evaluate_system")
        
        if not (isinstance(queries, list) and all(isinstance(q, tuple) and len(q) == 2 for q in queries)):
            raise ValueError(f"Error: Queries for {self.__class__.__name__} must be a list of (query, ground_truth) tuples.")

        print(f"\n--- Evaluating system (Yes/No): {system_name} with {len(queries)} queries using max {self.n_workers} workers ---")
        
        start_overall_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks to the executor
            future_to_query = {
                executor.submit(self._process_single_query, system, query_tuple): query_tuple 
                for query_tuple in queries
            }
            
            processed_count = 0
            for future in concurrent.futures.as_completed(future_to_query):
                query_tuple_original = future_to_query[future]
                try:
                    entry = future.result()
                    entries.append(entry)
                except Exception as exc:
                    print(f"Query '{query_tuple_original[0][:50]}...' generated an exception: {exc}")
                finally:
                    processed_count += 1
                    if processed_count % 10 == 0 or processed_count == len(queries):
                        print(f"  Processed {processed_count}/{len(queries)} queries...")

        overall_latency = time.time() - start_overall_time
        print(f"--- Overall evaluation for {system_name} took {overall_latency:.2f} seconds ---")

        # Create result object with all collected entries
        result = BinaryClassificationEvaluationResult(
            system_name=system_name,
            entries=entries,
        )
        
        self.results[system_name] = result # Store in the inherited results dictionary
        return result
    
    @staticmethod
    def get_yesno_query(query: str) -> str:
        """
        Convert a query to a yes/no question format
        
        Args:
            query: Original query string
            
        Returns:
            Yes/No formatted question
        """
        return f"{query}\nRespond only with 'yes' or 'no'."

    def _get_system_response(self, system: LLMSystem, query: str) -> Tuple[str, int]:
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

    def _process_single_query(self, system: LLMSystem, query_tuple: Tuple[str, str]) -> EvaluationEntry:
        """
        Processes a single query: formats it, gets system response, calculates metrics.
        This function is executed by each worker thread.
        """
        query, ground_truth = query_tuple
        
        start_time = time.time()
        yesno_formatted_query = RAGEvaluatorYesNoQuestion.get_yesno_query(query)
        
        response, token_count = self._get_system_response(system, yesno_formatted_query)
        
        latency = time.time() - start_time

        # Normalize response and ground_truth for consistent "yes" / "no"
        normalized_response = response.strip(" .").lower()
        normalized_ground_truth = ground_truth.strip(" .").lower()

        # This can be used by BinaryClassificationEvaluationResult
        metrics = {
            "latency": latency,
            "token_count": float(token_count),
        }
            
        return EvaluationEntry(
            question=query,
            ground_truth=normalized_ground_truth, # Store normalized ground_truth
            response=normalized_response,       # Store normalized response
            metrics=metrics
        )


class RAGEvaluatorOpenQuestion(RAGEvaluator):
    """Framework for evaluating and comparing RAG systems"""
    
    def __init__(self, n_workers, use_llm_judge=False):
        """
        Initialize the evaluator
        
        Args:
            judge_model: Model to use for LLM-as-judge evaluations
        """
        super().__init__(n_workers=n_workers)
        self.use_llm_judge = use_llm_judge
        
    def evaluate_system(self, system: LLMSystem, queries: List[Any]) -> OpenDomainEvaluationResult:
        """
        Evaluate a RAG system on a set of queries
        
        Args:
            system_name: Name of the system being evaluated
            system: The actual system object with query method
            queries: List of query strings
            
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
            if self.use_llm_judge:
                content_scores = self._llm_judge_evaluation(query, ground_truth, response)
                metrics.update(content_scores)
            
            entry = EvaluationEntry(
                question=query,
                ground_truth=ground_truth,
                response=response,
                metrics=metrics
            )
            
            entries.append(entry)
           
        # Create result with all entries
        result = OpenDomainEvaluationResult(
            system_name=system_name,
            entries=entries
        )
         
        self.results[system_name] = result
        return result

    def _get_system_response(self, system: LLMSystem, query: str) -> Tuple[str, List[str]]:
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
        response, _ = call_llm_assessment(prompt=prompt, max_tokens=500)
        return response

    
    def _parse_judge_scores(self, judge_response: str) -> Dict[str, float]:
        """Parse scores from judge response"""
        # Extract JSON from response
        json_str = judge_response.split("```json")[1].split("```")[0].strip()
        scores = json.loads(json_str)
        return scores

if __name__ == "__main__":
    # Initialize evaluator
    from config import config
    n_workers = config.n_workers
    evaluator = RAGEvaluatorYesNoQuestion(n_workers=n_workers)

    # Import queries
    from llm_system import NoRAGSystem, SelfRAGSystem, FusionRAGSystem, CRAGRAGSystem

    bioasq_queries = load_bioasq_yesno_questions()

    no_rag_system = NoRAGSystem("No-RAG System")
    self_rag_system = SelfRAGSystem("Self-RAG System")
    #fusion_rag_system = FusionRAGSystem("Fusion-RAG System")
    #crag_rag_system = CRAGRAGSystem("CRAG-RAG System")

    print("Evaluating systems...")

    # Evaluate basic LLM
    basic_llm_results = evaluator.evaluate_system(
        no_rag_system,
        bioasq_queries
    )

    print("Basic LLM evaluation complete.")

    # Evaluate Self-RAG
    self_rag_results = evaluator.evaluate_system(
        self_rag_system,
        bioasq_queries
    )

    print("Self-RAG evaluation complete.")

    # Compare systems
    comparison = evaluator.compare_systems()
    print(json.dumps(comparison, indent=2))

    # Save results
    evaluator.save_results("rag_evaluation_results.json")