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
from abc import ABC, abstractmethod # Import ABC and abstractmethod
from bioasq import load_bioasq_yesno_questions, load_bioasq_open_questions
import concurrent.futures # Import for ThreadPoolExecutor
from prompts import get_llm_judge_prompt
from vectordb.qdrant_db import QdrantDB
from embedding_model import GoogleEmbeddingModel, NebiusEmbeddingModel, DeepInfraEmbeddingModel
from no_rag import NoRAGSystem
from simple_rag import SimpleRAGSystem
from self_rag import SelfRAGSystem
from crag_rag import CRAGRAGSystem
from reranker_rag import RerankerRAGSystem
from hyde_rag import HyDERAGSystem
import os
from config import config
from llm_client import NebiusLLMClient

# Define evaluation result data structure
@dataclass
class EvaluationEntry:
    """A single evaluation entry for a specific question"""
    question: str
    ground_truth: str # Can be string for open-domain, or specific type for others
    response: str
    # Metrics specific to this single entry (e.g., latency for this query, or if it was correct for binary)
    metrics: Dict[str, Any]
    
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
        
        all_individual_metric_keys = set()
        for entry in self.entries:
            all_individual_metric_keys.update(entry.metrics.keys())
            # Ensure ground_truth and response are strings for normalization
            gt_normalized = self._normalize_label(str(entry.ground_truth))
            resp_normalized = self._normalize_label(entry.response)
            # Check if the normalized ground truth matches the normalized response
            if gt_normalized == resp_normalized:
                correct_predictions += 1
        
        accuracy = (correct_predictions / total_predictions) if total_predictions > 0 else -1
            
        calculated_summary_metrics = {}

        for metric_key in all_individual_metric_keys:
            values = [entry.metrics.get(metric_key) for entry in self.entries if entry.metrics.get(metric_key) is not None]
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            bool_values = [v for v in values if isinstance(v, bool)]
            if bool_values:
                # If there are boolean values, calculate the proportion of True values
                calculated_summary_metrics[f"prop_{metric_key}"] = sum(bool_values) / len(bool_values)
            elif numeric_values:
                # Ensure all values are numeric for averaging
                calculated_summary_metrics[f"avg_{metric_key}"] = sum(numeric_values) / len(numeric_values)
            # for the remaining data types, we dont store as summary metrics (for example strings, lists)
        
        self.summary_metrics = calculated_summary_metrics
        
        self.summary_metrics.update({
            "accuracy": accuracy,
            "total_questions": float(total_predictions),
            "correct_predictions": float(correct_predictions),
        })
    
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
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the evaluator class"""
        pass

    def get_result_dicts(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in list(self.results.values())]
    
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

    def _process_single_query(self, system: LLMSystem, query_tuple: Tuple[str, str]) -> EvaluationEntry:
        """
        Processes a single query: formats it, gets system response, calculates metrics.
        This function is executed by each worker thread.
        """
        query, ground_truth = query_tuple
        
        start_time = time.time()
        yesno_formatted_query = RAGEvaluatorYesNoQuestion.get_yesno_query(query)
        
        response, metrics = system.query(query, yesno_formatted_query)
        
        latency = time.time() - start_time

        # Normalize response and ground_truth for consistent "yes" / "no"
        normalized_response = response.strip(" .").lower()
        normalized_ground_truth = ground_truth.strip(" .").lower()

        # This can be used by BinaryClassificationEvaluationResult
        metrics["latency"] = latency
            
        return EvaluationEntry(
            question=query,
            ground_truth=normalized_ground_truth, # Store normalized ground_truth
            response=normalized_response,       # Store normalized response
            metrics=metrics
        )
    
    def get_name(self) -> str:
        """Get the name of the evaluator class"""
        return "Yes/No Question Evaluator"


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

    def _process_single_open_query(self, system: LLMSystem, query_tuple: Tuple[str, str]) -> EvaluationEntry:
        """
        Processes a single open-ended query: gets system response, calculates metrics,
        and optionally performs LLM-as-judge evaluation.
        This function is executed by each worker thread.
        """
        query, ground_truth = query_tuple
        
        start_time = time.time()
        
        # Get response from the system being evaluated
        response_text, metrics = system.query(query, query) # Uses inherited method
        
        latency = time.time() - start_time
        
        metrics["latency"] = latency
            
        # Calculate content quality metrics using LLM-as-judge if applicable
        if self.use_llm_judge and ground_truth is not None: # LLM Judge typically needs ground truth
            # _llm_judge_evaluation is inherited from RAGEvaluator (or defined there)
            content_scores = self._llm_judge_evaluation(query, ground_truth, response_text)
            metrics.update(content_scores)
        elif self.use_llm_judge and ground_truth is None:
            print(f"  Skipping LLM judge for query '{query[:50]}...' as no ground truth provided.")
            
        return EvaluationEntry(
            question=query,
            ground_truth=ground_truth,
            response=response_text,
            metrics=metrics
        )
        
    def evaluate_system(self, system: LLMSystem, queries: List[Tuple[str, str]]) -> OpenDomainEvaluationResult:
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
            raise ValueError(f"No queries provided for evaluation of {system_name}. Please provide a list of query strings or tuples (query, ground_truth).")
        
        if isinstance(queries[0], str):
            raise ValueError(f"Queries for {self.__class__.__name__} must be a list of strings or tuples (query, ground_truth). Got {type(queries[0])} instead.")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_query_tuple = {
                executor.submit(self._process_single_open_query, system, query_tuple): query_tuple
                for query_tuple in queries
            }
            
            processed_count = 0
            for future in concurrent.futures.as_completed(future_to_query_tuple):
                original_query_tuple = future_to_query_tuple[future]
                try:
                    entry = future.result()
                    entries.append(entry)
                except Exception as exc:
                    print(f"Query '{original_query_tuple[0][:50]}...' generated an exception during processing: {exc}")
                finally:
                    processed_count += 1
                    if processed_count % 10 == 0 or processed_count == len(queries):
                        print(f"  Processed {processed_count}/{len(queries)} open-domain queries...")

        # Create result with all entries
        result = OpenDomainEvaluationResult(
            system_name=system_name,
            entries=entries
        )
         
        self.results[system_name] = result
        return result
        
    def _llm_judge_evaluation(self, query: str, ground_truth: str, response: str) -> Dict[str, float]:
        """
        Use an LLM to evaluate the quality of the response
        
        Returns:
            Dictionary of scores for different quality dimensions
        """
        # Create prompt for LLM judge
        prompt = get_llm_judge_prompt(query, ground_truth, response)
        
        # Get evaluation from judge model
        # This implementation depends on your specific judge model
        judge_response, _ = call_llm_assessment(prompt=prompt, use_judge_model=True)
        
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
    
    def _parse_judge_scores(self, judge_response: str) -> Dict[str, float]:
        """Parse scores from judge response"""
        try:
            # Try direct JSON parsing
            return json.loads(judge_response)
        except json.JSONDecodeError:
            # Fallback if response is wrapped in markdown code block
            try:
                json_str = judge_response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            except Exception as e:
                raise ValueError(f"Failed to parse judge response: {e}")
    
    def get_name(self) -> str:
        """Get the name of the evaluator class"""
        return "Open Question Evaluator"
    
class RAGBenchmark:
    """
    A benchmark class to evaluate multiple RAG systems on sets of queries.
    """
    def __init__(self):
        self.evaluators: List[RAGEvaluator] = []

    def eval_yes_no_questions(self, systems: List[LLMSystem], queries: List[Tuple[str, str]], n_workers: int = 1):
        """
        Evaluate multiple systems on yes/no questions.
        
        Args:
            systems: List of LLMSystem objects to evaluate.
            queries: List of (query_string, ground_truth_string) tuples.
            n_workers: Number of parallel workers to use.
            
        Returns:
            Dictionary mapping system names to their evaluation results.
        """
        evaluator = RAGEvaluatorYesNoQuestion(n_workers=n_workers)
        for system in systems:
            evaluator.evaluate_system(system, queries)
        self.evaluators.append(evaluator)  # Store evaluator for potential future use
    
    def eval_open_questions(self, systems: List[LLMSystem], queries: List[Tuple[str, str]], n_workers: int = 1):
        """
        Evaluate multiple systems on open-domain questions.
        
        Args:
            systems: List of LLMSystem objects to evaluate.
            queries: List of query strings.
            n_workers: Number of parallel workers to use.
            use_llm_judge: Whether to use an LLM as a judge for content quality.
            
        Returns:
            Dictionary mapping system names to their evaluation results.
        """
        evaluator = RAGEvaluatorOpenQuestion(n_workers=n_workers, use_llm_judge=True)
        for system in systems:
            evaluator.evaluate_system(system, queries)
        self.evaluators.append(evaluator)  # Store evaluator for potential future use

    def save_results(self, output_file: str):
        """
        Save all evaluation results from all evaluators to a single file.
        
        Args:
            output_file: Path to save the results JSON file.
        """
        all_results = {}
        for evaluator in self.evaluators:
            all_results[evaluator.get_name()] = evaluator.get_result_dicts()
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

def _setup_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Populate a vector database with a specified number of documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--db-type", type=str, default="qdrant", choices=["qdrant"], help="The type of vector database to use.")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-004", choices=["text-embedding-004", "Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-Embedding-0.6B"], help="The embedding model to use for population.")
    parser.add_argument("--use-quantization", action='store_true', help="Enable quantization in the vector database (if supported).")
    parser.add_argument('--n-workers', type=int, default=8, help='Number of worker threads for parallel processing.')

    args = parser.parse_args()
    return args

def _run_norag_evaluation(llm_client, yesno_queries, output_path):
    """
    Main function to run the RAG evaluation.
    """
    no_rag_system = NoRAGSystem("No-RAG System", llm_client)
    no_rag_benchmark = RAGBenchmark()
    start = time.time()
    no_rag_benchmark.eval_yes_no_questions(
        systems=[no_rag_system],
        queries=yesno_queries,
        n_workers=args.n_workers
    )
    end = time.time()
    print(f"No-RAG evaluation took {end - start:.2f} seconds")
    no_rag_benchmark.save_results(output_path)
            

def _run_rag_evaluation(rag_k, llm_client, db_instance, yesno_queries, output_path):
    """
    Main function to run the RAG evaluation.
    """
    simple_rag_system = SimpleRAGSystem("Simple RAG System", llm_client, db_instance, rag_k)  # Placeholder for a simple RAG system
    self_rag_system = SelfRAGSystem("Self-RAG System", llm_client, db_instance, rag_k)
    reranker_rag_system = RerankerRAGSystem("Reranker-RAG System", llm_client, db_instance, rag_k)
    crag_rag_system = CRAGRAGSystem("CRAG-RAG System", llm_client, db_instance, rag_k)
    hyde_rag_system = HyDERAGSystem("HyDE-RAG System", llm_client, db_instance, rag_k)

    rag_benchmark = RAGBenchmark()

    start = time.time()
    rag_benchmark.eval_yes_no_questions(
        systems=[simple_rag_system, self_rag_system, reranker_rag_system, crag_rag_system, hyde_rag_system],
        queries=yesno_queries,
        #queries=pubmedqa_queries,
        n_workers=args.n_workers
    )
    end = time.time()
    print(f"Yes/No question evaluation took {end - start:.2f} seconds")

    #start = time.time()
    #rag_benchmark.eval_open_questions(
    #    systems=[no_rag_system, simple_rag_system, self_rag_system, reranker_rag_system, crag_rag_system, hyde_rag_system], 
    #    queries=bioasq_open_queries[:100], 
    #    n_workers=n_workers
    #)
    #end = time.time()
    #print(f"Open question evaluation took {end - start:.2f} seconds")

    # Save results
    rag_benchmark.save_results(output_path)
    
if __name__ == "__main__":
    args = _setup_args()

    llm_list = config.LLM_LIST
    rag_k_list = config.RAG_K_LIST
    question_limit = config.EVAL_N_QUESTIONS

    output_dir = config.get("OUTPUT_DIR")
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) # Use makedirs to create all intermediate directories

    #bioasq_yesno_queries = load_bioasq_yesno_questions()[:question_limit]
    bioasq_yesno_queries = load_bioasq_yesno_questions()[:10]
    #bioasq_open_queries = load_bioasq_open_questions()[:question_limit]

    #from dataset_loaders import load_pubmedqa_questions
    #pubmedqa_queries = load_pubmedqa_questions(100)

    # --- Initialize the embedding model ---
    if args.embedding_model == "text-embedding-004":
        embedding_model = GoogleEmbeddingModel()
    elif args.embedding_model == "Qwen/Qwen3-Embedding-8B":
        embedding_model = NebiusEmbeddingModel() 
    elif args.embedding_model == "Qwen/Qwen3-Embedding-0.6B":
        embedding_model = DeepInfraEmbeddingModel()
    else:
        raise ValueError(f"Unsupported embedding model: {args.embedding_model}")
    
    print(f"Using embedding model: {embedding_model.embedding_dim} dimensions")

    if args.db_type == "qdrant":
        db_instance = QdrantDB(
            embedding_model=embedding_model,
            n_workers=args.n_workers,
            use_quantization=args.use_quantization
        )
    else:
        # This block can be expanded if you add more database types
        raise ValueError(f"Unsupported database type: {args.db_type}")

    print("Evaluating systems...")
    for llm in llm_list:
        llm_client = NebiusLLMClient(base_llm=llm)

        # Create output path for no-RAG evaluation
        llm_name = llm_client.base_llm
        model_name = llm_name.split("/")[-1]  # Extract model name from the full identifier
        output_filename = f"eval_norag_{model_name}.json"
        norag_output_path = os.path.join(output_dir, output_filename)

        _run_norag_evaluation(llm_client, bioasq_yesno_queries, norag_output_path)  # run no-RAG evaluation separately
        for rag_k in rag_k_list:
            # Create output path for RAG evaluation
            output_filename = f"eval_rag_{model_name}_{rag_k}k.json"
            rag_output_path = os.path.join(output_dir, output_filename)
            # Run the RAG evaluation for each k
            _run_rag_evaluation(rag_k, llm_client, db_instance, bioasq_yesno_queries, rag_output_path)
