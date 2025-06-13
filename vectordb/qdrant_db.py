import time
import concurrent.futures
from .base_db import VectorDB
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, Batch, ScalarQuantization, ScalarType, ScalarQuantizationConfig, SearchParams, QuantizationSearchParams, HnswConfigDiff
from tenacity import retry, stop_after_attempt, wait_exponential
from embedding_model import EmbeddingModel, GoogleEmbeddingModel
from dataset_loaders import load_pubmed_data
from config import config

# RAG Configuration
DEFAULT_N_RESULTS = config.get("RAG_FINAL_CONTEXT_K") # Default number of results to return from retrieval

class QdrantDB(VectorDB):
    def __init__(self, embedding_model: EmbeddingModel, n_workers: int = 4, use_quantization: bool = True):
        """
        Initializes the QdrantDB with the specified embedding model and number of worker threads.
        :param embedding_model: The embedding model to use for generating embeddings.
        :param n_workers: The number of worker threads to use for parallel processing.
        :param use_quantization: Whether to use quantization for the embeddings.
        """
        super().__init__(embedding_model, n_workers)
        
        # --- API BATCH LIMIT ---
        self.API_BATCH_SIZE = 1000

        # set quantization parameters if enabled
        self.use_quantization = use_quantization
        if use_quantization:
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=1.0,
                    always_ram=True,
                ),
            )
        else:
            quantization_config = None

        # --- Initialize Qdrant Client ---
        self.client = QdrantClient(
            url="http://localhost:6333",
            timeout=60 # Set timeout to 60 seconds (default is typically 5 or 10))
        )
        self.collection_name = "qdrant_vectordb"
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_model.embedding_dim,
                    distance=Distance.COSINE,
                    on_disk=True, # Use on-disk storage for low memory usage
                ),
                quantization_config=quantization_config,
                hnsw_config=HnswConfigDiff(
                    m=0, # Defer HNSW graph construction for Low memory usage during upload
                    on_disk=True, # Use on-disk storage for low memory usage
                ),
            )
        self.populate_data(4000000, load_pubmed_data)
        self.client.update_collection(
            collection_name=self.collection_name,
            hnsw_config=HnswConfigDiff(
                m=16, # Once ingestion is complete, re-enable HNSW by setting m to production value
            ),
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60), # Wait 2^x * 1 seconds between each retry, starting with 4s, maxing out at 60s
        stop=stop_after_attempt(5), # Stop after 5 attempts
        reraise=True # Reraise the exception if all retries fail
    )
    def _embed_and_upsert_chunk(self, chunk_texts: list[str], chunk_ids: list[int]) -> int:
        """
        This is the core task for each worker thread.
        It takes a chunk of texts, embeds them, and upserts them into Qdrant.
        """
        # 1. Embed the chunk of texts
        embeddings = self.embedding_model.embed_documents(chunk_texts)
        
        # 2. Upsert the embedded chunk into Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=Batch(
                ids=chunk_ids,
                vectors=embeddings,
                payloads=[{"text": text} for text in chunk_texts]
            )
        )
        return len(chunk_texts)

    def populate_data(self, target_docs_to_process: int, data_loader_func):
        """
        Populates Qdrant using a streaming producer-consumer pattern.
        """
        assert target_docs_to_process > 0, "target_docs_to_process must be greater than 0"
        num_points = self.client.count(
            collection_name=self.collection_name,
            exact=True, # Use exact count for accuracy
        ).count
        print(f"Collection '{self.collection_name}' contains {num_points} points.")
        if num_points == target_docs_to_process:
            # No need to populate, the collection already has the target number of documents
            print(f"Collection already has {num_points} points, which matches the target of {target_docs_to_process}. No population needed.")
            return
        remaining_docs = target_docs_to_process - num_points
        print(f"Need to add {remaining_docs} more documents to reach the target of {target_docs_to_process}.")
        
        MAX_PENDING_FUTURES_FACTOR = 2
        max_concurrent_futures = self.n_workers * MAX_PENDING_FUTURES_FACTOR

        print(f"\n--- Populating Qdrant vectordb with embeddings (streaming & parallel) ---")
        population_start_time = time.time()
        
        # The unit of work for a worker will be the API's batch size
        WORKER_BATCH_SIZE = self.API_BATCH_SIZE
        
        # skip the first num_points documents, as they are already in the collection
        batch_iterator = data_loader_func(remaining_docs, WORKER_BATCH_SIZE, num_points)
        
        total_docs_added = 0
        total_docs_submitted = 0
        processed_tasks_count = 0
        submitted_tasks_count = 0
        active_futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            
            def submit_next_task(iterator):
                nonlocal submitted_tasks_count, total_docs_submitted
                try:
                    batch_texts = next(iterator)
                    if not batch_texts: return False
                    # Make the ID absolute by adding the number of points already in the DB.
                    start_id = num_points + total_docs_submitted
                    #start_id = total_docs_submitted
                    batch_ids = list(range(start_id, start_id + len(batch_texts)))
                    
                    future = executor.submit(self._embed_and_upsert_chunk, batch_texts, batch_ids)
                    active_futures.append(future)
                    
                    submitted_tasks_count += 1
                    total_docs_submitted += len(batch_texts)
                    return True
                except StopIteration:
                    return False

            print(f"Initially submitting up to {max_concurrent_futures} tasks...")
            for _ in range(max_concurrent_futures):
                if not submit_next_task(batch_iterator): break
            print(f"Initial submission done. {len(active_futures)} tasks running.")

            while active_futures:
                done_futures, active_futures_set = concurrent.futures.wait(
                    active_futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                active_futures = list(active_futures_set)

                for future in done_futures:
                    processed_tasks_count += 1
                    try:
                        added_count = future.result()
                        total_docs_added += added_count
                    except Exception as exc:
                        print(f'A task generated an exception: {exc}')
                    
                    if processed_tasks_count % 10 == 0 or not active_futures:
                        elapsed = time.time() - population_start_time
                        print(f"  Completed {processed_tasks_count}/{submitted_tasks_count} tasks. "
                              f"Total docs added: {total_docs_added:,}. "
                              f"Pending tasks: {len(active_futures)}. Elapsed: {elapsed:.1f}s")
                
                while len(active_futures) < max_concurrent_futures:
                    if not submit_next_task(batch_iterator): break

        # Finalize the population
        if active_futures:
            done_futures, _ = concurrent.futures.wait(active_futures)
            for future in done_futures:
                try:
                    added_count = future.result()
                    total_docs_added += added_count
                except Exception as exc:
                    print(f'A final task generated an exception: {exc}')
        population_end_time = time.time()
        print("\n--- Population Complete ---")
        print(f"Finished Qdrant population. Added ~{total_docs_added:,} documents "
              f"in {population_end_time - population_start_time:.2f} seconds.")

    def retrieve_context(self, query: str, n_results: int = DEFAULT_N_RESULTS) -> list[str]:
        """Retrieves context from the Qdrant database based on the query."""
        start = time.time()
        query_vector = self.embedding_model.embed_query(query)
        search_params = None
        if self.use_quantization:
            search_params=SearchParams(
                quantization=QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=3.0,
                )
            )
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=n_results,
            search_params=search_params
        )
        end = time.time()
        print(f"Retrieved {len(results.points)} results in {end - start:.2f} seconds.")
        # return payloads only
        #for point in results.points:
        #    print(f"ID: {point.id}, Score: {point.score:.4f}")
        return [point.payload['text'] for point in results.points]

if __name__ == "__main__":
    # Using workers for parallel processing
    embedding_model = GoogleEmbeddingModel()
    qdrant_db = QdrantDB(n_workers=4, embedding_model=embedding_model, use_quantization=True)
    
    query = "Can PRL3-zumab inhibit PRL3+ cancer cells in vitro and in vivo?"
    results = qdrant_db.retrieve_context(query)
    print("\n--- Retrieval Results ---")
    for text in results:
        #print(f"ID: {point.id}, Score: {point.score:.4f}, Text: {point.payload['text']}")
        print(text)
        print()