import time
import concurrent.futures
from .base_db import VectorDB
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, Batch, ScalarQuantization, ScalarType, ScalarQuantizationConfig, BinaryQuantization, BinaryQuantizationConfig, SearchParams, QuantizationSearchParams, HnswConfigDiff
from tenacity import retry, stop_after_attempt, wait_exponential
from embedding_model import EmbeddingModel, GoogleEmbeddingModel
from dataset_loaders import load_pubmed_data
from config import config

# RAG Configuration
BATCH_SIZE = config.get("BATCH_SIZE") # Batch size for vector database operations

class QdrantDB(VectorDB):
    def __init__(self, embedding_model: EmbeddingModel, n_workers: int = 4, use_quantization: bool = True):
        """
        Initializes the QdrantDB with the specified embedding model and number of worker threads.
        :param embedding_model: The embedding model to use for generating embeddings.
        :param n_workers: The number of worker threads to use for parallel processing.
        :param use_quantization: Whether to use quantization for the embeddings.
        """
        super().__init__(embedding_model, n_workers)
        
        # set quantization parameters if enabled
        self.use_quantization = use_quantization
        if use_quantization:
            #quantization_config=ScalarQuantization(
            #    scalar=ScalarQuantizationConfig(
            #        type=ScalarType.INT8,
            #        quantile=1.0,
            #        always_ram=True,
            #    ),
            #)
            quantization_config=BinaryQuantization(
                binary=BinaryQuantizationConfig(
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
        self.collection_name = "qdrant_vectordb_binary_quant"
        if not self.client.collection_exists(self.collection_name):
            print(f"Collection '{self.collection_name}' not found. Creating a new one.")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_model.embedding_dim,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
                quantization_config=quantization_config,
                hnsw_config=HnswConfigDiff(
                    m=0,  # Defer HNSW graph construction for faster initial upload
                    on_disk=True,
                ),
            )
        else:
            print(f"Using existing collection: '{self.collection_name}'")

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
        print(f"Embedding {len(chunk_texts)} documents.. {chunk_ids[0]}-{chunk_ids[-1]}")
        embeddings = self.embedding_model.embed_documents(chunk_texts)
        print(f"Embedded {chunk_ids[0]}-{chunk_ids[-1]}")
        
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

        print("\n--- Preparing for Population ---")
        self.client.update_collection(collection_name=self.collection_name, hnsw_config=HnswConfigDiff(m=0))
        print("HNSW indexing deferred for faster ingestion.")

        num_points = self.client.count(collection_name=self.collection_name, exact=True).count # Use exact count for accuracy
        print(f"Collection '{self.collection_name}' contains {num_points} points.")

        if num_points >= target_docs_to_process:
            print(f"Collection already meets or exceeds the target of {target_docs_to_process} documents. No population needed.")
            self.client.update_collection(collection_name=self.collection_name, hnsw_config=HnswConfigDiff(m=16, on_disk=True))
            print("HNSW indexing has been re-enabled.")
            return
        
        remaining_docs = target_docs_to_process - num_points
        print(f"Need to add {remaining_docs} more documents to reach the target of {target_docs_to_process}.")
        
        max_concurrent_futures = self.n_workers * 2

        print(f"\n--- Populating Qdrant vectordb with embeddings (streaming & parallel) ---")
        population_start_time = time.time()
        
        # skip the first num_points documents, as they are already in the collection
        batch_iterator = data_loader_func(remaining_docs, BATCH_SIZE, num_points)
        
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

        print("\n--- Population Ingestion Complete ---")
        print(f"Added ~{total_docs_added:,} documents in {time.time() - population_start_time:.2f} seconds.")

        print("\n--- Finalizing Collection: Re-enabling HNSW indexing (m=16). This may take some time... ---")
        finalizing_start_time = time.time()
        self.client.update_collection(
            collection_name=self.collection_name,
            hnsw_config=HnswConfigDiff(m=16, on_disk=True),
        )
        print(f"HNSW indexing re-enabled in {time.time() - finalizing_start_time:.2f} seconds.")


    def retrieve_context(self, query: str, n_results: int) -> list[str]:
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