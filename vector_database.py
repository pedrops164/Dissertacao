# vector_database.py

import numpy as np
from llm_client import openai_client # Reuse the client from llm_client.py
import chromadb                     # For the persistent vector database
import uuid                         # For generating unique document IDs
import time
import re                           # For regex tokenization
import os
import pickle
import concurrent.futures
from collections.abc import Callable
from dataset_loaders import load_pubmed_data  # Import dataset loaders
from config import config

# --- Configuration ---

# BM25 Persistence Configuration
BM25_INDEX_DIR_BASE = config.get("BM25_DIR") # Directory for BM25 index files
BM25_INDEX_FILENAME = "bm25_index.pkl"
BM25_TEXTS_FILENAME = "bm25_texts.pkl" # To store texts associated with BM25
USE_BM25 = config.get("USE_BM25", True) # Whether to use BM25 for keyword search

MAX_WORKERS = 8 # Number of parallel workers for embedding
MAX_PENDING_FUTURES_FACTOR = 3 # e.g., allow up to MAX_WORKERS * 2 pending tasks

# Embedding model configuration
EMBEDDING_MODEL = config.get("NEBIUS_EMBEDDING_MODEL") # Get the embedding model from config

# RAG Configuration
DEFAULT_N_RESULTS = config.get("SELF_RAG_FINAL_CONTEXT_K") # Default number of results to return from retrieval

# Vector Database Configuration
# Ensure the database size is one of the available options
AVAILABLE_DB_SIZES = [5000, 50000, 100000, 200000, 500000] # Available num rows for the database to store
DB_SIZE = config.get("DB_SIZE") # Ensure this is set to one of AVAILABLE_DB_SIZES
if DB_SIZE not in AVAILABLE_DB_SIZES:
    raise ValueError(f"DB_SIZE must be one of {AVAILABLE_DB_SIZES}. Current value: {DB_SIZE}")
DATABASE_NAME = f"db_{DB_SIZE}" # Name of the database collection in ChromaDB

# Batch size for adding documents to ChromaDB and embedding
BATCH_SIZE = config.get("BATCH_SIZE") # Process these documents at a time

# Ensure the ChromaDB persistence directory exists
DB_DIR = config.get("DB_DIR") # Use the configured directory or default
if not os.path.exists(DB_DIR):
    print(f"Creating ChromaDB persistence directory: {DB_DIR}")
    os.makedirs(DB_DIR, exist_ok=True)

# --- Vector Database Class using ChromaDB ---
class VectorDatabase:
    """
    Manages document embeddings and similarity search using ChromaDB for persistence.
    """
    def __init__(self):
        database_persist_dir = os.path.join(DB_DIR, DATABASE_NAME)
        self.client = chromadb.PersistentClient(path=database_persist_dir)
        self.embedding_model = EMBEDDING_MODEL
        self.collection_name = DATABASE_NAME
        print(f"Getting or creating ChromaDB collection: '{DATABASE_NAME}'")
        self.collection = self.client.get_or_create_collection(name=DATABASE_NAME)
        
        # Ensure the collection is populated with data
        self.populate_collection_if_needed(
            target_original_docs_to_process=DB_SIZE,
            data_loader_func=load_pubmed_data,  # Use the appropriate data loader function
        )
        print(f"Chroma Collection '{DATABASE_NAME}' loaded. Current count: {self.collection.count():,} documents (chunks).")

        if USE_BM25:
            # In-memory storage for BM25
            self.bm25_index = BM25Index(self._get_all_documents_from_chroma)  # Initialize BM25 index object

    def _get_embedding_batch(self, batch_texts):
        """Generates embeddings for a batch of texts using the configured client."""
        try:
            # Replace newlines for API compatibility
            processed_texts = [text.replace("\n", " ") for text in batch_texts]
            response = openai_client.embeddings.create(input=processed_texts, model=self.embedding_model)
            # Return a list of numpy arrays
            return [np.array(embedding_data.embedding) for embedding_data in response.data]
        except Exception as e:
            print(f"Warning: Error getting embedding batch (model: {self.embedding_model}): {e}. Returning None for batch.")
            # Returning None signals failure for the whole batch
            return None

    def _add_documents_batch_chroma(self, batch_texts):
        """Embeds a batch of texts and adds them to the Chroma collection."""
        if not batch_texts:
            return 0

        start_time = time.time()
        embeddings = self._get_embedding_batch(batch_texts)
        embed_time = time.time()

        if embeddings is None or len(embeddings) != len(batch_texts):
            print(f"Error embedding batch of size {len(batch_texts)}. Skipping Chroma add.")
            return 0 # Indicate 0 documents added for this batch

        # Generate unique IDs for this batch
        ids = [str(uuid.uuid4()) for _ in batch_texts]

        try:
            # Add the batch to ChromaDB
            # Convert numpy arrays to lists for ChromaDB compatibility
            self.collection.add(
                ids=ids,
                embeddings=[emb.tolist() for emb in embeddings],
                documents=batch_texts
            )
            add_time = time.time()
            print(f"  Embedded {len(batch_texts)} docs in {embed_time - start_time:.2f}s. Added to Chroma in {add_time - embed_time:.2f}s.")
            return len(batch_texts) # Return number successfully added
        except Exception as e:
            print(f"Error adding batch to ChromaDB: {e}")
            return 0

    def _get_all_documents_from_chroma(self):
        """Retrieves all documents (texts) from the ChromaDB collection."""
        print("Attempting to load all documents from ChromaDB for BM25...")
        if self.collection.count() == 0:
            print("Warning: ChromaDB is empty. Cannot load documents.")
            return []
        try:
            # Batch retrieval might be needed for very large collections
            # Chroma's get() with no IDs/where returns all. Max limit might apply.
            # For truly massive collections, this needs pagination.
            # Default limit for get() is 100, so we must fetch all with a limit.
            results = self.collection.get(include=['documents'], limit=self.collection.count())
            print(f"Retrieved {len(results['documents'])} documents from ChromaDB.")
            return results['documents']
        except Exception as e:
            print(f"Error retrieving all documents from ChromaDB: {e}")
            return []

    def populate_collection_if_needed(self, target_original_docs_to_process, data_loader_func):
        """Checks collection count and populates it if necessary."""
        current_chunk_count = self.collection.count()
        needs_chroma_population = current_chunk_count == 0

        print(f"Checking collection '{self.collection_name}'. Found {current_chunk_count:,} chunks.")
        print(f"Target original documents to process for potential population: {target_original_docs_to_process:,}")

        # --- Step 1 & 2: Load Data & Populate ChromaDB (if needed) ---
        if needs_chroma_population:
            print(f"\n--- Populating ChromaDB with Parallel Embedding (Streaming) ---")
            total_added_chroma = 0
            population_start_time = time.time()
            
            active_futures = []
            # data_loader_func is expected to be a generator yielding batches of CHUNKED texts
            batch_iterator = data_loader_func(target_original_docs_to_process, BATCH_SIZE) 
            
            processed_batches_count = 0
            # This counter will track batches submitted to the executor
            # It's distinct from processed_batches_count which tracks completed batches.
            _submitted_batch_global_counter = 0
            
            max_concurrent_futures = MAX_WORKERS * MAX_PENDING_FUTURES_FACTOR


            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                
                # Inner helper function to encapsulate submission logic
                def _submit_next_batch_to_executor(iterator, active_futures_list):
                    nonlocal _submitted_batch_global_counter # To modify the counter in the outer scope
                    try:
                        batch_texts = next(iterator)
                        if not batch_texts: # Handle empty batch from loader
                            print("Warning: Data loader yielded an empty batch. Stopping submission.")
                            return False # Indicate nothing was submitted
                        
                        future = executor.submit(self._add_documents_batch_chroma, batch_texts)
                        active_futures_list.append(future)
                        _submitted_batch_global_counter += 1
                        # Log submission of initial batches
                        if _submitted_batch_global_counter <= max_concurrent_futures:
                             print(f"  Submitted initial task {_submitted_batch_global_counter}/{max_concurrent_futures} to executor...")
                        elif _submitted_batch_global_counter % 50 == 0: # Log every 50 submissions after initial fill
                             print(f"  Submitted task {_submitted_batch_global_counter} to executor...")
                        return True # Task submitted
                    except StopIteration:
                        pass # Iterator exhausted
                    return False # No task submitted

                # Fill the queue initially
                for _ in range(max_concurrent_futures):
                    if not _submit_next_batch_to_executor(batch_iterator, active_futures):
                        break 
                
                while active_futures:
                    done_futures, active_futures_set = concurrent.futures.wait(
                        active_futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    active_futures = list(active_futures_set) 
                    
                    for future in done_futures:
                        processed_batches_count += 1
                        try:
                            added_count = future.result()
                            total_added_chroma += added_count
                        except Exception as exc:
                            print(f'A batch (task {processed_batches_count}) generated an exception during processing: {exc}')
                        
                        if processed_batches_count % 20 == 0 or not active_futures: 
                            elapsed = time.time() - population_start_time
                            print(f"  Completed {processed_batches_count} tasks. Total chunks added: {total_added_chroma:,}. Pending: {len(active_futures)}. Elapsed: {elapsed:.1f}s")

                    # Try to replenish the queue with new tasks
                    while len(active_futures) < max_concurrent_futures:
                        if not _submit_next_batch_to_executor(batch_iterator, active_futures):
                            break 
                
                print(f"All {_submitted_batch_global_counter} submitted tasks have been processed.")
            # End of 'with executor' block

            population_end_time = time.time()
            print(f"Finished ChromaDB population. Added {total_added_chroma:,} chunks in {population_end_time - population_start_time:.2f}s.")
            print(f"Collection '{self.collection_name}' now contains {self.collection.count():,} chunks.")

    def search_vector(self, query, n_results=DEFAULT_N_RESULTS):
        """Performs vector search using ChromaDB."""
        print(f"Performing vector search (ChromaDB) for: '{query[:50]}...' (k={n_results})")
        if self.collection.count() == 0:
            print("ChromaDB collection is empty. Cannot perform vector search.")
            return []
        start_time = time.time()
        query_embedding_list = self._get_embedding_batch([query])
        if query_embedding_list is None or not query_embedding_list:
            print("Failed to get query embedding.")
            return []
        query_embedding = query_embedding_list[0]
        try:
            results = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=n_results, include=['documents'])
            documents_list = results.get('documents')
            if documents_list is None or not documents_list or documents_list[0] is None:
                return []
            documents = documents_list[0]
            print(f"Vector search completed in {time.time() - start_time:.4f}s, found {len(documents)} results.")
            return documents
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return []
        
class BM25Index:
    def __init__(self, text_loader_func: Callable):
        """
        Initializes the BM25 index with optional pre-loaded texts.
        If texts are provided, it builds the index immediately.
        """
        # Ensure BM25 directory exists
        self.bm25_index_dir = os.path.join(BM25_INDEX_DIR_BASE, DATABASE_NAME)
        os.makedirs(self.bm25_index_dir, exist_ok=True) # Ensure directory for this specific DB exists

        self.bm25_index_path = os.path.join(self.bm25_index_dir, BM25_INDEX_FILENAME)
        self.bm25_texts_path = os.path.join(self.bm25_index_dir, BM25_TEXTS_FILENAME)

        self.bm25_index = None
        self.texts = None # Texts associated with the BM25 index

        print(f"Initializing BM25Index for '{DATABASE_NAME}'.")
        if self._load_bm25():
            print(f"BM25Index for '{DATABASE_NAME}' loaded successfully from persisted files.")
        else:
            print(f"Could not load persisted BM25Index for '{DATABASE_NAME}'. Attempting to build...")
            assert text_loader_func is not None, "text_loader_func must be provided to build BM25 index."
            print("Building BM25 index using text_loader_func...")
            loaded_texts = text_loader_func()
            if not loaded_texts:
                raise ValueError("text_loader_func returned no texts. Cannot build BM25 index.")
            self.texts = loaded_texts
            self._build_bm25()

    # --- Simple Tokenizer for BM25 ---
    @staticmethod
    def simple_tokenizer(text):
        """Basic tokenizer: lowercase and split by non-alphanumeric characters."""
        if not isinstance(text, str):
            return []
        # Lowercase, then split by one or more non-alphanumeric characters
        tokens = re.split(r'\W+', text.lower())
        # Filter out empty strings that can result from multiple delimiters
        return [token for token in tokens if token]

    def _build_bm25(self):
        from rank_bm25 import BM25Okapi # Import here to avoid circular dependency if BM25Okapi is heavy

        if not self.texts: # self.texts should be populated before calling _build_bm25
            print(f"Error: Cannot build BM25 index for '{DATABASE_NAME}' as self.texts is not populated.")
            return
        
        print(f"Building BM25 index for '{DATABASE_NAME}' from {len(self.texts):,} documents...")
        start_time = time.time()
        tokenized_corpus = [BM25Index.simple_tokenizer(doc) for doc in self.texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        end_time = time.time()
        print(f"BM25 index for '{DATABASE_NAME}' built in {end_time - start_time:.2f} seconds.")
        self._save_bm25()

    def _save_bm25(self):
        if self.bm25_index and self.texts:
            print(f"Saving BM25 index to {self.bm25_index_path}")
            with open(self.bm25_index_path, 'wb') as f:
                pickle.dump(self.bm25_index, f)
            print(f"Saving BM25 texts ({len(self.texts)} chunks) to {self.bm25_texts_path}")
            with open(self.bm25_texts_path, 'wb') as f:
                pickle.dump(self.texts, f)
        else:
            print("BM25 index or texts not available for saving.")

    def _load_bm25(self):
        print(f"Attempting to load BM25 index and texts for '{DATABASE_NAME}'...")
        if os.path.exists(self.bm25_index_path) and os.path.exists(self.bm25_texts_path):
            try:
                with open(self.bm25_index_path, 'rb') as f:
                    self.bm25_index = pickle.load(f)
                with open(self.bm25_texts_path, 'rb') as f:
                    self.texts = pickle.load(f)
                print(f"  Successfully loaded BM25 index and {len(self.texts):,} associated texts for '{DATABASE_NAME}'.")
                return True
            except Exception as e:
                print(f"  Error loading BM25 data for '{DATABASE_NAME}': {e}. Will attempt to rebuild if possible.")
                self.bm25_index = None
                self.texts = None
                return False
        else:
            print(f"  BM25 index file ({self.bm25_index_path}) or texts file ({self.bm25_texts_path}) not found for '{DATABASE_NAME}'.")
            return False

    def search_bm25(self, query, n_results=DEFAULT_N_RESULTS):
        """Performs keyword search using BM25."""
        print(f"Performing keyword search (BM25) for: '{query[:50]}...' (k={n_results})")
        if self.bm25_index is None or self.texts is None:
            raise ValueError("BM25 index or texts not loaded. Cannot perform search.")
        start_time = time.time()
        tokenized_query = BM25Index.simple_tokenizer(query)
        if not tokenized_query: return []
        # BM25Okapi.get_top_n returns the actual documents (texts)
        try:
            scores = self.bm25_index.get_scores(tokenized_query)
            # Get indices sorted by score, then map back to texts
            # Ensure we don't request more results than available texts or with zero score
            # Filter out zero scores before sorting for efficiency if many zero scores are expected
            
            # Get top N indices for scores > 0
            positive_score_indices = np.where(scores > 0)[0]
            if len(positive_score_indices) == 0:
                 print(f"BM25 search completed in {time.time() - start_time:.4f}s, found 0 results with score > 0.")
                 return []

            top_n_indices = sorted(positive_score_indices, key=lambda i: float(scores[i]), reverse=True)[:n_results]
            documents = [self.texts[i] for i in top_n_indices]
            
            # for i in top_n_indices:
            # print(f"  Score: {scores[i]:.4f} - {self.texts[i][:100]}...")
            print(f"BM25 search completed in {time.time() - start_time:.4f}s, found {len(documents)} result(s).")
            return documents
        except Exception as e:
             print(f"Error during BM25 search: {e}")
             return []


# --- Initialization and Population ---
print("--- Initializing Vector Database Setup (ChromaDB) ---")
# Initialize the VectorDatabase class, connecting to ChromaDB
vector_db = VectorDatabase()

print("--- Vector Database Setup Complete ---")

# --- Main retrieval function for RAG ---
def retrieve_context(query, n_results=DEFAULT_N_RESULTS):
    """
    Retrieves relevant context from the ChromaDB vector database based on the query.
    """

    print(f"\nRetrieving context for query: '{query}' using ChromaDB")
    results = vector_db.search_vector(query, n_results=n_results)
    return results
