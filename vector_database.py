# vector_database.py

import numpy as np
from llm_client import openai_client # Reuse the client from llm_client.py
import chromadb                     # For the persistent vector database
import uuid                         # For generating unique document IDs
import time
from rank_bm25 import BM25Okapi     # For keyword search
import re                           # For regex tokenization
import os
import pickle
import traceback
import concurrent.futures
from dataset_loaders import load_googlenq_data, load_pubmed_data, get_googlenq_dev_questions  # Import dataset loaders for Google NQ and MedRAG

# --- Configuration ---
EMBEDDING_MODEL = "BAAI/bge-en-icl"
DEFAULT_N_RESULTS = 3
NUM_ROWS_TO_LOAD = 50000 # Number of ORIGINAL documents to process

# ChromaDB Configuration
CHROMA_PERSIST_DIR = "./chroma_db_wikipedia" # Directory to store ChromaDB data
CHROMA_COLLECTION_NAME = "wikipedia_collection"
# Batch size for adding documents to ChromaDB and embedding
BATCH_SIZE = 500 # Process these documents at a time

# BM25 Persistence Configuration
BM25_INDEX_DIR = "./bm25_data"
BM25_INDEX_FILENAME = "bm25_index.pkl"
BM25_TEXTS_FILENAME = "bm25_texts.pkl" # To store texts associated with BM25

MAX_WORKERS = 8 # <--- ADDED: Number of parallel workers for embedding

# --- Vector Database Class using ChromaDB ---
class VectorDatabase:
    """
    Manages document embeddings and similarity search using ChromaDB for persistence.
    """
    def __init__(self, persist_directory, collection_name, embedding_model=EMBEDDING_MODEL):
        print(f"Initializing ChromaDB client at '{persist_directory}'")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        print(f"Getting or creating ChromaDB collection: '{collection_name}'")
        # Note: Embedding function details aren't needed here if we embed before adding
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # In-memory storage for BM25
        self.bm25_index = BM25Index()  # Initialize BM25 index object
        print(f"Chroma Collection '{collection_name}' loaded. Current count: {self.collection.count():,} documents (chunks).")

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
            print("ChromaDB is empty. Cannot load documents.")
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
            #all_texts_for_bm25 = []
            total_added_chroma = 0
            population_start_time = time.time()
            futures = []

            # Use ThreadPoolExecutor for parallel embedding and adding
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                print("Starting to load data and submit batches to executor...")
                batch_count = 0
                # Iterate through batches yielded by the data loader
                for batch_texts in data_loader_func(target_original_docs_to_process, BATCH_SIZE):
                    if not batch_texts:
                        print("Warning: Empty batch received from data loader. Skipping.")
                        continue

                    batch_count += 1
                    # Store texts for BM25 (still needs all texts in memory for rank_bm25)
                    #all_texts_for_bm25.extend(batch_texts)

                    # Submit the *received batch* directly for processing
                    future = executor.submit(self._add_documents_batch_chroma, batch_texts)
                    futures.append(future)

                    if batch_count % 20 == 0: # Log progress every 20 batches submitted
                        #print(f"  Submitted {batch_count} batches ({len(all_texts_for_bm25):,} docs) to executor...")
                        print(f"  Submitted {batch_count} batches to executor...")

                print(f"Finished loading data. Submitted {len(futures)} total batches. Waiting for processing...")

                # Process results as they complete to get the total count
                results = []
                processed_count = 0
                for future in concurrent.futures.as_completed(futures):
                    processed_count += 1
                    try:
                        data = future.result() # Get the count from _add_documents_batch_chroma
                        results.append(data)
                    except Exception as exc:
                        print(f'Batch generated an exception during processing: {exc}')
                        results.append(0) # Count as 0 added if an error occurs

                    if processed_count % 10 == 0 or processed_count == len(futures):
                        elapsed = time.time() - population_start_time
                        print(f"  Processed {processed_count}/{len(futures)} batches... ({elapsed:.1f}s)")


            total_added_chroma = sum(results)
            #self.bm25_index.set_texts(all_texts_for_bm25)

            population_end_time = time.time()
            print(f"Finished ChromaDB population. Added {total_added_chroma:,} chunks in {population_end_time - population_start_time:.2f}s.")
            print(f"Collection '{self.collection_name}' now contains {self.collection.count():,} chunks.")
            #print(f"Total documents now in memory for BM25: {len(all_texts_for_bm25):,}")
        else:
            # Chroma has data, but script restarted, bm25 index isnt built.
            # Load texts from Chroma to build BM25.
            print("\n--- Loading Texts from ChromaDB for BM25 ---")
            texts = self._get_all_documents_from_chroma()
            if not texts:
                raise ValueError("No texts available from ChromaDB for BM25 index.")
            self.bm25_index.set_texts(texts)
            print(f"Loaded {len(texts):,} chunks from ChromaDB into memory for BM25.")

        return True

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
            documents = results.get('documents', [[]])[0]
            print(f"Vector search completed in {time.time() - start_time:.4f}s, found {len(documents)} results.")
            return documents
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return []
        
class BM25Index:
    def __init__(self, texts=None):
        """
        Initializes the BM25 index with optional pre-loaded texts.
        If texts are provided, it builds the index immediately.
        """
        # Ensure BM25 directory exists
        os.makedirs(BM25_INDEX_DIR, exist_ok=True)
        self.bm25_index_path = os.path.join(BM25_INDEX_DIR, BM25_INDEX_FILENAME)
        self.bm25_texts_path = os.path.join(BM25_INDEX_DIR, BM25_TEXTS_FILENAME)

        self.bm25_index = None
        self.texts = texts
        if self.texts:
            # If texts are provided, build the BM25 index immediately
            self._build_bm25()
        else:
            # attempt to load existing BM25 index and texts
            self._load_bm25()

    def set_texts(self, texts):
        """Sets the texts for the BM25 index and rebuilds it."""
        self.texts = texts
        self._build_bm25()

    # --- Simple Tokenizer for BM25 ---
    def simple_tokenizer(text):
        """Basic tokenizer: lowercase and split by non-alphanumeric characters."""
        if not isinstance(text, str):
            return []
        # Lowercase, then split by one or more non-alphanumeric characters
        tokens = re.split(r'\W+', text.lower())
        # Filter out empty strings that can result from multiple delimiters
        return [token for token in tokens if token]

    def _build_bm25(self):
        """Builds the BM25 index from self.texts."""
        if self.texts is None:
            raise ValueError("Text data not loaded for BM25 index.")
        print(f"\nBuilding BM25 index from {len(self.texts):,} documents...")
        start_time = time.time()
        # Tokenize documents
        tokenized_corpus = [BM25Index.simple_tokenizer(doc) for doc in self.texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        end_time = time.time()
        print(f"BM25 index built in {end_time - start_time:.2f} seconds.")
        self._save_bm25() # Save after building

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
        if os.path.exists(self.bm25_index_path) and os.path.exists(self.bm25_texts_path):
            print(f"Loading BM25 index from {self.bm25_index_path}...")
            try:
                with open(self.bm25_index_path, 'rb') as f:
                    self.bm25_index = pickle.load(f)
                print(f"Loading BM25 texts from {self.bm25_texts_path}...")
                with open(self.bm25_texts_path, 'rb') as f:
                    self.texts = pickle.load(f)
                print(f"Successfully loaded BM25 index and {len(self.texts)} associated text chunks.")
            except Exception as e:
                print(f"Error loading BM25 data: {e}. Index or texts might need rebuilding.")
                self.bm25_index = None
                self.texts = None # Clear texts if index loading failed to maintain consistency
        else:
            print("BM25 index file or texts file not found. Index will be built if data is populated.")

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

            top_n_indices = sorted(positive_score_indices, key=lambda i: scores[i], reverse=True)[:n_results]
            documents = [self.texts[i] for i in top_n_indices]
            
            # for i in top_n_indices:
            # print(f"  Score: {scores[i]:.4f} - {self.texts[i][:100]}...")
            print(f"BM25 search completed in {time.time() - start_time:.4f}s, found {len(documents)} result(s).")
            return documents
        except Exception as e:
             print(f"Error during BM25 search: {e}")
             return []


# --- Initialization and Population ---
vector_db = None
print("--- Initializing Vector Database Setup (ChromaDB) ---")
try:
    # Initialize the VectorDatabase class, connecting to ChromaDB
    vector_db = VectorDatabase(persist_directory=CHROMA_PERSIST_DIR,
                               collection_name=CHROMA_COLLECTION_NAME,
                               embedding_model=EMBEDDING_MODEL)

    #vector_db.populate_collection_if_needed(
    #    target_original_docs_to_process=NUM_ROWS_TO_LOAD,
    #    data_loader_func=load_googlenq_data,
    #)
    vector_db.populate_collection_if_needed(
        target_original_docs_to_process=NUM_ROWS_TO_LOAD,
        data_loader_func=load_pubmed_data,
    )
    print("--- Vector Database Setup Complete ---")

except Exception as e:
    print(f"Failed during VectorDatabase setup or population: {e}")
    traceback.print_exc()
    print("Ensure ChromaDB is installed (`pip install chromadb rank_bm25 datasets sentence-transformers`) and dependencies are met.")
    print("Check API key (if using OpenAI embeddings) and network connection.")
    print("--- Vector Database Setup Failed ---")
    vector_db = None


# --- Main retrieval function for RAG ---
def retrieve_context(query, n_results=DEFAULT_N_RESULTS):
    """
    Retrieves relevant context from the ChromaDB vector database based on the query.
    """
    if vector_db is None:
        print("Error: VectorDatabase (ChromaDB) is not initialized. Cannot retrieve context.")
        return ""

    print(f"\nRetrieving context for query: '{query}' using ChromaDB")
    results = vector_db.search_vector(query, n_results=n_results)

    # Combine the results into a single context string, separated clearly
    context = "\n\n---\n\n".join(results)
    if not context:
        print("No relevant context found in the ChromaDB database for this query.")
    return context

# --- Example Usage (for direct execution testing) ---
if __name__ == "__main__":
    if vector_db: # Only run tests if initialization succeeded
        print("\n--- Testing vector_database.py with ChromaDB & GigaVerbo data ---")

        test_queries = [
            "What are the common side effects of metformin in the treatment of type 2 diabetes?",
            "How does the gut microbiota influence the efficacy of immunotherapy in cancer treatment?",
            "Summarize the current understanding of the role of amyloid-beta in the pathogenesis of Alzheimer's disease.",
            "What biomarkers are currently being investigated for early detection of pancreatic cancer?",
            "Compare the efficacy of mRNA vaccines versus traditional inactivated vaccines for COVID-19.",
        ]
        #dev_google_nq_questions = get_googlenq_dev_questions(5)

        for query in test_queries:
            print("-" * 40)
            print(f"Query: {query}")
            context = retrieve_context(query, n_results=3)
            print("\nRetrieved Context Snippets (from ChromaDB):")
            print(context if context else "[No context retrieved]")
            
            # Test BM25 search
            #if vector_db.bm25_index:
            #     print("\nPerforming BM25 test search...")
            #     bm25_results = vector_db.bm25_index.search_bm25(query, n_results=2)
            #     print("\nRetrieved Context Snippets (from BM25 Keyword Search):")
            #     bm25_context = "\n\n---\n\n".join(bm25_results)
            #     print(bm25_context if bm25_context else "[No context retrieved from BM25 search]")
            #else:
            #    print("\nBM25 index not available for testing.")
            #print("-" * 40)

        print("\n--- Test Complete ---")
    else:
        print("\nSkipping tests as VectorDatabase (ChromaDB) failed to initialize or populate.")