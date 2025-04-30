# vector_database.py

import os
import numpy as np
from numpy.linalg import norm
from llm_client import openai_client # Reuse the client from llm_client.py
from datasets import load_dataset   # To load data from Hugging Face
import chromadb                     # For the persistent vector database
import uuid                         # For generating unique document IDs
import time
from rank_bm25 import BM25Okapi     # For keyword search
import re                           # For regex tokenization

# --- Configuration ---
EMBEDDING_MODEL = "BAAI/bge-en-icl" # Verify model availability on Nebius
DEFAULT_N_RESULTS = 3
DATASET_NAME = "TucanoBR/GigaVerbo"
NUM_ROWS_TO_LOAD = 10_000

# ChromaDB Configuration
CHROMA_PERSIST_DIR = "./chroma_db_gigaverbo" # Directory to store ChromaDB data
CHROMA_COLLECTION_NAME = "gigaverbo_collection"
# Batch size for adding documents to ChromaDB and embedding
BATCH_SIZE = 200 # Process these documents at a time

# --- Simple Tokenizer for BM25 ---
def simple_tokenizer(text):
    """Basic tokenizer: lowercase and split by non-alphanumeric characters."""
    if not isinstance(text, str):
        return []
    # Lowercase, then split by one or more non-alphanumeric characters
    tokens = re.split(r'\W+', text.lower())
    # Filter out empty strings that can result from multiple delimiters
    return [token for token in tokens if token]

# --- Data Loading Function (Unchanged) ---
def load_huggingface_data(dataset_name, num_rows):
    """
    Loads the specified number of rows from a Hugging Face dataset stream.
    Extracts the 'text' field.
    """
    print(f"Attempting to load first {num_rows:,} rows from '{dataset_name}'...")
    texts = []
    try:
        dataset_stream = load_dataset(dataset_name, streaming=True, split='train', trust_remote_code=True)
        start_time = time.time()
        processed_count = 0
        limited_stream = dataset_stream.take(num_rows)

        for row in limited_stream:
            if row and 'text' in row and isinstance(row['text'], str) and row['text'].strip():
                texts.append(row['text'].strip())
            processed_count += 1
            if processed_count % (num_rows // 20 or 1) == 0: # Print progress more often
                 elapsed = time.time() - start_time
                 print(f"  Loaded {processed_count: ,}/{num_rows:,} rows... ({elapsed:.1f}s elapsed)")

        end_time = time.time()
        print(f"Successfully loaded {len(texts):,} non-empty text snippets in {end_time - start_time:.2f} seconds.")
        if len(texts) < num_rows:
             print(f"  Note: Fewer rows loaded ({len(texts):,}) than requested ({num_rows:,}) likely due to empty/invalid text entries.")
        return texts
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        return None

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
        self.texts = None           # List of original text documents
        self.bm25_index = None      # BM25 index object
        print(f"Chroma Collection '{collection_name}' loaded. Current count: {self.collection.count():,} documents.")

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

    def _build_bm25(self):
        """Builds the BM25 index from self.texts."""
        if self.texts is None:
            print("Error: Cannot build BM25 index, text data not loaded.")
            return
        print(f"\nBuilding BM25 index from {len(self.texts):,} documents...")
        start_time = time.time()
        # Tokenize documents - this can consume memory
        tokenized_corpus = [simple_tokenizer(doc) for doc in self.texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        end_time = time.time()
        print(f"BM25 index built in {end_time - start_time:.2f} seconds.")

    def populate_collection_if_needed(self, num_expected_docs, data_loader_func, dataset_name):
        """Checks collection count and populates it if necessary."""
        current_count = self.collection.count()
        need_to_load_texts = self.texts is None # Need texts if not already in memory for BM25
        need_to_populate_chroma = current_count < num_expected_docs
        print(f"Checking collection '{self.collection_name}'. Found {current_count:,} documents. Expected ~{num_expected_docs:,}.")
        print(f"Need to load texts? {need_to_load_texts}. Need to populate Chroma? {need_to_populate_chroma}.")

        # --- Step 1: Load Texts if necessary ---
        if need_to_load_texts:
            print("\n--- Loading Text Data ---")
            loaded_texts = data_loader_func(dataset_name, num_expected_docs)
            if not loaded_texts:
                print("Failed to load documents. Aborting.")
                return False # Indicate failure
            self.texts = loaded_texts
            print(f"Stored {len(self.texts):,} documents in memory.")
        else:
            print("Text data already in memory or not needed for loading.")

        # --- Step 2: Populate ChromaDB if necessary ---
        if need_to_populate_chroma:
            if self.texts is None: # Should have been loaded above if needed
                 print("Error: Texts required for Chroma population are missing.")
                 return False
            print(f"\n--- Populating ChromaDB ({current_count:,} -> ~{num_expected_docs:,}) ---")
            # We assume texts are now loaded in self.texts
            total_added_chroma = 0
            population_start_time = time.time()
            docs_to_process = self.texts # Use the texts we just loaded/already had
            
            # NOTE: This simple check doesn't handle resuming population precisely.
            # A more robust approach might check IDs if possible.
            print(f"Starting ChromaDB population with {len(docs_to_process):,} docs...")

            for i in range(0, len(docs_to_process), BATCH_SIZE):
                batch_texts = docs_to_process[i:i + BATCH_SIZE]
                #print(f"Processing Chroma batch {i // BATCH_SIZE + 1}...") # Verbose
                added_count = self._add_documents_batch_chroma(batch_texts)
                total_added_chroma += added_count
                if (i // BATCH_SIZE + 1) % 10 == 0: # Progress update every 10 batches
                    elapsed = time.time() - population_start_time
                    print(f"  Chroma progress: Processed batch {i // BATCH_SIZE + 1}, Added {total_added_chroma:,} new docs... ({elapsed:.1f}s)")

            population_end_time = time.time()
            print(f"Finished ChromaDB population. Added {total_added_chroma:,} docs in {population_end_time - population_start_time:.2f}s.")
            print(f"Collection '{self.collection_name}' now contains {self.collection.count():,} documents.")
        else:
            print("ChromaDB population not needed.")

        # --- Step 3: Build BM25 index if necessary ---
        if self.bm25_index is None:
            if self.texts is None:
                print("Error: Cannot build BM25 index as texts are not available.")
                # Attempt to load texts again if they weren't loaded but chroma was populated?
                # This indicates a state inconsistency, should ideally not happen with above logic.
                return False # Indicate setup failed
            print("\n--- Building BM25 Index ---")
            self._build_bm25()
        else:
            print("BM25 index already exists.")
        
        return True # Indicate success

    def search_vector(self, query, n_results=DEFAULT_N_RESULTS):
        """Performs vector search using ChromaDB."""
        print(f"Performing vector search (ChromaDB) for: '{query[:50]}...' (k={n_results})")
        if self.collection.count() == 0: return []
        start_time = time.time()
        query_embedding_list = self._get_embedding_batch([query])
        if query_embedding_list is None or not query_embedding_list: return []
        query_embedding = query_embedding_list[0]
        try:
            results = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=n_results, include=['documents'])
            documents = results.get('documents', [[]])[0]
            print(f"Vector search completed in {time.time() - start_time:.4f}s, found {len(documents)} results.")
            return documents
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return []

    def search_bm25(self, query, n_results=DEFAULT_N_RESULTS):
        """Performs keyword search using BM25."""
        print(f"Performing keyword search (BM25) for: '{query[:50]}...' (k={n_results})")
        if self.bm25_index is None or self.texts is None:
            print("Error: BM25 index or text data not available.")
            return []
        start_time = time.time()
        tokenized_query = simple_tokenizer(query)
        if not tokenized_query: return []
        # BM25Okapi.get_top_n returns the actual documents (texts)
        try:
            # Note: get_top_n requires the original corpus (self.texts) to return documents
            # This is inefficient for large corpora but necessary for rank_bm25.
            # A production system might store doc IDs and retrieve texts separately.
            scores = self.bm25_index.get_scores(tokenized_query)
            # Get indices sorted by score, then map back to texts
            sorted_indices = np.argsort(scores)[::-1] # Get indices of scores in descending order
            top_n_indices = sorted_indices[:n_results]
            documents = [self.texts[i] for i in top_n_indices if scores[i] > 0] # Only return actual matches > 0 score
            print(f"BM25 search completed in {time.time() - start_time:.4f}s, found {len(documents)} results.")
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

    # Check if the collection needs populating and run it if necessary
    # This will load data and embed only if the DB doesn't have enough docs
    vector_db.populate_collection_if_needed(
        num_expected_docs=NUM_ROWS_TO_LOAD,
        data_loader_func=load_huggingface_data,
        dataset_name=DATASET_NAME
    )
    print("--- Vector Database Setup Complete ---")

except Exception as e:
    print(f"Failed during VectorDatabase setup or population: {e}")
    print("Ensure ChromaDB is installed (`pip install chromadb`) and dependencies are met.")
    print("Check API key and network connection.")
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
            "Qual é a importância da Amazônia para o Brasil?",
            "Quais são os principais desafios da educação em Portugal?",
            "Fale sobre a história do futebol brasileiro.",
            "O que é inteligência artificial?",
            "Explique o conceito de saudade."
        ]

        for query in test_queries:
            print("-" * 40)
            print(f"Query: {query}")
            context = retrieve_context(query, n_results=3)
            print("\nRetrieved Context Snippets (from ChromaDB):")
            print(context if context else "[No context retrieved]")
            print("-" * 40)

        print("\n--- Test Complete ---")
    else:
        print("\nSkipping tests as VectorDatabase (ChromaDB) failed to initialize or populate.")