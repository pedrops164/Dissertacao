import argparse
from vectordb.qdrant_db import QdrantDB
from dataset_loaders import load_pubmed_data
from embedding_model import GoogleEmbeddingModel, NebiusEmbeddingModel, DeepInfraEmbeddingModel
import time

def main():
    """
    Main function to parse arguments and populate the selected vector database.
    """
    parser = argparse.ArgumentParser(
        description="Populate a vector database with a specified number of documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--num-docs",
        type=int,
        required=True,
        help="The total number of documents to populate the database with."
    )
    parser.add_argument(
        "--db-type",
        type=str,
        default="qdrant",
        choices=["qdrant"],
        help="The type of vector database to use."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pubmed",
        choices=["pubmed"],
        help="The dataset to use for population."
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-004",
        choices=["text-embedding-004", "Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-Embedding-0.6B"],
        help="The embedding model to use for population."
    )
    parser.add_argument(
        "--use-quantization",
        action='store_true',
        help="Enable quantization in the vector database (if supported)."
    )
    parser.add_argument(
        '--n-workers',
        type=int,
        default=8,
        help='Number of worker threads for parallel processing.'
    )

    args = parser.parse_args()

    print("Starting database population with the following settings:")
    print(f"  - Database Type: {args.db_type}")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Embedding Model: {args.embedding_model}")
    print(f"  - Target Documents: {args.num_docs:,}")
    print(f"  - Quantization: {'Enabled' if args.use_quantization else 'Disabled'}")
    print(f"  - Worker Threads: {args.n_workers}")
    print("-" * 30)

    # --- Select data loader based on arguments ---
    if args.dataset == "pubmed":
        data_loader_func = load_pubmed_data
    else:
        # This block can be expanded if you add more datasets
        raise ValueError(f"Unsupported dataset: {args.dataset}")

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
    #import sys
    #sys.exit(0)  # Exit early for testing purposes

    # --- Initialize and populate the database based on arguments ---
    start = time.time()
    if args.db_type == "qdrant":
        db_instance = QdrantDB(
            embedding_model=embedding_model,
            n_workers=args.n_workers,
            use_quantization=args.use_quantization
        )
        db_instance.populate_data(
            target_docs_to_process=args.num_docs,
            data_loader_func=data_loader_func
        )
    else:
        # This block can be expanded if you add more database types
        raise ValueError(f"Unsupported database type: {args.db_type}")
    end = time.time()

    print(f"Database populated successfully in {end - start:.2f} seconds.")

if __name__ == "__main__":
    main()