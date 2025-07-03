from .base_embedding import EmbeddingModel
import os
from openai import OpenAI # DeepInfra uses an OpenAI-compatible API

# Assuming 'config' is available from your project structure, or you rely on os.environ.
# from config import config

class DeepInfraEmbeddingModel(EmbeddingModel):
    """
    DeepInfra Embedding model implementation for Qwen/Qwen3-Embedding-0.6B.
    """
    def __init__(self, embedding_model: str = "Qwen/Qwen3-Embedding-0.6B", embedding_dim: int = 1024):
        """
        Initialize the DeepInfra Embedding model with the specified model and desired output dimensionality.

        The 'Qwen/Qwen3-Embedding-0.6B' model on DeepInfra supports user-defined output dimensions
        ranging from 32 to 1024, thanks to its Matryoshka Representation Learning (MRL) support.

        :param embedding_model: The model to use for embeddings (default: "Qwen/Qwen3-Embedding-0.6B").
        :param embedding_dim: The desired dimensionality of the output embeddings.
                              Must be between 32 and 1024 (inclusive) for this model.
        """
        super().__init__(embedding_model, embedding_dim)

        # Qwen3-Embedding-0.6B has a native dimension of 1024 and supports MRL from 32 to 1024.
        MIN_SUPPORTED_DIM = 32
        MAX_SUPPORTED_DIM = 1024

        if not (MIN_SUPPORTED_DIM <= self.embedding_dim <= MAX_SUPPORTED_DIM):
            raise ValueError(
                f"Desired embedding_dim ({self.embedding_dim}) must be between "
                f"{MIN_SUPPORTED_DIM} and {MAX_SUPPORTED_DIM} (inclusive) for "
                f"{self.embedding_model} due to its MRL support."
            )

        # --- Configure the DeepInfra client ---
        # DeepInfra uses an OpenAI-compatible API, so we use the openai client.
        DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")
        if not DEEPINFRA_API_KEY:
            raise ValueError("DEEPINFRA_API_KEY environment variable not found.")

        self.client = OpenAI(
            api_key=DEEPINFRA_API_KEY,
            base_url="https://api.deepinfra.com/v1/openai",
        )

    def embed_documents(self, doc_texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for the given batch of texts using DeepInfra Embedding.

        The API call will directly request the specified output_dimensionality.

        :param doc_texts: The input batch of texts to embed.
        :return: A list of embeddings for each input text, with the desired dimension.
        """
        if not doc_texts:
            return []

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=doc_texts,
            dimensions=self.embedding_dim, # Request the desired dimension directly from the API
            encoding_format="float"
        )

        # Extract embeddings
        embeddings = [data_item.embedding for data_item in response.data]

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a single query using DeepInfra Embedding.

        The API call will directly request the specified output_dimensionality.

        :param query: The input query to embed.
        :return: An embedding for the input query, with the desired dimension.
        """
        if not query:
            return []

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=query,
            dimensions=self.embedding_dim, # Request the desired dimension directly from the API
            encoding_format="float"
        )

        # Extract the single embedding
        if response.data and len(response.data) > 0:
            return response.data[0].embedding
        return []

