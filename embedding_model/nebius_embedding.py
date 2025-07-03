from .base_embedding import EmbeddingModel
import os
from openai import OpenAI # Nebius AI Studio uses an OpenAI-compatible API

from config import config 

class NebiusEmbeddingModel(EmbeddingModel):
    """
    Nebius Embedding model implementation for Qwen/Qwen3-Embedding-8B.
    """
    def __init__(self, embedding_model: str = "Qwen/Qwen3-Embedding-8B", embedding_dim: int = 768):
        """
        Initialize the Nebius Embedding model with the specified model and desired output dimensionality.

        Note: The 'Qwen/Qwen3-Embedding-8B' model on Nebius typically outputs 4096 dimensions.
        If 'embedding_dim' is set to a value less than 4096, the output embeddings will be
        truncated to the specified dimension. This can save storage but might impact
        the semantic quality of the embeddings if the model is not specifically designed
        for such truncation (e.g., Matryoshka embeddings).

        :param embedding_model: The model to use for embeddings (default: "Qwen/Qwen3-Embedding-8B").
        :param embedding_dim: The desired dimensionality of the output embeddings.
                              Must be <= 4096 for this model.
        """
        super().__init__(embedding_model, embedding_dim)

        # Ensure the desired embedding_dim is not greater than the model's native dimension
        if self.embedding_dim > 4096:
            raise ValueError(
                f"Desired embedding_dim ({self.embedding_dim}) cannot be greater than "
                f"the native model dimension (4096) for {self.embedding_model}."
            )

        # --- Configure the Nebius AI Studio client ---
        # Nebius AI Studio uses an OpenAI-compatible API, so we use the openai client.
        NEBIUS_API_KEY = config.get("nebius_api_key")
        if not NEBIUS_API_KEY:
            raise ValueError("NEBIUS_API_KEY environment variable not found.")
        
        self.client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=NEBIUS_API_KEY
        )

    def _truncate_embedding(self, embedding: list[float]) -> list[float]:
        """
        Helper method to truncate an embedding to the desired dimension.
        """
        if len(embedding) > self.embedding_dim:
            return embedding[:self.embedding_dim]
        return embedding # Return as is if already smaller or equal

    def embed_documents(self, doc_texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for the given batch of texts using Nebius Embedding.

        :param doc_texts: The input batch of texts to embed.
        :return: A list of embeddings for each input text, truncated to desired dimension.
        """
        if not doc_texts:
            return []

        # Nebius API expects a single string or a list of strings for 'input'
        # It handles batching internally based on its limits.
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=doc_texts,
        )
        
        # Extract embeddings and truncate each one
        embeddings = []
        for data_item in response.data:
            truncated_embedding = self._truncate_embedding(data_item.embedding)
            embeddings.append(truncated_embedding)
            
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a single query using Nebius Embedding.

        :param query: The input query to embed.
        :return: An embedding for the input query, truncated to desired dimension.
        """
        if not query:
            return []

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=query,
        )
        
        # Extract the single embedding and truncate it
        if response.data and len(response.data) > 0:
            truncated_embedding = self._truncate_embedding(response.data[0].embedding)
            return truncated_embedding
        return []

