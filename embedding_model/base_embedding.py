from abc import ABC, abstractmethod

class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    """
    def __init__(self, embedding_model: str, embedding_dim: int):
        """
        Initialize the embedding model with the specified model and dimensionality.

        :param embedding_model: The model to use for embeddings.
        :param embedding_dim: The dimensionality of the embeddings.
        """
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

    @abstractmethod
    def embed_documents(self, doc_texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for the given batch of texts.

        :param text: The input batch of texts to embed.
        :return: A list of embeddings for each input text.
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a single query.
        
        :param query: The input query to embed.
        :return: An embedding for the input query.
        """
        pass