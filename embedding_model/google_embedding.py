from .base_embedding import EmbeddingModel
import google.generativeai.client as gemini_client
import google.generativeai.embedding as embedding_client
from config import config

class GoogleEmbeddingModel(EmbeddingModel):
    """
    Google Embedding model implementation.
    """
    def __init__(self, embedding_model: str = "models/text-embedding-004", embedding_dim: int = 768):
        """
        Initialize the Google Embedding model with the specified model and dimensionality.

        :param embedding_model: The model to use for embeddings.
        :param embedding_dim: The dimensionality of the embeddings.
        """
        super().__init__(embedding_model, embedding_dim)

        # --- Configure the Gemini API client ---
        GEMINI_API_KEY = config.get("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        gemini_client.configure(api_key=GEMINI_API_KEY)

    def embed_documents(self, doc_texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for the given batch of texts using Google Embedding.

        :param doc_texts: The input batch of texts to embed.
        :return: A list of embeddings for each input text.
        """
        """Embeds a BATCH of documents using the Google Generative AI API."""
        result = embedding_client.embed_content(
            model=self.embedding_model,
            output_dimensionality=self.embedding_dim,
            content=doc_texts,
            task_type="retrieval_document"
        )
        return result['embedding']

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a single query using Google Embedding.

        :param query: The input query to embed.
        :return: An embedding for the input query.
        """
        result = embedding_client.embed_content(
            model=self.embedding_model,
            output_dimensionality=self.embedding_dim,
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']
    

# example embed query