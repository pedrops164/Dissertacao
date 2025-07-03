from .base_db import VectorDB
from .qdrant_db import QdrantDB
from embedding_model import GoogleEmbeddingModel

#embedding_model = GoogleEmbeddingModel()
#vector_db = QdrantDB(n_workers=4, embedding_model=embedding_model, use_quantization=True)