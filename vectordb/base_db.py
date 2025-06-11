from abc import ABC, abstractmethod
from embedding_model import EmbeddingModel

class VectorDB(ABC):
    def __init__(self, embedding_model: EmbeddingModel, n_workers: int = 4):
        self.embedding_model = embedding_model
        self.n_workers = n_workers
    
    @abstractmethod
    def populate_data(self, target_docs_to_process: int, data_loader_func):
        pass

    @abstractmethod
    def retrieve_context(self, query: str, n_results: int) -> list[str]:
        pass
