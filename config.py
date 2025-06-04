import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize configuration values from environment variables."""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.nebius_api_key = os.environ.get("NEBIUS_API_KEY")
        self.tavily_api_key = os.environ.get("TAVILY_API_KEY")

        # llm config
        self.LLM_MODEL = os.environ.get("LLM_MODEL") # Default LLM model
        
        # rag config
        self.RAG_FINAL_CONTEXT_K = int(os.environ.get("RAG_FINAL_CONTEXT_K", 5)) # How many relevant context documents to finally use for generation
        # self rag config
        self.SELF_RAG_INITIAL_K = int(os.environ.get("SELF_RAG_INITIAL_K", 10)) # Number of docs to retrieve for self-rag
        # fusion rag config
        self.FUSION_RAG_INITIAL_K = int(os.environ.get("FUSION_RAG_INITIAL_K", 10)) # Number of docs to retrieve for fusion rag

        # worker threads
        self.n_workers = int(os.environ.get("N_WORKERS", 8))

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key name."""
        return getattr(self, key, default)

# Create a singleton instance to be imported by other modules
config = Config()