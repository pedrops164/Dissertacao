import os
from typing import Any
from dotenv import load_dotenv
import json

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
        self.GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

        # llm config
        self.JUDGE_LLM_MODEL = os.environ.get("JUDGE_LLM_MODEL") # Model for judge open question evaluation
        self.LLM_LIST = json.loads(os.environ.get("LLM_LIST"))

        # vector database config
        self.BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "100")) # Batch size for vector database operations
        
        # rag config
        self.RAG_K_LIST = json.loads(os.environ.get("RAG_K_LIST"))

        # benchmarking config
        self.EVAL_N_QUESTIONS = int(os.environ.get("EVAL_N_QUESTIONS", 100))
        self.OUTPUT_DIR = os.environ.get("OUTPUT_DIR")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key name."""
        return getattr(self, key, default)

# Create a singleton instance to be imported by other modules
config = Config()