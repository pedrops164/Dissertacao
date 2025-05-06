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
        # Define your environment variables here with defaults
        self.prompts_lang = os.environ.get("PROMPTS_LANG", "en")
        self.self_rag_retrieval_k = int(os.environ.get("SELF_RAG_RETRIEVAL_K", 7))
        self.self_rag_final_context_n = int(os.environ.get("SELF_RAG_FINAL_CONTEXT_N", 3))
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.nebius_api_key = os.environ.get("NEBIUS_API_KEY")
        self.tavily_api_key = os.environ.get("TAVILY_API_KEY")
        
        # self rag config
        self.SELF_RAG_RETRIEVAL_K = os.environ.get("SELF_RAG_RETRIEVAL_K")
        self.SELF_RAG_FINAL_CONTEXT_N = os.environ.get("SELF_RAG_FINAL_CONTEXT_N")


        # You can add validation here if needed
        if self.prompts_lang not in ["en", "pt"]:  # Example validation
            print(f"Warning: Unsupported language '{self.prompts_lang}'. Falling back to 'en'")
            self.prompts_lang = "en"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key name."""
        return getattr(self, key, default)

# Create a singleton instance to be imported by other modules
config = Config()