"""
Configuration management for TokuTel RAG Assistant.
Supports environment variables and sensible defaults.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Configuration for embedding and LLM models."""
    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Reranker model (cross-encoder)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True
    
    # LLM settings
    llm_provider: str = "openai"  # "openai", "anthropic", or "offline"
    llm_model: str = "gpt-4o-mini"  # Cost-effective and fast
    llm_temperature: float = 0.1  # Low temperature for factual answers
    llm_max_tokens: int = 500
    
    # API keys (from environment)
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))


@dataclass
class RetrievalConfig:
    """Configuration for retrieval settings."""
    # Hybrid search weights
    semantic_weight: float = 0.6
    bm25_weight: float = 0.4  # Increased for better keyword matching
    
    # Retrieval parameters
    initial_k: int = 20  # Retrieve more, then rerank
    final_k: int = 6  # Return top-k after reranking
    relevance_threshold: float = 0.05  # Very low threshold for better recall
    
    # Query expansion
    use_query_expansion: bool = True
    max_expansion_terms: int = 3


@dataclass 
class CacheConfig:
    """Configuration for caching."""
    enable_embedding_cache: bool = True
    enable_query_cache: bool = True
    cache_dir: str = ".cache"
    cache_ttl_seconds: int = 3600  # 1 hour


@dataclass
class PolicyConfig:
    """Configuration for policy enforcement."""
    mask_pii: bool = True
    enable_escalation_detection: bool = True
    default_customer_type: str = "standard"  # "standard" or "enterprise"


@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    
    # Data paths
    data_dir: str = "data"
    plans_file: str = "data/plans.csv"
    kb_file: str = "data/kb.yaml"
    transcripts_file: str = "data/transcripts.json"
    faq_file: str = "data/faq.jsonl"
    eval_prompts_file: str = "data/eval_prompts.txt"
    
    # Output paths
    output_dir: str = "."
    eval_output_json: str = "evaluation_outputs.json"
    eval_output_txt: str = "evaluation_outputs.txt"
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if os.getenv("LLM_PROVIDER"):
            config.model.llm_provider = os.getenv("LLM_PROVIDER")
        if os.getenv("LLM_MODEL"):
            config.model.llm_model = os.getenv("LLM_MODEL")
        if os.getenv("USE_RERANKER"):
            config.model.use_reranker = os.getenv("USE_RERANKER", "true").lower() == "true"
        if os.getenv("SEMANTIC_WEIGHT"):
            config.retrieval.semantic_weight = float(os.getenv("SEMANTIC_WEIGHT"))
        if os.getenv("BM25_WEIGHT"):
            config.retrieval.bm25_weight = float(os.getenv("BM25_WEIGHT"))
            
        return config
    
    def is_llm_available(self) -> bool:
        """Check if LLM API is available."""
        if self.model.llm_provider == "offline":
            return False
        if self.model.llm_provider == "openai" and self.model.openai_api_key:
            return True
        if self.model.llm_provider == "anthropic" and self.model.anthropic_api_key:
            return True
        return False


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config):
    """Set global config instance."""
    global _config
    _config = config
