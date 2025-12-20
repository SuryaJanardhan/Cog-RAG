"""
Configuration settings for RAG application.
Manages all environment variables and application settings.
"""
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Environment
    environment: Literal["dev", "prod"] = Field(default="dev")
    
    # Gemini Configuration
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-1.5-flash")
    gemini_temperature: float = Field(default=0.7)
    gemini_max_output_tokens: int = Field(default=2048)
    
    # Vector Database
    vector_db: Literal["chroma", "qdrant"] = Field(default="chroma")
    
    # Qdrant Configuration
    qdrant_url: str = Field(default="")
    qdrant_api_key: str = Field(default="")
    qdrant_collection_name: str = Field(default="rag_documents")
    
    # Chroma Configuration
    chroma_persist_directory: str = Field(default="./data/chroma_db")
    
    # Embedding Configuration
    embedding_model: str = Field(default="models/embedding-001")
    embedding_dimension: int = Field(default=768)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    
    # Cache Configuration
    embedding_cache_type: Literal["sqlite", "redis"] = Field(default="sqlite")
    embedding_cache_sqlite_path: str = Field(default="./cache/embeddings.db")
    
    response_cache_type: Literal["redis", "postgres"] = Field(default="redis")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: str = Field(default="")
    redis_ttl: int = Field(default=3600)
    
    # PostgreSQL Configuration
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="rag_cache")
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(default="")
    
    # Retrieval Configuration
    retrieval_top_k: int = Field(default=5)
    retrieval_score_threshold: float = Field(default=0.7)
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=True)
    
    # Logging
    log_level: str = Field(default="INFO")
    
    # Tool Configuration (Phase 2)
    tavily_api_key: str = Field(default="")
    enable_web_search: bool = Field(default=False)
    
    @property
    def is_dev(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "dev"
    
    @property
    def is_prod(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "prod"
    
    @property
    def postgres_url(self) -> str:
        """Generate PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


# Global settings instance
settings = Settings()
