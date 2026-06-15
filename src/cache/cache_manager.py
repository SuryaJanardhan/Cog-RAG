"""
Cache management for embeddings and responses.
Supports SQLite and Redis for embedding cache, Redis/Postgres for response cache.
"""
import hashlib
import json
import sqlite3
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Dict
import redis
import psycopg2
from psycopg2.extras import Json

from ..config import settings


class EmbeddingCache(ABC):
    """Abstract base class for embedding cache."""
    
    @abstractmethod
    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        pass
    
    @abstractmethod
    def set(self, text: str, embedding: List[float]) -> None:
        """Cache embedding for text."""
        pass
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """Generate hash for text."""
        return hashlib.sha256(text.encode()).hexdigest()


class SQLiteEmbeddingCache(EmbeddingCache):
    """SQLite-based embedding cache for development."""
    
    def __init__(self):
        self.db_path = settings.embedding_cache_sqlite_path
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Create the embeddings table if it doesn't exist."""
        import os
        if os.path.dirname(self.db_path):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        print(f"Initialized SQLite embedding cache at: {self.db_path}")
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding from SQLite."""
        text_hash = self._hash_text(text)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT embedding FROM embeddings WHERE text_hash = ?", (text_hash,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return None
    
    def set(self, text: str, embedding: List[float]) -> None:
        """Cache embedding in SQLite."""
        text_hash = self._hash_text(text)
        embedding_json = json.dumps(embedding)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO embeddings (text_hash, embedding) VALUES (?, ?)",
            (text_hash, embedding_json)
        )
        conn.commit()
        conn.close()


class RedisEmbeddingCache(EmbeddingCache):
    """Redis-based embedding cache for production."""
    
    def __init__(self):
        if settings.redis_url:
            self.client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            print("Initialized Redis embedding cache from connection URL")
        else:
            self.client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password if settings.redis_password else None,
                decode_responses=True,
            )
            print("Initialized Redis embedding cache")
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding from Redis."""
        text_hash = self._hash_text(text)
        result = self.client.get(f"emb:{text_hash}")
        
        if result:
            return json.loads(result)
        return None
    
    def set(self, text: str, embedding: List[float]) -> None:
        """Cache embedding in Redis."""
        text_hash = self._hash_text(text)
        self.client.set(
            f"emb:{text_hash}",
            json.dumps(embedding),
            ex=settings.redis_ttl
        )


class ResponseCache(ABC):
    """Abstract base class for response cache."""
    
    @abstractmethod
    def get(self, query: str, doc_ids: List[str]) -> Optional[str]:
        """Get cached response for query and document IDs."""
        pass
    
    @abstractmethod
    def set(self, query: str, doc_ids: List[str], response: str) -> None:
        """Cache response for query and document IDs."""
        pass
    
    @staticmethod
    def _generate_cache_key(query: str, doc_ids: List[str]) -> str:
        """Generate cache key from query and retrieved document IDs."""
        sorted_ids = sorted(doc_ids)
        key_data = f"{query}:{':'.join(sorted_ids)}"
        return hashlib.sha256(key_data.encode()).hexdigest()


class RedisResponseCache(ResponseCache):
    """Redis-based response cache."""
    
    def __init__(self):
        if settings.redis_url:
            self.client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            print("Initialized Redis response cache from connection URL")
        else:
            self.client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password if settings.redis_password else None,
                decode_responses=True,
            )
            print("Initialized Redis response cache")
    
    def get(self, query: str, doc_ids: List[str]) -> Optional[str]:
        """Get cached response from Redis."""
        cache_key = self._generate_cache_key(query, doc_ids)
        return self.client.get(f"resp:{cache_key}")
    
    def set(self, query: str, doc_ids: List[str], response: str) -> None:
        """Cache response in Redis."""
        cache_key = self._generate_cache_key(query, doc_ids)
        self.client.set(
            f"resp:{cache_key}",
            response,
            ex=settings.redis_ttl
        )


class PostgresResponseCache(ResponseCache):
    """PostgreSQL-based response cache."""
    
    def __init__(self):
        self.conn_params = {
            "host": settings.postgres_host,
            "port": settings.postgres_port,
            "database": settings.postgres_db,
            "user": settings.postgres_user,
            "password": settings.postgres_password,
        }
        self._initialize_db()
        print("Initialized PostgreSQL response cache")
    
    def _initialize_db(self) -> None:
        """Create the responses table if it doesn't exist."""
        conn = psycopg2.connect(**self.conn_params)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                cache_key TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                doc_ids JSONB NOT NULL,
                response TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_response_cache_created 
            ON response_cache(created_at)
        """)
        conn.commit()
        cursor.close()
        conn.close()
    
    def get(self, query: str, doc_ids: List[str]) -> Optional[str]:
        """Get cached response from PostgreSQL."""
        cache_key = self._generate_cache_key(query, doc_ids)
        
        conn = psycopg2.connect(**self.conn_params)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT response FROM response_cache WHERE cache_key = %s",
            (cache_key,)
        )
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result:
            return result[0]
        return None
    
    def set(self, query: str, doc_ids: List[str], response: str) -> None:
        """Cache response in PostgreSQL."""
        cache_key = self._generate_cache_key(query, doc_ids)
        
        conn = psycopg2.connect(**self.conn_params)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO response_cache (cache_key, query, doc_ids, response)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (cache_key) DO UPDATE SET
                response = EXCLUDED.response,
                created_at = CURRENT_TIMESTAMP
            """,
            (cache_key, query, Json(doc_ids), response)
        )
        conn.commit()
        cursor.close()
        conn.close()


class SQLiteResponseCache(ResponseCache):
    """SQLite-based response cache for offline/local development."""
    def __init__(self, db_path: str = "./data/response_cache.db"):
        self.db_path = db_path
        self._initialize_db()
        
    def _initialize_db(self) -> None:
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                cache_key TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                doc_ids TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def get(self, query: str, doc_ids: List[str]) -> Optional[str]:
        cache_key = self._generate_cache_key(query, doc_ids)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT response FROM response_cache WHERE cache_key = ?", (cache_key,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]
        return None

    def set(self, query: str, doc_ids: List[str], response: str) -> None:
        cache_key = self._generate_cache_key(query, doc_ids)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO response_cache (cache_key, query, doc_ids, response) VALUES (?, ?, ?, ?)",
            (cache_key, query, json.dumps(doc_ids), response)
        )
        conn.commit()
        conn.close()


def get_embedding_cache() -> EmbeddingCache:
    """Factory function to get embedding cache based on configuration."""
    if settings.embedding_cache_type == "redis":
        try:
            return RedisEmbeddingCache()
        except Exception as e:
            print(f"[CACHE] Redis embedding cache failed to initialize: {e}. Falling back to SQLite.")
            return SQLiteEmbeddingCache()
    else:
        return SQLiteEmbeddingCache()


def get_response_cache() -> ResponseCache:
    """Factory function to get response cache based on configuration."""
    if settings.response_cache_type == "postgres":
        try:
            return PostgresResponseCache()
        except Exception as e:
            print(f"[CACHE] Postgres response cache failed to initialize: {e}. Falling back to SQLite.")
            return SQLiteResponseCache()
    elif settings.response_cache_type == "redis":
        try:
            return RedisResponseCache()
        except Exception as e:
            print(f"[CACHE] Redis response cache failed to initialize: {e}. Falling back to SQLite.")
            return SQLiteResponseCache()
    else:
        return SQLiteResponseCache()


# Global cache instances
embedding_cache: Optional[EmbeddingCache] = None
response_cache: Optional[ResponseCache] = None


def initialize_caches() -> tuple[EmbeddingCache, ResponseCache]:
    """Initialize and return global cache instances."""
    global embedding_cache, response_cache
    
    if embedding_cache is None:
        embedding_cache = get_embedding_cache()
    
    if response_cache is None:
        response_cache = get_response_cache()
    
    return embedding_cache, response_cache
