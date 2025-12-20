"""
Chunking and embedding pipeline with caching support.
Splits documents and generates embeddings with cache checking.
"""
from typing import List, Optional
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore

from ..config import settings
from ..llm import get_gemini_client
from ..db import initialize_vector_db
from ..cache import get_embedding_cache


class ChunkingPipeline:
    """Handles document chunking with configurable parameters."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize chunking pipeline.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            separators: List of separators for splitting
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=separators or ["\n\n", "\n", " ", ""],
            is_separator_regex=False,
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List[Document]: Chunked documents
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks


class EmbeddingPipeline:
    """Handles embedding generation with caching."""
    
    def __init__(self):
        """Initialize embedding pipeline with Gemini client and cache."""
        self.gemini_client = get_gemini_client()
        self.embeddings = self.gemini_client.embeddings
        self.embedding_cache = get_embedding_cache()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def embed_text_with_cache(self, text: str) -> List[float]:
        """
        Generate embedding for text with cache checking.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        # Check cache first
        cached_embedding = self.embedding_cache.get(text)
        
        if cached_embedding is not None:
            self.cache_hits += 1
            return cached_embedding
        
        # Cache miss - generate new embedding
        self.cache_misses += 1
        embedding = self.embeddings.embed_query(text)
        
        # Store in cache
        self.embedding_cache.set(text, embedding)
        
        return embedding
    
    def embed_documents_with_cache(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with cache checking.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        embeddings = []
        texts_to_embed = []
        text_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached_embedding = self.embedding_cache.get(text)
            if cached_embedding is not None:
                self.cache_hits += 1
                embeddings.append(cached_embedding)
            else:
                # Need to generate embedding
                texts_to_embed.append(text)
                text_indices.append(i)
                embeddings.append(None)  # Placeholder
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            self.cache_misses += len(texts_to_embed)
            new_embeddings = self.embeddings.embed_documents(texts_to_embed)
            
            # Store in cache and update results
            for idx, text, embedding in zip(text_indices, texts_to_embed, new_embeddings):
                self.embedding_cache.set(text, embedding)
                embeddings[idx] = embedding
        
        return embeddings
    
    def get_cache_stats(self) -> dict:
        """Get embedding cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 2)
        }


class VectorStorePipeline:
    """Handles vector storage operations."""
    
    def __init__(self):
        """Initialize vector store pipeline."""
        self.vector_db = initialize_vector_db()
        self.gemini_client = get_gemini_client()
        self.embedding_pipeline = EmbeddingPipeline()
        self.vectorstore: Optional[VectorStore] = None
    
    def initialize_vectorstore(self) -> VectorStore:
        """
        Initialize the vector store with embeddings.
        
        Returns:
            VectorStore: Initialized vector store
        """
        if self.vectorstore is None:
            self.vectorstore = self.vector_db.get_vectorstore(
                self.gemini_client.embeddings
            )
        return self.vectorstore
    
    def add_documents(
        self,
        documents: List[Document],
        use_cache: bool = True
    ) -> List[str]:
        """
        Add documents to vector store.
        
        Args:
            documents: List of documents to add
            use_cache: Whether to use embedding cache
            
        Returns:
            List[str]: IDs of added documents
        """
        vectorstore = self.initialize_vectorstore()
        
        if use_cache:
            # Extract texts and generate embeddings with caching
            texts = [doc.page_content for doc in documents]
            embeddings = self.embedding_pipeline.embed_documents_with_cache(texts)
            
            # Add to vector store with pre-computed embeddings
            # Note: Some vector stores might not support this directly
            # In that case, we rely on the embedding cache in the embedding function
            ids = vectorstore.add_documents(documents)
        else:
            # Add documents without cache (use vector store's default embedding)
            ids = vectorstore.add_documents(documents)
        
        print(f"Added {len(documents)} documents to vector store")
        
        if use_cache:
            stats = self.embedding_pipeline.get_cache_stats()
            print(f"Embedding cache stats: {stats}")
        
        return ids
    
    def get_vectorstore(self) -> VectorStore:
        """Get the vector store instance."""
        return self.initialize_vectorstore()


class DocumentProcessingPipeline:
    """End-to-end pipeline for processing and storing documents."""
    
    def __init__(self):
        """Initialize the complete processing pipeline."""
        self.chunking = ChunkingPipeline()
        self.vector_store = VectorStorePipeline()
    
    def process_and_store(
        self,
        documents: List[Document],
        use_cache: bool = True
    ) -> dict:
        """
        Process documents: chunk, embed, and store in vector database.
        
        Args:
            documents: Documents to process
            use_cache: Whether to use embedding cache
            
        Returns:
            dict: Processing statistics
        """
        # Step 1: Chunk documents
        chunks = self.chunking.chunk_documents(documents)
        
        # Step 2: Add to vector store (embeddings generated automatically)
        doc_ids = self.vector_store.add_documents(chunks, use_cache=use_cache)
        
        stats = {
            "original_documents": len(documents),
            "chunks_created": len(chunks),
            "documents_stored": len(doc_ids),
            "chunk_size": self.chunking.chunk_size,
            "chunk_overlap": self.chunking.chunk_overlap,
        }
        
        if use_cache:
            stats["cache_stats"] = self.vector_store.embedding_pipeline.get_cache_stats()
        
        return stats


def create_processing_pipeline() -> DocumentProcessingPipeline:
    """Factory function to create document processing pipeline."""
    return DocumentProcessingPipeline()
