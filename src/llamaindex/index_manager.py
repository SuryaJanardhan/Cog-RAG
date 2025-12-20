"""
LlamaIndex index management for advanced retrieval.
"""

import os
from typing import List, Optional
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document as LlamaDocument,
    Settings,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from qdrant_client import QdrantClient
import chromadb

from src.config.settings import get_settings


class LlamaIndexManager:
    """
    Manages LlamaIndex vector store and indexing operations.
    
    Supports both Qdrant (production) and Chroma (development) backends.
    """
    
    def __init__(self):
        """Initialize LlamaIndex with configured vector store."""
        self.settings = get_settings()
        self._setup_global_settings()
        self.vector_store = self._create_vector_store()
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.index: Optional[VectorStoreIndex] = None
        
    def _setup_global_settings(self):
        """Configure LlamaIndex global settings."""
        # Set up embedding model
        Settings.embed_model = GeminiEmbedding(
            model_name="models/embedding-001",
            api_key=self.settings.gemini_api_key,
        )
        
        # Set up LLM
        Settings.llm = Gemini(
            model=self.settings.gemini_model,
            api_key=self.settings.gemini_api_key,
            temperature=self.settings.temperature,
        )
        
        # Set up node parser
        Settings.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        
        # Set retrieval settings
        Settings.num_output = 512
        Settings.context_window = 4096
        
    def _create_vector_store(self):
        """Create vector store based on configuration."""
        if self.settings.vector_db == "qdrant":
            return self._create_qdrant_store()
        else:
            return self._create_chroma_store()
    
    def _create_qdrant_store(self) -> QdrantVectorStore:
        """Create Qdrant vector store."""
        if self.settings.environment == "production":
            # Production: Qdrant Cloud
            client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
            )
        else:
            # Development: Local Qdrant
            client = QdrantClient(path=self.settings.qdrant_path)
        
        return QdrantVectorStore(
            client=client,
            collection_name=self.settings.qdrant_collection,
        )
    
    def _create_chroma_store(self) -> ChromaVectorStore:
        """Create Chroma vector store."""
        chroma_client = chromadb.PersistentClient(
            path=self.settings.chroma_persist_dir
        )
        chroma_collection = chroma_client.get_or_create_collection(
            name=self.settings.chroma_collection
        )
        
        return ChromaVectorStore(chroma_collection=chroma_collection)
    
    def create_index(
        self,
        documents: Optional[List[LlamaDocument]] = None,
        show_progress: bool = True,
    ) -> VectorStoreIndex:
        """
        Create or load VectorStoreIndex.
        
        Args:
            documents: Optional list of documents to index
            show_progress: Whether to show indexing progress
            
        Returns:
            VectorStoreIndex instance
        """
        if documents:
            # Create new index from documents
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                show_progress=show_progress,
            )
        else:
            # Load existing index
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context,
            )
        
        return self.index
    
    def add_documents(
        self,
        documents: List[LlamaDocument],
        show_progress: bool = True,
    ) -> None:
        """
        Add documents to existing index.
        
        Args:
            documents: Documents to add
            show_progress: Whether to show progress
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index first.")
        
        for doc in documents:
            self.index.insert(doc, show_progress=show_progress)
    
    def get_retriever(self, similarity_top_k: Optional[int] = None):
        """
        Get retriever from index.
        
        Args:
            similarity_top_k: Number of top results to retrieve
            
        Returns:
            VectorIndexRetriever instance
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index first.")
        
        k = similarity_top_k or self.settings.retrieval_top_k
        return self.index.as_retriever(similarity_top_k=k)
    
    def get_query_engine(
        self,
        similarity_top_k: Optional[int] = None,
        response_mode: str = "compact",
    ):
        """
        Get query engine from index.
        
        Args:
            similarity_top_k: Number of documents to retrieve
            response_mode: Response synthesis mode
                - "compact": Concatenate chunks, refine answer
                - "refine": Iteratively refine answer
                - "tree_summarize": Build tree summary
                - "simple_summarize": Truncate chunks to fit
            
        Returns:
            QueryEngine instance
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index first.")
        
        k = similarity_top_k or self.settings.retrieval_top_k
        return self.index.as_query_engine(
            similarity_top_k=k,
            response_mode=response_mode,
        )
    
    def delete_index(self) -> None:
        """Delete the index and reset state."""
        self.index = None
        # Note: Actual vector store deletion depends on backend
        # For Qdrant: client.delete_collection(collection_name)
        # For Chroma: client.delete_collection(collection_name)
