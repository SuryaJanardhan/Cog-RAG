"""
Retrieval module with caching and configurable parameters.
Implements vector-based retrieval with score thresholding.
"""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from ..config import settings
from ..processing import VectorStorePipeline


class CachedRetriever:
    """Retriever with configurable parameters and built-in caching."""
    
    def __init__(
        self,
        vectorstore: Optional[VectorStore] = None,
        k: int = None,
        score_threshold: float = None,
        search_type: str = "similarity"
    ):
        """
        Initialize retriever.
        
        Args:
            vectorstore: Vector store to use for retrieval
            k: Number of documents to retrieve
            score_threshold: Minimum relevance score (0-1)
            search_type: Type of search ("similarity" or "mmr")
        """
        self.k = k or settings.retrieval_top_k
        self.score_threshold = score_threshold or settings.retrieval_score_threshold
        self.search_type = search_type
        
        # Initialize vector store if not provided
        if vectorstore is None:
            vector_pipeline = VectorStorePipeline()
            self.vectorstore = vector_pipeline.initialize_vectorstore()
        else:
            self.vectorstore = vectorstore
        
        # Create LangChain retriever
        self.retriever = self._create_retriever()
    
    def _create_retriever(self) -> BaseRetriever:
        """Create LangChain retriever with configured parameters."""
        search_kwargs = {
            "k": self.k,
        }
        
        # Add score threshold if using similarity search
        if self.search_type == "similarity":
            search_kwargs["score_threshold"] = self.score_threshold
        
        return self.vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs=search_kwargs
        )
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            
        Returns:
            List[Document]: Retrieved documents
        """
        documents = self.retriever.invoke(query)
        print(f"Retrieved {len(documents)} documents for query")
        return documents
    
    def retrieve_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """
        Retrieve documents with relevance scores.
        
        Args:
            query: Search query
            
        Returns:
            List[tuple[Document, float]]: Documents with scores
        """
        results = self.vectorstore.similarity_search_with_score(query, k=self.k)
        
        # Filter by score threshold
        filtered_results = [
            (doc, score) for doc, score in results
            if score >= self.score_threshold
        ]
        
        print(f"Retrieved {len(filtered_results)} documents (filtered from {len(results)})")
        return filtered_results
    
    def get_document_ids(self, documents: List[Document]) -> List[str]:
        """
        Extract unique identifiers from documents for cache keys.
        
        Args:
            documents: List of documents
            
        Returns:
            List[str]: Document IDs
        """
        doc_ids = []
        for doc in documents:
            # Try to get a unique ID from metadata
            doc_id = doc.metadata.get("id") or doc.metadata.get("source", "unknown")
            # Add page number if available
            if "page" in doc.metadata:
                doc_id = f"{doc_id}_page_{doc.metadata['page']}"
            doc_ids.append(str(hash(doc_id)))
        
        return doc_ids
    
    def format_documents(self, documents: List[Document]) -> str:
        """
        Format documents into a single context string.
        
        Args:
            documents: List of documents
            
        Returns:
            str: Formatted context
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f" (Page {page})" if page else ""
            
            context_parts.append(
                f"[Document {i}] Source: {source}{page_info}\n"
                f"{doc.page_content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def get_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract source information from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            List[Dict]: Source metadata
        """
        sources = []
        for doc in documents:
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "type": doc.metadata.get("type", "Unknown"),
            }
            
            # Add optional fields if present
            if "page" in doc.metadata:
                source_info["page"] = doc.metadata["page"]
            if "filename" in doc.metadata:
                source_info["filename"] = doc.metadata["filename"]
            if "url" in doc.metadata:
                source_info["url"] = doc.metadata["url"]
            
            sources.append(source_info)
        
        return sources


class RetrievalManager:
    """Manages retrieval operations with multiple retriever configurations."""
    
    def __init__(self):
        """Initialize retrieval manager."""
        self.default_retriever = CachedRetriever()
        self.custom_retrievers: Dict[str, CachedRetriever] = {}
    
    def create_retriever(
        self,
        name: str,
        k: int = None,
        score_threshold: float = None,
        search_type: str = "similarity"
    ) -> CachedRetriever:
        """
        Create and register a custom retriever.
        
        Args:
            name: Name for the retriever
            k: Number of documents to retrieve
            score_threshold: Minimum relevance score
            search_type: Type of search
            
        Returns:
            CachedRetriever: Configured retriever
        """
        retriever = CachedRetriever(
            k=k,
            score_threshold=score_threshold,
            search_type=search_type
        )
        self.custom_retrievers[name] = retriever
        return retriever
    
    def get_retriever(self, name: Optional[str] = None) -> CachedRetriever:
        """
        Get a retriever by name or return default.
        
        Args:
            name: Name of the retriever
            
        Returns:
            CachedRetriever: Retrieved or default retriever
        """
        if name and name in self.custom_retrievers:
            return self.custom_retrievers[name]
        return self.default_retriever
    
    def retrieve(
        self,
        query: str,
        retriever_name: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve documents using specified or default retriever.
        
        Args:
            query: Search query
            retriever_name: Name of retriever to use
            
        Returns:
            List[Document]: Retrieved documents
        """
        retriever = self.get_retriever(retriever_name)
        return retriever.retrieve(query)


def create_retriever(
    k: int = None,
    score_threshold: float = None
) -> CachedRetriever:
    """
    Factory function to create a retriever.
    
    Args:
        k: Number of documents to retrieve
        score_threshold: Minimum relevance score
        
    Returns:
        CachedRetriever: Configured retriever
    """
    return CachedRetriever(k=k, score_threshold=score_threshold)
