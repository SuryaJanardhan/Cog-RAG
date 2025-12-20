"""
RAG Chain implementation with prompt templates and response caching.
Implements the core RAG pipeline: retrieval -> context injection -> generation.
"""
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from ..config import settings
from ..llm import get_gemini_client
from ..retrieval import CachedRetriever, create_retriever
from ..cache import get_response_cache


# Default RAG prompt template
DEFAULT_RAG_TEMPLATE = """You are a helpful AI assistant. Use the following context to answer the user's question accurately and concisely.

If the context doesn't contain relevant information to answer the question, say "I don't have enough information in the provided context to answer that question."

Context:
{context}

Question: {question}

Answer:"""


class RAGChain:
    """RAG chain with retrieval, context injection, and generation."""
    
    def __init__(
        self,
        retriever: Optional[CachedRetriever] = None,
        prompt_template: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize RAG chain.
        
        Args:
            retriever: Document retriever (created if not provided)
            prompt_template: Custom prompt template
            use_cache: Whether to use response caching
        """
        # Initialize components
        self.retriever = retriever or create_retriever()
        self.llm_client = get_gemini_client()
        self.llm = self.llm_client.chat_model
        self.use_cache = use_cache
        
        if use_cache:
            self.response_cache = get_response_cache()
        
        # Set up prompt template
        self.prompt_template = prompt_template or DEFAULT_RAG_TEMPLATE
        self.prompt = PromptTemplate.from_template(self.prompt_template)
        
        # Build the chain
        self.chain = self._build_chain()
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _build_chain(self):
        """Build the LangChain RAG chain."""
        # Create the chain: retrieve -> format context -> prompt -> LLM -> parse
        chain = (
            RunnableParallel(
                {
                    "context": lambda x: self.retriever.format_documents(
                        self.retriever.retrieve(x["question"])
                    ),
                    "question": lambda x: x["question"]
                }
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def _check_cache(self, query: str, doc_ids: List[str]) -> Optional[str]:
        """
        Check response cache for a previous answer.
        
        Args:
            query: User query
            doc_ids: Retrieved document IDs
            
        Returns:
            Optional[str]: Cached response or None
        """
        if not self.use_cache:
            return None
        
        cached_response = self.response_cache.get(query, doc_ids)
        if cached_response:
            self.cache_hits += 1
            print("✓ Cache hit - returning cached response")
        return cached_response
    
    def _store_in_cache(self, query: str, doc_ids: List[str], response: str) -> None:
        """
        Store response in cache.
        
        Args:
            query: User query
            doc_ids: Retrieved document IDs
            response: Generated response
        """
        if self.use_cache:
            self.response_cache.set(query, doc_ids, response)
    
    def query(
        self,
        question: str,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Execute RAG query with caching.
        
        Args:
            question: User question
            return_sources: Whether to include source documents
            
        Returns:
            Dict with answer and optional sources
        """
        # Step 1: Retrieve documents
        documents = self.retriever.retrieve(question)
        
        # Step 2: Get document IDs for cache key
        doc_ids = self.retriever.get_document_ids(documents)
        
        # Step 3: Check cache
        cached_answer = self._check_cache(question, doc_ids)
        
        if cached_answer:
            answer = cached_answer
        else:
            # Cache miss - generate new answer
            self.cache_misses += 1
            print("✗ Cache miss - generating new response")
            
            # Step 4: Generate answer using the chain
            answer = self.chain.invoke({"question": question})
            
            # Step 5: Store in cache
            self._store_in_cache(question, doc_ids, answer)
        
        # Prepare response
        result = {"answer": answer}
        
        if return_sources:
            result["sources"] = self.retriever.get_sources(documents)
            result["num_sources"] = len(documents)
        
        return result
    
    def batch_query(
        self,
        questions: List[str],
        return_sources: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple RAG queries.
        
        Args:
            questions: List of questions
            return_sources: Whether to include sources
            
        Returns:
            List of results
        """
        results = []
        for question in questions:
            result = self.query(question, return_sources=return_sources)
            results.append(result)
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_queries": total,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_enabled": self.use_cache
        }


class RAGPipeline:
    """Complete RAG pipeline with ingestion, processing, and querying."""
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize RAG pipeline.
        
        Args:
            use_cache: Whether to enable caching
        """
        self.rag_chain = RAGChain(use_cache=use_cache)
    
    def query(
        self,
        question: str,
        user_id: Optional[str] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            user_id: Optional user identifier
            return_sources: Whether to include sources
            
        Returns:
            Dict with answer, sources, and metadata
        """
        result = self.rag_chain.query(question, return_sources=return_sources)
        
        # Add metadata
        result["metadata"] = {
            "user_id": user_id,
            "cache_stats": self.rag_chain.get_cache_stats()
        }
        
        return result
    
    def create_custom_chain(
        self,
        prompt_template: str,
        retriever: Optional[CachedRetriever] = None
    ) -> RAGChain:
        """
        Create a custom RAG chain with specific prompt.
        
        Args:
            prompt_template: Custom prompt template
            retriever: Optional custom retriever
            
        Returns:
            RAGChain: Configured chain
        """
        return RAGChain(
            retriever=retriever,
            prompt_template=prompt_template,
            use_cache=self.rag_chain.use_cache
        )


def create_rag_chain(
    use_cache: bool = True,
    prompt_template: Optional[str] = None
) -> RAGChain:
    """
    Factory function to create a RAG chain.
    
    Args:
        use_cache: Whether to enable caching
        prompt_template: Custom prompt template
        
    Returns:
        RAGChain: Configured RAG chain
    """
    return RAGChain(use_cache=use_cache, prompt_template=prompt_template)


def create_rag_pipeline(use_cache: bool = True) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline.
    
    Args:
        use_cache: Whether to enable caching
        
    Returns:
        RAGPipeline: Configured pipeline
    """
    return RAGPipeline(use_cache=use_cache)
