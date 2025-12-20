"""
Advanced query engines for LlamaIndex.
"""

from typing import List, Optional, Dict, Any
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import (
    RouterQueryEngine,
    SubQuestionQueryEngine,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.response_synthesizers import ResponseMode

from src.config.settings import get_settings


class RouterEngine:
    """
    Router query engine for multi-source selection.
    
    Routes queries to the most appropriate index/tool based on query content.
    """
    
    def __init__(self, indices: Dict[str, VectorStoreIndex]):
        """
        Initialize router engine.
        
        Args:
            indices: Dictionary mapping source names to VectorStoreIndex instances
                    Example: {"technical": tech_index, "general": gen_index}
        """
        self.settings = get_settings()
        self.indices = indices
        self.engine = self._build_router()
    
    def _build_router(self) -> RouterQueryEngine:
        """Build router query engine from indices."""
        # Create query engine tools from indices
        tools = []
        for name, index in self.indices.items():
            query_engine = index.as_query_engine(
                similarity_top_k=self.settings.retrieval_top_k,
                response_mode="compact",
            )
            
            tool = QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name=name,
                    description=f"Query engine for {name} documents",
                ),
            )
            tools.append(tool)
        
        # Build router with LLM selector
        return RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=tools,
        )
    
    def query(self, query_str: str) -> Any:
        """
        Execute query with routing.
        
        Args:
            query_str: Query string
            
        Returns:
            Query response
        """
        return self.engine.query(query_str)
    
    def add_source(self, name: str, index: VectorStoreIndex) -> None:
        """
        Add a new source to router.
        
        Args:
            name: Source name
            index: VectorStoreIndex for source
        """
        self.indices[name] = index
        self.engine = self._build_router()


class SubQuestionEngine:
    """
    Sub-question query engine for complex query decomposition.
    
    Breaks down complex queries into sub-questions and synthesizes answers.
    """
    
    def __init__(self, indices: Dict[str, VectorStoreIndex]):
        """
        Initialize sub-question engine.
        
        Args:
            indices: Dictionary mapping source names to VectorStoreIndex instances
        """
        self.settings = get_settings()
        self.indices = indices
        self.engine = self._build_engine()
    
    def _build_engine(self) -> SubQuestionQueryEngine:
        """Build sub-question query engine."""
        # Create query engine tools
        tools = []
        for name, index in self.indices.items():
            query_engine = index.as_query_engine(
                similarity_top_k=self.settings.retrieval_top_k,
            )
            
            tool = QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name=name,
                    description=f"Useful for answering questions about {name}",
                ),
            )
            tools.append(tool)
        
        # Build sub-question engine
        return SubQuestionQueryEngine.from_defaults(
            query_engine_tools=tools,
            use_async=False,  # Set to True for async execution
        )
    
    def query(self, query_str: str) -> Any:
        """
        Execute query with sub-question decomposition.
        
        Args:
            query_str: Complex query string
            
        Returns:
            Synthesized response
        """
        return self.engine.query(query_str)


class CompactEngine:
    """
    Compact query engine with response synthesis.
    
    Simple but effective query engine with configurable response modes.
    """
    
    def __init__(
        self,
        index: VectorStoreIndex,
        response_mode: str = "compact",
        similarity_top_k: Optional[int] = None,
    ):
        """
        Initialize compact engine.
        
        Args:
            index: VectorStoreIndex to query
            response_mode: Response synthesis mode
                - "compact": Concatenate chunks, refine (default)
                - "refine": Iteratively refine answer
                - "tree_summarize": Build tree summary
                - "simple_summarize": Truncate to fit
            similarity_top_k: Number of top results
        """
        self.settings = get_settings()
        self.index = index
        self.response_mode = response_mode
        self.similarity_top_k = similarity_top_k or self.settings.retrieval_top_k
        self.engine = self._build_engine()
    
    def _build_engine(self):
        """Build query engine."""
        return self.index.as_query_engine(
            similarity_top_k=self.similarity_top_k,
            response_mode=self.response_mode,
        )
    
    def query(self, query_str: str) -> Any:
        """
        Execute query.
        
        Args:
            query_str: Query string
            
        Returns:
            Query response
        """
        return self.engine.query(query_str)
    
    def get_retriever(self):
        """Get retriever from index."""
        return self.index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )


class HybridEngine:
    """
    Hybrid query engine combining multiple strategies.
    
    Can switch between compact, router, and sub-question modes.
    """
    
    def __init__(
        self,
        primary_index: VectorStoreIndex,
        secondary_indices: Optional[Dict[str, VectorStoreIndex]] = None,
    ):
        """
        Initialize hybrid engine.
        
        Args:
            primary_index: Main VectorStoreIndex
            secondary_indices: Optional additional indices for routing
        """
        self.settings = get_settings()
        self.primary_index = primary_index
        self.secondary_indices = secondary_indices or {}
        
        # Initialize engines
        self.compact_engine = CompactEngine(primary_index)
        
        if self.secondary_indices:
            all_indices = {"primary": primary_index, **self.secondary_indices}
            self.router_engine = RouterEngine(all_indices)
            self.subq_engine = SubQuestionEngine(all_indices)
        else:
            self.router_engine = None
            self.subq_engine = None
    
    def query(
        self,
        query_str: str,
        mode: str = "auto",
    ) -> Any:
        """
        Execute query with specified mode.
        
        Args:
            query_str: Query string
            mode: Query mode
                - "auto": Automatically select best mode
                - "compact": Use compact engine
                - "router": Use router engine (requires secondary indices)
                - "subquestion": Use sub-question engine (requires secondary indices)
                
        Returns:
            Query response
        """
        if mode == "auto":
            mode = self._select_mode(query_str)
        
        if mode == "compact":
            return self.compact_engine.query(query_str)
        elif mode == "router" and self.router_engine:
            return self.router_engine.query(query_str)
        elif mode == "subquestion" and self.subq_engine:
            return self.subq_engine.query(query_str)
        else:
            # Fallback to compact
            return self.compact_engine.query(query_str)
    
    def _select_mode(self, query_str: str) -> str:
        """
        Automatically select best query mode.
        
        Args:
            query_str: Query string
            
        Returns:
            Selected mode
        """
        # Simple heuristics (can be improved with LLM classification)
        query_lower = query_str.lower()
        
        # Check for multi-part questions
        if any(word in query_lower for word in ["and", "also", "compare", "difference"]):
            if self.subq_engine:
                return "subquestion"
        
        # Check for specific topics (if secondary indices exist)
        if self.router_engine and len(self.secondary_indices) > 0:
            return "router"
        
        # Default to compact
        return "compact"
