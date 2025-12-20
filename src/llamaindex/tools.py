"""
LangChain tool wrappers for LlamaIndex query engines.

Enables integration of LlamaIndex capabilities into LangGraph workflows.
"""

from typing import Optional, Dict, Any
from langchain.tools import Tool
from llama_index.core import VectorStoreIndex

from .index_manager import LlamaIndexManager
from .query_engines import CompactEngine, RouterEngine, SubQuestionEngine, HybridEngine


class LlamaIndexTools:
    """
    Wraps LlamaIndex query engines as LangChain tools.
    
    Enables LlamaIndex capabilities to be called from LangGraph nodes.
    """
    
    def __init__(self, index_manager: LlamaIndexManager):
        """
        Initialize tools.
        
        Args:
            index_manager: LlamaIndexManager instance
        """
        self.index_manager = index_manager
        self.tools = []
        self._build_tools()
    
    def _build_tools(self):
        """Build LangChain tools from query engines."""
        # Ensure index is initialized
        if self.index_manager.index is None:
            self.index_manager.create_index()
        
        # Compact query tool
        compact_engine = CompactEngine(
            self.index_manager.index,
            response_mode="compact",
        )
        
        compact_tool = Tool(
            name="llamaindex_compact_query",
            description=(
                "Query the knowledge base using LlamaIndex compact mode. "
                "Best for straightforward questions requiring synthesized answers "
                "from multiple document chunks. Returns a concise answer."
            ),
            func=lambda q: str(compact_engine.query(q)),
        )
        self.tools.append(compact_tool)
        
        # Refine query tool
        refine_engine = CompactEngine(
            self.index_manager.index,
            response_mode="refine",
        )
        
        refine_tool = Tool(
            name="llamaindex_refine_query",
            description=(
                "Query the knowledge base using LlamaIndex refine mode. "
                "Iteratively refines the answer across multiple document chunks. "
                "Best for questions requiring detailed, comprehensive answers."
            ),
            func=lambda q: str(refine_engine.query(q)),
        )
        self.tools.append(refine_tool)
        
        # Tree summarize tool
        tree_engine = CompactEngine(
            self.index_manager.index,
            response_mode="tree_summarize",
        )
        
        tree_tool = Tool(
            name="llamaindex_tree_query",
            description=(
                "Query the knowledge base using LlamaIndex tree summarize mode. "
                "Builds a tree summary from document chunks. "
                "Best for summarization tasks or broad overview questions."
            ),
            func=lambda q: str(tree_engine.query(q)),
        )
        self.tools.append(tree_tool)
    
    def get_tools(self) -> list:
        """
        Get list of LangChain tools.
        
        Returns:
            List of Tool instances
        """
        return self.tools
    
    def add_router_tool(
        self,
        indices: Dict[str, VectorStoreIndex],
        name: str = "llamaindex_router_query",
    ):
        """
        Add router query tool.
        
        Args:
            indices: Dictionary of named indices
            name: Tool name
        """
        router_engine = RouterEngine(indices)
        
        router_tool = Tool(
            name=name,
            description=(
                "Query multiple knowledge bases using intelligent routing. "
                "Automatically selects the most relevant source for the question. "
                f"Available sources: {', '.join(indices.keys())}"
            ),
            func=lambda q: str(router_engine.query(q)),
        )
        self.tools.append(router_tool)
    
    def add_subquestion_tool(
        self,
        indices: Dict[str, VectorStoreIndex],
        name: str = "llamaindex_subquestion_query",
    ):
        """
        Add sub-question query tool.
        
        Args:
            indices: Dictionary of named indices
            name: Tool name
        """
        subq_engine = SubQuestionEngine(indices)
        
        subq_tool = Tool(
            name=name,
            description=(
                "Answer complex questions by breaking them into sub-questions. "
                "Each sub-question is answered independently and results are synthesized. "
                "Best for multi-part questions or questions requiring information from multiple sources."
            ),
            func=lambda q: str(subq_engine.query(q)),
        )
        self.tools.append(subq_tool)
    
    def add_hybrid_tool(
        self,
        primary_index: VectorStoreIndex,
        secondary_indices: Optional[Dict[str, VectorStoreIndex]] = None,
        name: str = "llamaindex_hybrid_query",
    ):
        """
        Add hybrid query tool with mode selection.
        
        Args:
            primary_index: Primary VectorStoreIndex
            secondary_indices: Optional additional indices
            name: Tool name
        """
        hybrid_engine = HybridEngine(primary_index, secondary_indices)
        
        hybrid_tool = Tool(
            name=name,
            description=(
                "Query the knowledge base using automatic mode selection. "
                "Intelligently chooses between compact, router, or sub-question modes "
                "based on query complexity and structure."
            ),
            func=lambda q: str(hybrid_engine.query(q, mode="auto")),
        )
        self.tools.append(hybrid_tool)


def create_llamaindex_tool(
    index_manager: LlamaIndexManager,
    mode: str = "compact",
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Tool:
    """
    Create a single LlamaIndex tool.
    
    Args:
        index_manager: LlamaIndexManager instance
        mode: Query mode ("compact", "refine", "tree_summarize")
        name: Optional tool name
        description: Optional tool description
        
    Returns:
        LangChain Tool instance
    """
    if index_manager.index is None:
        index_manager.create_index()
    
    engine = CompactEngine(index_manager.index, response_mode=mode)
    
    default_names = {
        "compact": "llamaindex_compact_query",
        "refine": "llamaindex_refine_query",
        "tree_summarize": "llamaindex_tree_query",
    }
    
    default_descriptions = {
        "compact": "Query knowledge base with compact response synthesis",
        "refine": "Query knowledge base with iterative refinement",
        "tree_summarize": "Query knowledge base with tree summarization",
    }
    
    return Tool(
        name=name or default_names.get(mode, "llamaindex_query"),
        description=description or default_descriptions.get(mode, "Query the knowledge base"),
        func=lambda q: str(engine.query(q)),
    )
