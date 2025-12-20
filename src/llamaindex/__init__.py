"""
LlamaIndex integration for advanced retrieval patterns.

This module provides:
- VectorStoreIndex backed by Qdrant/Chroma
- RouterQueryEngine for multi-source selection
- SubQuestionQueryEngine for query decomposition
- Tool wrappers for LangGraph integration
"""

from .index_manager import LlamaIndexManager
from .query_engines import (
    RouterEngine,
    SubQuestionEngine,
    CompactEngine,
)
from .tools import LlamaIndexTools

__all__ = [
    "LlamaIndexManager",
    "RouterEngine",
    "SubQuestionEngine",
    "CompactEngine",
    "LlamaIndexTools",
]
