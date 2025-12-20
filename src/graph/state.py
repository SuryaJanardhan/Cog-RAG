"""
Graph state definition for LangGraph agentic RAG.
Defines the state structure that flows through the graph nodes.
"""
from typing import List, Optional, TypedDict, Annotated
from operator import add
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    State for the agentic RAG graph.
    
    Attributes:
        question: The original user question
        documents: Retrieved documents (if any)
        answer: Generated answer
        retry_count: Number of query rewrites attempted
        needs_retrieval: Whether retrieval is needed
        retrieval_attempted: Whether retrieval has been tried
        use_tools: Whether to use external tools
        tool_results: Results from tool calls
        error: Any error that occurred
    """
    question: str
    documents: Annotated[List[Document], add]  # Supports appending
    answer: Optional[str]
    retry_count: int
    needs_retrieval: bool
    retrieval_attempted: bool
    use_tools: bool
    tool_results: Optional[str]
    error: Optional[str]


def create_initial_state(question: str) -> GraphState:
    """
    Create initial graph state for a question.
    
    Args:
        question: User question
        
    Returns:
        GraphState: Initialized state
    """
    return GraphState(
        question=question,
        documents=[],
        answer=None,
        retry_count=0,
        needs_retrieval=False,
        retrieval_attempted=False,
        use_tools=False,
        tool_results=None,
        error=None
    )
