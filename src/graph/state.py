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
    
    # Phase 2 Agentic extensions
    plan: Optional[List[str]]
    current_step_idx: int
    cache_hit: bool
    human_approval_required: bool
    human_approved: bool


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
        error=None,
        plan=None,
        current_step_idx=0,
        cache_hit=False,
        human_approval_required=False,
        human_approved=False
    )
