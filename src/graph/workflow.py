"""
LangGraph workflow builder for agentic RAG.
Defines the graph structure with conditional edges and routing.
"""
from typing import Literal, Optional
from langgraph.graph import StateGraph, END

from .state import GraphState, create_initial_state
from .nodes import create_nodes, MAX_RETRIES


class AgenticRAGGraph:
    """Agentic RAG workflow using LangGraph."""
    
    def __init__(self, use_llamaindex: bool = False):
        """
        Initialize the graph with nodes.
        
        Args:
            use_llamaindex: Whether to use LlamaIndex for retrieval
        """
        self.nodes = create_nodes(use_llamaindex=use_llamaindex)
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Graph structure:
        START -> classify_or_answer
          ├─> (needs retrieval) -> retrieve -> grade_documents
          │                         ├─> (relevant) -> generate_answer -> END
          │                         └─> (not relevant) -> rewrite_question -> classify_or_answer
          ├─> (needs tools) -> call_tools -> generate_answer -> END
          └─> (direct answer) -> generate_answer -> END
        """
        # Create graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("classify_or_answer", self.nodes.classify_or_answer)
        workflow.add_node("retrieve", self.nodes.retrieve)
        workflow.add_node("grade_documents", self.nodes.grade_documents)
        workflow.add_node("rewrite_question", self.nodes.rewrite_question)
        workflow.add_node("generate_answer", self.nodes.generate_answer)
        workflow.add_node("call_tools", self.nodes.call_tools)
        
        # Set entry point
        workflow.set_entry_point("classify_or_answer")
        
        # Add conditional edges from classify_or_answer
        workflow.add_conditional_edges(
            "classify_or_answer",
            self._route_after_classification,
            {
                "retrieve": "retrieve",
                "tools": "call_tools",
                "answer": "generate_answer"
            }
        )
        
        # Add edge from retrieve to grade
        workflow.add_edge("retrieve", "grade_documents")
        
        # Add conditional edges from grade_documents
        workflow.add_conditional_edges(
            "grade_documents",
            self._route_after_grading,
            {
                "generate": "generate_answer",
                "rewrite": "rewrite_question",
                "answer": "generate_answer"
            }
        )
        
        # Add conditional edges from rewrite_question
        workflow.add_conditional_edges(
            "rewrite_question",
            self._route_after_rewrite,
            {
                "classify": "classify_or_answer",
                "answer": "generate_answer"
            }
        )
        
        # Add edges from tools and generate to END
        workflow.add_edge("call_tools", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow
    
    def _route_after_classification(
        self,
        state: GraphState
    ) -> Literal["retrieve", "tools", "answer"]:
        """
        Route after classification node.
        
        Decides whether to retrieve, use tools, or answer directly.
        """
        if state.get("use_tools", False):
            return "tools"
        elif state.get("needs_retrieval", True):
            return "retrieve"
        else:
            return "answer"
    
    def _route_after_grading(
        self,
        state: GraphState
    ) -> Literal["generate", "rewrite", "answer"]:
        """
        Route after document grading.
        
        Decides whether to generate answer or rewrite query.
        """
        # If documents are relevant (needs_retrieval = False), generate
        if not state.get("needs_retrieval", True):
            return "generate"
        
        # If we've tried too many times, generate anyway
        if state.get("retry_count", 0) >= MAX_RETRIES:
            print(f"[ROUTE] Max retries reached, generating with available context")
            return "answer"
        
        # Otherwise, rewrite the question
        return "rewrite"
    
    def _route_after_rewrite(
        self,
        state: GraphState
    ) -> Literal["classify", "answer"]:
        """
        Route after question rewriting.
        
        Decides whether to try classification again or give up.
        """
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return "answer"
        return "classify"
    
    def invoke(self, question: str) -> dict:
        """
        Execute the graph for a question.
        
        Args:
            question: User question
            
        Returns:
            dict: Result with answer and metadata
        """
        print("\n" + "="*60)
        print("AGENTIC RAG EXECUTION")
        print("="*60)
        print(f"Question: {question}")
        print("="*60)
        
        # Create initial state
        initial_state = create_initial_state(question)
        
        # Run the graph
        final_state = self.app.invoke(initial_state)
        
        print("\n" + "="*60)
        print("EXECUTION COMPLETE")
        print("="*60)
        
        # Format response
        result = {
            "answer": final_state.get("answer", "No answer generated"),
            "question": final_state.get("question"),  # May be rewritten
            "documents": final_state.get("documents", []),
            "retry_count": final_state.get("retry_count", 0),
            "retrieval_attempted": final_state.get("retrieval_attempted", False),
            "error": final_state.get("error"),
        }
        
        return result
    
    def stream(self, question: str):
        """
        Stream graph execution.
        
        Args:
            question: User question
            
        Yields:
            Graph state updates
        """
        initial_state = create_initial_state(question)
        
        for output in self.app.stream(initial_state):
            yield output


def create_agentic_rag() -> AgenticRAGGraph:
    """Factory function to create agentic RAG graph."""
    return AgenticRAGGraph()
