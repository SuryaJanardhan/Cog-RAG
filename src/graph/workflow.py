"""
LangGraph workflow builder for agentic RAG.
Defines the graph structure with conditional edges and routing.
"""
from typing import Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import GraphState, create_initial_state
from .nodes import create_nodes, MAX_RETRIES


class AgenticRAGGraph:
    """Agentic RAG workflow using LangGraph."""
    
    def __init__(self, use_llamaindex: bool = False):
        """Initialize the graph with nodes."""
        self.nodes = create_nodes(use_llamaindex=use_llamaindex)
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=MemorySaver())
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Graph structure:
        START -> check_semantic_cache
                  ├─> (cache_hit) -> END
                  └─> (cache_miss) -> classify_or_answer
                                         ├─> (needs retrieval) -> retrieve -> grade_documents
                                         │                         ├─> (relevant) -> generate_answer -> END
                                         │                         └─> (not relevant) -> rewrite_question -> classify_or_answer
                                         ├─> (needs tools) -> call_tools (loop for plan)
                                         │                       ├─> (HITL pause) -> END
                                         │                       └─> (completed) -> generate_answer -> END
                                         └─> (direct answer) -> generate_answer -> END
        """
        # Create graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("check_semantic_cache", self.nodes.check_semantic_cache)
        workflow.add_node("classify_or_answer", self.nodes.classify_or_answer)
        workflow.add_node("retrieve", self.nodes.retrieve)
        workflow.add_node("grade_documents", self.nodes.grade_documents)
        workflow.add_node("rewrite_question", self.nodes.rewrite_question)
        workflow.add_node("generate_answer", self.nodes.generate_answer)
        workflow.add_node("call_tools", self.nodes.call_tools)
        
        # Set entry point
        workflow.set_entry_point("check_semantic_cache")
        
        # Add conditional edges from check_semantic_cache
        workflow.add_conditional_edges(
            "check_semantic_cache",
            self._route_after_cache,
            {
                "hit": END,
                "miss": "classify_or_answer"
            }
        )
        
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
        
        # Add conditional edges from call_tools
        workflow.add_conditional_edges(
            "call_tools",
            self._route_after_tools,
            {
                "generate": "generate_answer",
                "tools": "call_tools",
                "pause": END
            }
        )
        
        # Add edge from generate_answer to END
        workflow.add_edge("generate_answer", END)
        
        return workflow
        
    def _route_after_cache(self, state: GraphState) -> Literal["hit", "miss"]:
        """Route based on semantic cache check."""
        if state.get("cache_hit", False):
            return "hit"
        return "miss"
    
    def _route_after_classification(
        self,
        state: GraphState
    ) -> Literal["retrieve", "tools", "answer"]:
        """Route after classification node."""
        if state.get("cache_hit", False):
            return "answer"
        if state.get("use_tools", False) or state.get("plan"):
            return "tools"
        elif state.get("needs_retrieval", True):
            return "retrieve"
        else:
            return "answer"
    
    def _route_after_grading(
        self,
        state: GraphState
    ) -> Literal["generate", "rewrite", "answer"]:
        """Route after document grading."""
        if state.get("cache_hit", False):
            return "answer"
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
        """Route after question rewriting."""
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return "answer"
        return "classify"
        
    def _route_after_tools(self, state: GraphState) -> Literal["generate", "tools", "pause"]:
        """Route after tool execution step."""
        if state.get("human_approval_required", False):
            return "pause"
        # If we have plan steps remaining
        if state.get("plan") and state.get("current_step_idx", 0) < len(state["plan"]):
            return "tools"
        return "generate"
    
    def invoke(self, question: str, thread_id: str = "default_thread", human_approved: bool = False) -> dict:
        """Execute the graph for a question with checkpointer context."""
        print("\n" + "="*60)
        print("AGENTIC RAG EXECUTION")
        print("="*60)
        print(f"Question: {question}")
        print("="*60)
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # If human_approved is passed, update state before resuming
        if human_approved:
            print("[HITL] Human approved the pending action. Resuming graph...")
            current_state = self.app.get_state(config)
            if current_state and current_state.values:
                updated_values = dict(current_state.values)
                updated_values["human_approved"] = True
                updated_values["human_approval_required"] = False
                self.app.update_state(config, updated_values)
                final_state = self.app.invoke(None, config)
            else:
                initial_state = create_initial_state(question)
                initial_state["human_approved"] = True
                final_state = self.app.invoke(initial_state, config)
        else:
            # Create initial state and run
            initial_state = create_initial_state(question)
            final_state = self.app.invoke(initial_state, config)
        
        print("\n" + "="*60)
        print("EXECUTION COMPLETE")
        print("="*60)
        
        # Format response
        result = {
            "answer": final_state.get("answer", "No answer generated"),
            "question": final_state.get("question"),
            "documents": final_state.get("documents", []),
            "retry_count": final_state.get("retry_count", 0),
            "retrieval_attempted": final_state.get("retrieval_attempted", False),
            "error": final_state.get("error"),
            "human_approval_required": final_state.get("human_approval_required", False),
            "cache_hit": final_state.get("cache_hit", False),
        }
        
        return result
    
    def stream(self, question: str, thread_id: str = "default_thread"):
        """Stream graph execution."""
        initial_state = create_initial_state(question)
        config = {"configurable": {"thread_id": thread_id}}
        
        for output in self.app.stream(initial_state, config):
            yield output


def create_agentic_rag() -> AgenticRAGGraph:
    """Factory function to create agentic RAG graph."""
    return AgenticRAGGraph()
