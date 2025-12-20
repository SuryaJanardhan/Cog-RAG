"""
LangGraph nodes for agentic RAG workflow.
Each node is a function that takes GraphState and returns updated state.
"""
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

from ..graph.state import GraphState
from ..llm import get_gemini_client
from ..retrieval import create_retriever


# Maximum retry attempts for query rewriting
MAX_RETRIES = 2

# Prompt templates
CLASSIFICATION_PROMPT = """You are an AI assistant that decides if a question needs document retrieval.

Question: {question}

Analyze if this question:
1. Can be answered with general knowledge (no retrieval needed)
2. Requires specific information from documents (retrieval needed)
3. Requires external tools like web search, calculator, or APIs

Respond with a JSON object:
{{
    "needs_retrieval": true/false,
    "use_tools": true/false,
    "reasoning": "brief explanation"
}}

Response:"""

GRADING_PROMPT = """You are a document relevance grader. Determine if retrieved documents are relevant to the question.

Question: {question}

Documents:
{documents}

Are these documents relevant and sufficient to answer the question?

Respond with a JSON object:
{{
    "relevant": true/false,
    "reasoning": "brief explanation"
}}

Response:"""

REWRITE_PROMPT = """You are a question rewriter. The initial retrieval didn't find relevant documents.

Original Question: {question}

Rewrite this question to be more specific and likely to find relevant documents.
Use different keywords, rephrase, or break down complex questions.

Rewritten Question:"""

ANSWER_PROMPT = """You are a helpful AI assistant. Answer the question based on the provided context.

If the context is insufficient or empty, use your general knowledge but clearly state you're doing so.

Context:
{context}

Question: {question}

Answer:"""


class RAGNodes:
    """LangGraph node implementations for agentic RAG."""
    
    def __init__(self):
        """Initialize nodes with LLM and retriever."""
        self.llm_client = get_gemini_client()
        self.llm = self.llm_client.chat_model
        self.retriever = create_retriever()
    
    def classify_or_answer(self, state: GraphState) -> GraphState:
        """
        Decide if retrieval or tools are needed.
        
        Node that analyzes the question and determines the next step.
        """
        print(f"\n[CLASSIFY] Analyzing question: {state['question'][:100]}...")
        
        prompt = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({"question": state["question"]})
            
            # Parse JSON response
            # Try to extract JSON from response
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                decision = json.loads(json_str)
            else:
                # Fallback: assume retrieval needed
                decision = {"needs_retrieval": True, "use_tools": False, "reasoning": "Default"}
            
            state["needs_retrieval"] = decision.get("needs_retrieval", True)
            state["use_tools"] = decision.get("use_tools", False)
            
            print(f"[CLASSIFY] Decision: retrieval={state['needs_retrieval']}, tools={state['use_tools']}")
            print(f"[CLASSIFY] Reasoning: {decision.get('reasoning', 'N/A')}")
            
        except Exception as e:
            print(f"[CLASSIFY] Error: {e}. Defaulting to retrieval.")
            state["needs_retrieval"] = True
            state["use_tools"] = False
        
        return state
    
    def retrieve(self, state: GraphState) -> GraphState:
        """
        Retrieve relevant documents.
        
        Node that performs vector similarity search.
        """
        print(f"\n[RETRIEVE] Fetching documents for: {state['question'][:100]}...")
        
        try:
            documents = self.retriever.retrieve(state["question"])
            state["documents"] = documents
            state["retrieval_attempted"] = True
            
            print(f"[RETRIEVE] Found {len(documents)} documents")
            
        except Exception as e:
            print(f"[RETRIEVE] Error: {e}")
            state["error"] = f"Retrieval error: {str(e)}"
            state["documents"] = []
        
        return state
    
    def grade_documents(self, state: GraphState) -> GraphState:
        """
        Grade document relevance.
        
        Node that checks if retrieved documents are relevant.
        """
        print(f"\n[GRADE] Evaluating {len(state['documents'])} documents...")
        
        if not state["documents"]:
            print("[GRADE] No documents to grade")
            return state
        
        # Format documents for grading
        doc_text = "\n---\n".join([
            f"Doc {i+1}: {doc.page_content[:200]}..."
            for i, doc in enumerate(state["documents"][:3])  # Grade top 3
        ])
        
        prompt = ChatPromptTemplate.from_template(GRADING_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "question": state["question"],
                "documents": doc_text
            })
            
            # Parse JSON response
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                grade = json.loads(json_str)
            else:
                # Fallback: assume relevant if we have docs
                grade = {"relevant": True, "reasoning": "Default"}
            
            is_relevant = grade.get("relevant", True)
            print(f"[GRADE] Relevant: {is_relevant}")
            print(f"[GRADE] Reasoning: {grade.get('reasoning', 'N/A')}")
            
            # Store relevance in state (we'll use this in routing)
            state["needs_retrieval"] = not is_relevant
            
        except Exception as e:
            print(f"[GRADE] Error: {e}. Assuming documents are relevant.")
            state["needs_retrieval"] = False
        
        return state
    
    def rewrite_question(self, state: GraphState) -> GraphState:
        """
        Rewrite question for better retrieval.
        
        Node that improves the query when initial retrieval fails.
        """
        print(f"\n[REWRITE] Attempt {state['retry_count'] + 1}/{MAX_RETRIES}")
        
        prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            rewritten = chain.invoke({"question": state["question"]})
            
            print(f"[REWRITE] Original: {state['question']}")
            print(f"[REWRITE] Rewritten: {rewritten}")
            
            state["question"] = rewritten.strip()
            state["retry_count"] += 1
            state["retrieval_attempted"] = False  # Allow retry
            
        except Exception as e:
            print(f"[REWRITE] Error: {e}")
            state["retry_count"] += 1
        
        return state
    
    def generate_answer(self, state: GraphState) -> GraphState:
        """
        Generate final answer.
        
        Node that produces the answer using context or general knowledge.
        """
        print(f"\n[GENERATE] Creating answer...")
        
        # Format context from documents
        if state["documents"]:
            context = "\n---\n".join([
                f"[Doc {i+1}] {doc.page_content}"
                for i, doc in enumerate(state["documents"])
            ])
        else:
            context = "No specific documents available."
        
        # Add tool results if available
        if state.get("tool_results"):
            context += f"\n\nTool Results:\n{state['tool_results']}"
        
        prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            answer = chain.invoke({
                "context": context,
                "question": state["question"]
            })
            
            state["answer"] = answer
            print(f"[GENERATE] Answer generated ({len(answer)} chars)")
            
        except Exception as e:
            print(f"[GENERATE] Error: {e}")
            state["error"] = f"Generation error: {str(e)}"
            state["answer"] = f"Error generating answer: {str(e)}"
        
        return state
    
    def call_tools(self, state: GraphState) -> GraphState:
        """
        Execute external tools.
        
        Node that calls web search or other tools when needed.
        """
        print(f"\n[TOOLS] Executing tools for: {state['question'][:100]}...")
        
        # Tool execution will be implemented in tools module
        # For now, just mark as attempted
        state["use_tools"] = False  # Don't loop
        state["tool_results"] = "Tool integration pending - Phase 2 in progress"
        
        print("[TOOLS] Tool execution completed")
        
        return state


def create_nodes() -> RAGNodes:
    """Factory function to create node instances."""
    return RAGNodes()
