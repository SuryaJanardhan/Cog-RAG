"""
LangGraph nodes for agentic RAG workflow.
Each node is a function that takes GraphState and returns updated state.
"""
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import sqlite3
import re
import hashlib

from ..graph.state import GraphState
from ..llm import get_gemini_client
from ..retrieval import create_retriever
from ..config.settings import get_settings
from ..tools.registry import execute_tool, get_tool_registry

# Maximum retry attempts for query rewriting
MAX_RETRIES = 2

# Prompt templates
CLASSIFICATION_PROMPT = """You are an AI assistant that decides if a question needs document retrieval.

Question: {question}

Analyze if this question:
1. Can be answered with general knowledge (no retrieval needed)
2. Requires specific information from documents (retrieval needed)
3. Requires external tools like web search, database query, or calculator

Respond with a JSON object:
{{
    "needs_retrieval": true/false,
    "use_tools": true/false,
    "is_complex": true/false,
    "reasoning": "brief explanation"
}}

Response:"""

PLANNER_PROMPT = """You are an AI planner. Break down the user question into a sequential execution plan of search queries, tools, or database lookups.
Question: {question}

Available Tools:
- web_search: Search the web for current information
- sql_db_query: Query corporate SQL database (SELECT queries only)
- sql_db_execute: Modify database entries (INSERT, UPDATE, DELETE)
- calculator: Solve math expressions

Generate a list of 1 to 3 steps. Respond with a JSON object:
{{
    "plan": [
        "step description 1",
        "step description 2"
    ]
}}

JSON Response:"""

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
    
    def __init__(self, use_llamaindex: bool = False):
        """Initialize nodes with LLM and retriever."""
        self.settings = get_settings()
        self.llm_client = get_gemini_client()
        self.llm = self.llm_client.chat_model
        self.use_llamaindex = use_llamaindex or self.settings.llamaindex_enable_hybrid
        
        # Initialize retrievers
        self.retriever = create_retriever()
        self.llamaindex_retriever = None
        
        if self.use_llamaindex:
            try:
                from ..llamaindex import LlamaIndexManager
                self.llamaindex_manager = LlamaIndexManager()
                self.llamaindex_manager.create_index()
                self.llamaindex_retriever = self.llamaindex_manager.get_retriever()
                print("[INIT] LlamaIndex retriever initialized")
            except Exception as e:
                print(f"[INIT] LlamaIndex initialization failed: {e}. Falling back to LangChain.")
                self.use_llamaindex = False
                
    def check_semantic_cache(self, state: GraphState) -> GraphState:
        """Check the SQLite-based semantic cache first to save costs and latency."""
        print(f"\n[CACHE] Checking semantic cache for: {state['question'][:100]}...")
        try:
            conn = sqlite3.connect("./data/response_cache.db")
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    doc_ids TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            
            # Look for exact query match first
            cursor.execute("SELECT response FROM response_cache WHERE query = ?", (state["question"],))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                print(f"[CACHE] Hit! Found cached response for query.")
                state["answer"] = row[0]
                state["cache_hit"] = True
            else:
                print("[CACHE] Miss. Proceeding to execution pipeline.")
                state["cache_hit"] = False
        except Exception as e:
            print(f"[CACHE] Error checking cache: {e}")
            state["cache_hit"] = False
            
        return state
    
    def classify_or_answer(self, state: GraphState) -> GraphState:
        """Decide if retrieval, tools, or planning is needed."""
        # If cache hit occurred, skip classification
        if state.get("cache_hit"):
            return state
            
        print(f"\n[CLASSIFY] Analyzing question: {state['question'][:100]}...")
        
        prompt = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({"question": state["question"]})
            
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                decision = json.loads(json_str)
            else:
                decision = {"needs_retrieval": True, "use_tools": False, "is_complex": False, "reasoning": "Default"}
            
            state["needs_retrieval"] = decision.get("needs_retrieval", True)
            state["use_tools"] = decision.get("use_tools", False)
            
            print(f"[CLASSIFY] Decision: retrieval={state['needs_retrieval']}, tools={state['use_tools']}")
            print(f"[CLASSIFY] Reasoning: {decision.get('reasoning', 'N/A')}")
            
            # Formulate multi-step plan if the query is complex
            if decision.get("is_complex", False) or state["use_tools"]:
                print("[PLANNER] Question is complex. Formulating execution plan...")
                plan_prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
                plan_chain = plan_prompt | self.llm | StrOutputParser()
                plan_resp = plan_chain.invoke({"question": state["question"]})
                
                if "{" in plan_resp and "}" in plan_resp:
                    j_start = plan_resp.find("{")
                    j_end = plan_resp.rfind("}") + 1
                    plan_data = json.loads(plan_resp[j_start:j_end])
                    state["plan"] = plan_data.get("plan", [state["question"]])
                else:
                    state["plan"] = [state["question"]]
                    
                state["current_step_idx"] = 0
                print(f"[PLANNER] Plan formulated: {state['plan']}")
            
        except Exception as e:
            print(f"[CLASSIFY] Error: {e}. Defaulting to retrieval.")
            state["needs_retrieval"] = True
            state["use_tools"] = False
        
        return state
    
    def retrieve(self, state: GraphState) -> GraphState:
        """Retrieve relevant documents."""
        # If cache hit, skip retrieval
        if state.get("cache_hit"):
            return state
            
        print(f"\n[RETRIEVE] Fetching documents for: {state['question'][:100]}...")
        
        try:
            if self.use_llamaindex and self.llamaindex_retriever:
                print("[RETRIEVE] Using LlamaIndex retriever")
                nodes = self.llamaindex_retriever.retrieve(state["question"])
                documents = [Document(page_content=node.get_content(), metadata=node.metadata) for node in nodes]
            else:
                print("[RETRIEVE] Using LangChain retriever")
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
        """Grade document relevance."""
        if state.get("cache_hit"):
            return state
            
        print(f"\n[GRADE] Evaluating {len(state['documents'])} documents...")
        
        if not state["documents"]:
            print("[GRADE] No documents to grade")
            return state
        
        # Format top 3 documents for grading
        doc_text = "\n---\n".join([
            f"Doc {i+1}: {doc.page_content[:200]}..."
            for i, doc in enumerate(state["documents"][:3])
        ])
        
        prompt = ChatPromptTemplate.from_template(GRADING_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "question": state["question"],
                "documents": doc_text
            })
            
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                grade = json.loads(json_str)
            else:
                grade = {"relevant": True, "reasoning": "Default"}
            
            is_relevant = grade.get("relevant", True)
            print(f"[GRADE] Relevant: {is_relevant}")
            print(f"[GRADE] Reasoning: {grade.get('reasoning', 'N/A')}")
            
            state["needs_retrieval"] = not is_relevant
            
        except Exception as e:
            print(f"[GRADE] Error: {e}. Assuming documents are relevant.")
            state["needs_retrieval"] = False
        
        return state
    
    def rewrite_question(self, state: GraphState) -> GraphState:
        """Rewrite question for better retrieval."""
        if state.get("cache_hit"):
            return state
            
        print(f"\n[REWRITE] Attempt {state['retry_count'] + 1}/{MAX_RETRIES}")
        
        prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            rewritten = chain.invoke({"question": state["question"]})
            
            print(f"[REWRITE] Original: {state['question']}")
            print(f"[REWRITE] Rewritten: {rewritten}")
            
            state["question"] = rewritten.strip()
            state["retry_count"] += 1
            state["retrieval_attempted"] = False
            
        except Exception as e:
            print(f"[REWRITE] Error: {e}")
            state["retry_count"] += 1
        
        return state
    
    def generate_answer(self, state: GraphState) -> GraphState:
        """Generate final answer and save to cache."""
        if state.get("cache_hit"):
            return state
            
        print(f"\n[GENERATE] Creating answer...")
        
        # Format context
        if state["documents"]:
            context = "\n---\n".join([
                f"[Doc {i+1}] {doc.page_content}"
                for i, doc in enumerate(state["documents"])
            ])
        else:
            context = "No specific documents available."
        
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
            
            # Save to cache
            self._save_to_cache(state["question"], answer)
            
        except Exception as e:
            print(f"[GENERATE] Error: {e}")
            state["error"] = f"Generation error: {str(e)}"
            state["answer"] = f"Error generating answer: {str(e)}"
        
        return state
    
    def _save_to_cache(self, query: str, response: str) -> None:
        """Save query and answer into the SQLite response cache."""
        try:
            conn = sqlite3.connect("./data/response_cache.db")
            cursor = conn.cursor()
            cache_key = hashlib.sha256(f"{query}:general_cache".encode()).hexdigest()
            cursor.execute(
                "INSERT OR REPLACE INTO response_cache (cache_key, query, doc_ids, response) VALUES (?, ?, ?, ?)",
                (cache_key, query, json.dumps(["general_cache"]), response)
            )
            conn.commit()
            conn.close()
            print("[CACHE] Answer successfully cached in SQLite.")
        except Exception as e:
            print(f"[CACHE] Error writing to cache: {e}")
            
    def call_tools(self, state: GraphState) -> GraphState:
        """Execute tools. Supports planning sub-agents and checking HITL requirements."""
        if state.get("cache_hit"):
            return state
            
        # Get active step from plan, or fallback to the question
        step_query = state["question"]
        if state.get("plan") and state["current_step_idx"] < len(state["plan"]):
            step_query = state["plan"][state["current_step_idx"]]
            print(f"\n[TOOLS] Executing Plan Step {state['current_step_idx'] + 1}/{len(state['plan'])}: {step_query}")
        else:
            print(f"\n[TOOLS] Executing tools for query: {step_query[:100]}...")
            
        # Ask LLM which tool to use for the query
        registry = get_tool_registry()
        descriptions = registry.get_tool_descriptions()
        
        tool_select_prompt = f"""Given the query: '{step_query}'
And the available tools:
{descriptions}

Determine the tool name and input parameter to execute.
Respond ONLY with a JSON object:
{{
    "tool_name": "web_search" or "sql_db_query" or "sql_db_execute" or "calculator" or "none",
    "tool_input": "search query or query parameters"
}}
Response:"""
        
        try:
            resp = self.llm.invoke(tool_select_prompt).content
            if "{" in resp and "}" in resp:
                j_start = resp.find("{")
                j_end = resp.rfind("}") + 1
                decision = json.loads(resp[j_start:j_end])
                tool_name = decision.get("tool_name", "none")
                tool_input = decision.get("tool_input", "")
            else:
                tool_name = "none"
                tool_input = ""
                
            if tool_name != "none":
                # Check for Human-in-the-Loop approval gate (e.g. sql write operations)
                if tool_name == "sql_db_execute" and not state.get("human_approved"):
                    print(f"[HITL] Write operation detected: tool='{tool_name}', input='{tool_input}'")
                    print("[HITL] Flagging human approval required.")
                    state["human_approval_required"] = True
                    # Pause execution by returning
                    return state
                    
                print(f"[TOOLS] Executing tool '{tool_name}' with input '{tool_input}'")
                result = execute_tool(tool_name, tool_input)
                
                # Append result to tool_results
                old_results = state.get("tool_results") or ""
                state["tool_results"] = f"{old_results}\nStep query: {step_query}\nTool: {tool_name}\nResult: {result}\n"
            else:
                print(f"[TOOLS] No specific tool selected for: {step_query[:50]}")
                
        except Exception as e:
            print(f"[TOOLS] Error running tool agent: {e}")
            
        # Move to next plan step if using planner
        if state.get("plan"):
            state["current_step_idx"] += 1
            if state["current_step_idx"] >= len(state["plan"]):
                state["use_tools"] = False  # Done with all plan steps
        else:
            state["use_tools"] = False  # Single tool invocation done
            
        return state


def create_nodes(use_llamaindex: bool = False) -> RAGNodes:
    """Factory function to create node instances."""
    return RAGNodes(use_llamaindex=use_llamaindex)
