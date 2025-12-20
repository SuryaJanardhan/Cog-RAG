"""
Test script for LlamaIndex integration (Phase 3).

Demonstrates advanced query engines:
- Compact response synthesis
- Refine mode for detailed answers
- Tree summarize for overviews
- Router engine for multi-source selection
- Sub-question decomposition
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llama_index.core import Document as LlamaDocument

from src.llamaindex import LlamaIndexManager
from src.llamaindex.query_engines import (
    CompactEngine,
    RouterEngine,
    SubQuestionEngine,
    HybridEngine,
)
from src.llamaindex.tools import LlamaIndexTools
from src.config.settings import get_settings


def test_basic_indexing():
    """Test basic document indexing and retrieval."""
    print("\n" + "="*80)
    print("TEST 1: Basic Indexing and Compact Query")
    print("="*80)
    
    # Create sample documents
    documents = [
        LlamaDocument(
            text="LlamaIndex is a framework for building LLM applications with data. It provides tools for data ingestion, indexing, and querying.",
            metadata={"source": "doc1", "category": "technical"}
        ),
        LlamaDocument(
            text="Vector databases store embeddings for similarity search. Popular options include Qdrant, Chroma, and Pinecone.",
            metadata={"source": "doc2", "category": "technical"}
        ),
        LlamaDocument(
            text="The quick brown fox jumps over the lazy dog. This is a sample sentence for testing.",
            metadata={"source": "doc3", "category": "general"}
        ),
    ]
    
    # Initialize manager and create index
    manager = LlamaIndexManager()
    manager.create_index(documents, show_progress=True)
    
    # Test compact query engine
    compact_engine = CompactEngine(manager.index, response_mode="compact")
    
    query = "What is LlamaIndex?"
    print(f"\nQuery: {query}")
    
    response = compact_engine.query(query)
    print(f"\nResponse:\n{response}")
    print(f"\nSource nodes: {len(response.source_nodes)}")
    
    return manager


def test_response_modes(manager: LlamaIndexManager):
    """Test different response synthesis modes."""
    print("\n" + "="*80)
    print("TEST 2: Response Synthesis Modes")
    print("="*80)
    
    query = "Explain vector databases and their use cases"
    
    # Test compact mode
    print(f"\n--- Compact Mode ---")
    compact_engine = CompactEngine(manager.index, response_mode="compact")
    response = compact_engine.query(query)
    print(f"Response: {response}")
    
    # Test refine mode
    print(f"\n--- Refine Mode ---")
    refine_engine = CompactEngine(manager.index, response_mode="refine")
    response = refine_engine.query(query)
    print(f"Response: {response}")
    
    # Test tree summarize mode
    print(f"\n--- Tree Summarize Mode ---")
    tree_engine = CompactEngine(manager.index, response_mode="tree_summarize")
    response = tree_engine.query(query)
    print(f"Response: {response}")


def test_router_engine():
    """Test router query engine with multiple indices."""
    print("\n" + "="*80)
    print("TEST 3: Router Query Engine")
    print("="*80)
    
    # Create separate indices for different topics
    technical_docs = [
        LlamaDocument(
            text="LangChain is a framework for developing applications powered by language models. It provides modules for prompts, chains, and agents.",
            metadata={"source": "tech1", "category": "technical"}
        ),
        LlamaDocument(
            text="RAG (Retrieval-Augmented Generation) combines retrieval with generation. It fetches relevant context before generating responses.",
            metadata={"source": "tech2", "category": "technical"}
        ),
    ]
    
    general_docs = [
        LlamaDocument(
            text="Python is a high-level programming language. It's known for simplicity and readability.",
            metadata={"source": "gen1", "category": "general"}
        ),
        LlamaDocument(
            text="Machine learning is a subset of AI that enables systems to learn from data.",
            metadata={"source": "gen2", "category": "general"}
        ),
    ]
    
    # Create indices
    tech_manager = LlamaIndexManager()
    tech_index = tech_manager.create_index(technical_docs)
    
    gen_manager = LlamaIndexManager()
    gen_index = gen_manager.create_index(general_docs)
    
    # Create router
    router = RouterEngine({
        "technical_docs": tech_index,
        "general_docs": gen_index,
    })
    
    # Test routing
    queries = [
        "What is RAG?",
        "Tell me about Python",
        "Explain LangChain",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = router.query(query)
        print(f"Response: {response}")


def test_subquestion_engine():
    """Test sub-question query engine."""
    print("\n" + "="*80)
    print("TEST 4: Sub-Question Query Engine")
    print("="*80)
    
    # Create indices with different domains
    domain1_docs = [
        LlamaDocument(
            text="Vector embeddings represent text as numerical vectors. They capture semantic meaning in high-dimensional space.",
            metadata={"domain": "embeddings"}
        ),
    ]
    
    domain2_docs = [
        LlamaDocument(
            text="Similarity search finds the most similar items to a query. Common metrics include cosine similarity and dot product.",
            metadata={"domain": "search"}
        ),
    ]
    
    # Create indices
    embed_manager = LlamaIndexManager()
    embed_index = embed_manager.create_index(domain1_docs)
    
    search_manager = LlamaIndexManager()
    search_index = search_manager.create_index(domain2_docs)
    
    # Create sub-question engine
    subq_engine = SubQuestionEngine({
        "embeddings": embed_index,
        "search": search_index,
    })
    
    # Test complex query
    query = "How do vector embeddings enable similarity search?"
    print(f"\nComplex Query: {query}")
    print("\nDecomposing into sub-questions...")
    
    response = subq_engine.query(query)
    print(f"\nFinal Response:\n{response}")


def test_hybrid_engine():
    """Test hybrid query engine with automatic mode selection."""
    print("\n" + "="*80)
    print("TEST 5: Hybrid Query Engine")
    print("="*80)
    
    # Create primary index
    primary_docs = [
        LlamaDocument(text="LlamaIndex provides data connectors for loading documents from various sources."),
        LlamaDocument(text="Query engines in LlamaIndex handle retrieval and response synthesis."),
    ]
    
    primary_manager = LlamaIndexManager()
    primary_index = primary_manager.create_index(primary_docs)
    
    # Create hybrid engine
    hybrid = HybridEngine(primary_index)
    
    # Test different query types
    queries = [
        ("What is LlamaIndex?", "auto"),
        ("Explain query engines", "compact"),
    ]
    
    for query, mode in queries:
        print(f"\nQuery: {query}")
        print(f"Mode: {mode}")
        response = hybrid.query(query, mode=mode)
        print(f"Response: {response}")


def test_langchain_tools():
    """Test LangChain tool wrappers."""
    print("\n" + "="*80)
    print("TEST 6: LangChain Tool Wrappers")
    print("="*80)
    
    # Create index
    docs = [
        LlamaDocument(text="LlamaIndex integrates with LangChain through tool wrappers."),
        LlamaDocument(text="Tools enable LlamaIndex query engines to be used in LangGraph workflows."),
    ]
    
    manager = LlamaIndexManager()
    manager.create_index(docs)
    
    # Create tools
    tools = LlamaIndexTools(manager)
    tool_list = tools.get_tools()
    
    print(f"\nAvailable tools: {len(tool_list)}")
    for tool in tool_list:
        print(f"- {tool.name}: {tool.description}")
    
    # Test a tool
    query = "How does LlamaIndex integrate with LangChain?"
    print(f"\nTesting tool with query: {query}")
    
    compact_tool = tool_list[0]  # First tool is compact query
    result = compact_tool.func(query)
    print(f"\nResult: {result}")


def interactive_mode():
    """Interactive query mode."""
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("\nInitializing LlamaIndex...")
    
    # Create sample index
    docs = [
        LlamaDocument(text="LlamaIndex is a data framework for LLM applications."),
        LlamaDocument(text="It supports various query engines including compact, refine, and tree summarize."),
        LlamaDocument(text="Router engines enable multi-source selection based on query content."),
    ]
    
    manager = LlamaIndexManager()
    manager.create_index(docs)
    
    # Create hybrid engine
    hybrid = HybridEngine(manager.index)
    
    print("\nReady! Type your questions (or 'quit' to exit):")
    
    while True:
        query = input("\nQuery: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        try:
            response = hybrid.query(query, mode="auto")
            print(f"\nAnswer: {response}")
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Run all tests."""
    settings = get_settings()
    
    if not settings.gemini_api_key:
        print("ERROR: GEMINI_API_KEY not set in environment")
        return
    
    print("\n" + "="*80)
    print("LLAMAINDEX INTEGRATION TESTS (Phase 3)")
    print("="*80)
    
    try:
        # Run tests
        manager = test_basic_indexing()
        test_response_modes(manager)
        test_router_engine()
        test_subquestion_engine()
        test_hybrid_engine()
        test_langchain_tools()
        
        # Interactive mode
        print("\n\nAll tests completed!")
        choice = input("\nEnter interactive mode? (y/n): ").strip().lower()
        if choice == 'y':
            interactive_mode()
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
