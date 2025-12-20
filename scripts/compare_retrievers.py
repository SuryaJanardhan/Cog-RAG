"""
Compare LangChain vs LlamaIndex retrievers (Phase 3).

Benchmarks retrieval quality and performance between:
- LangChain retriever (Phase 1)
- LlamaIndex compact engine (Phase 3)
- LlamaIndex refine engine (Phase 3)
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llama_index.core import Document as LlamaDocument

from src.retrieval import create_retriever
from src.llamaindex import LlamaIndexManager
from src.llamaindex.query_engines import CompactEngine
from src.config.settings import get_settings


# Test queries
TEST_QUERIES = [
    "What is RAG?",
    "Explain vector embeddings",
    "How does similarity search work?",
    "Compare different retrieval methods",
]


def setup_sample_data():
    """Create sample documents for testing."""
    sample_docs = [
        """RAG (Retrieval-Augmented Generation) is a technique that combines retrieval with generation.
        It works by first retrieving relevant documents from a knowledge base, then using those documents
        as context for generating responses with a language model.""",
        
        """Vector embeddings are numerical representations of text that capture semantic meaning.
        They map words, sentences, or documents to points in a high-dimensional space where similar
        items are closer together. Common models include Word2Vec, GloVe, and transformer-based embeddings.""",
        
        """Similarity search finds the most similar items to a query in a vector database.
        Common metrics include cosine similarity, Euclidean distance, and dot product.
        These metrics measure how close two vectors are in the embedding space.""",
        
        """There are multiple retrieval methods: dense retrieval uses vector embeddings and similarity search,
        sparse retrieval uses traditional keyword matching like BM25, and hybrid retrieval combines both approaches.
        Each has trade-offs in terms of accuracy, speed, and resource requirements.""",
        
        """LangChain provides a standardized interface for retrieval through its retriever abstraction.
        It supports various backends including vector stores, keyword search, and hybrid approaches.
        Retrievers can be easily swapped and combined in LangChain applications.""",
        
        """LlamaIndex offers advanced query engines beyond basic retrieval. The compact engine concatenates
        chunks for synthesis, refine engine iteratively improves answers, and tree summarize builds hierarchical
        summaries. These engines provide more sophisticated response generation than simple retrieval.""",
    ]
    
    return sample_docs


def test_langchain_retriever(query: str, sample_docs: list):
    """Test LangChain retriever."""
    print(f"\n--- LangChain Retriever ---")
    
    # Note: In real scenario, documents would already be indexed
    # Here we're just testing the retrieval interface
    
    retriever = create_retriever()
    
    start_time = time.time()
    try:
        documents = retriever.retrieve(query)
        elapsed = time.time() - start_time
        
        print(f"Retrieved: {len(documents)} documents")
        print(f"Time: {elapsed:.3f}s")
        
        if documents:
            print(f"\nTop result (score: {documents[0].get('score', 'N/A')}):")
            content = documents[0].get('content', '')
            print(f"{content[:200]}...")
        
        return {
            "method": "LangChain",
            "num_docs": len(documents),
            "time": elapsed,
            "top_score": documents[0].get('score', 0) if documents else 0,
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"method": "LangChain", "error": str(e)}


def test_llamaindex_compact(query: str, manager: LlamaIndexManager):
    """Test LlamaIndex compact engine."""
    print(f"\n--- LlamaIndex Compact ---")
    
    compact_engine = CompactEngine(manager.index, response_mode="compact")
    
    start_time = time.time()
    try:
        response = compact_engine.query(query)
        elapsed = time.time() - start_time
        
        print(f"Response generated in {elapsed:.3f}s")
        print(f"Source nodes: {len(response.source_nodes)}")
        
        print(f"\nResponse:")
        print(f"{str(response)[:300]}...")
        
        return {
            "method": "LlamaIndex Compact",
            "num_sources": len(response.source_nodes),
            "time": elapsed,
            "response_length": len(str(response)),
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"method": "LlamaIndex Compact", "error": str(e)}


def test_llamaindex_refine(query: str, manager: LlamaIndexManager):
    """Test LlamaIndex refine engine."""
    print(f"\n--- LlamaIndex Refine ---")
    
    refine_engine = CompactEngine(manager.index, response_mode="refine")
    
    start_time = time.time()
    try:
        response = refine_engine.query(query)
        elapsed = time.time() - start_time
        
        print(f"Response generated in {elapsed:.3f}s")
        print(f"Source nodes: {len(response.source_nodes)}")
        
        print(f"\nResponse:")
        print(f"{str(response)[:300]}...")
        
        return {
            "method": "LlamaIndex Refine",
            "num_sources": len(response.source_nodes),
            "time": elapsed,
            "response_length": len(str(response)),
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"method": "LlamaIndex Refine", "error": str(e)}


def compare_retrievers():
    """Compare all retrieval methods."""
    print("\n" + "="*80)
    print("RETRIEVER COMPARISON (Phase 3)")
    print("="*80)
    
    settings = get_settings()
    
    if not settings.gemini_api_key:
        print("ERROR: GEMINI_API_KEY not set")
        return
    
    # Setup data
    print("\nSetting up sample data...")
    sample_docs = setup_sample_data()
    
    # Create LlamaIndex manager
    print("Initializing LlamaIndex...")
    llama_docs = [LlamaDocument(text=doc) for doc in sample_docs]
    manager = LlamaIndexManager()
    manager.create_index(llama_docs, show_progress=False)
    
    # Run comparisons
    all_results = []
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print("\n" + "="*80)
        print(f"QUERY {i}: {query}")
        print("="*80)
        
        results = []
        
        # Test LangChain
        try:
            result = test_langchain_retriever(query, sample_docs)
            results.append(result)
        except Exception as e:
            print(f"LangChain test failed: {e}")
        
        # Test LlamaIndex Compact
        try:
            result = test_llamaindex_compact(query, manager)
            results.append(result)
        except Exception as e:
            print(f"LlamaIndex Compact test failed: {e}")
        
        # Test LlamaIndex Refine
        try:
            result = test_llamaindex_refine(query, manager)
            results.append(result)
        except Exception as e:
            print(f"LlamaIndex Refine test failed: {e}")
        
        all_results.append({
            "query": query,
            "results": results,
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for query_result in all_results:
        print(f"\nQuery: {query_result['query']}")
        print("-" * 40)
        
        for result in query_result['results']:
            if 'error' in result:
                print(f"{result['method']}: ERROR - {result['error']}")
            else:
                method = result['method']
                time_taken = result.get('time', 'N/A')
                
                if method == "LangChain":
                    print(f"{method}: {result.get('num_docs', 0)} docs in {time_taken:.3f}s")
                else:
                    print(f"{method}: {result.get('num_sources', 0)} sources, "
                          f"{result.get('response_length', 0)} chars in {time_taken:.3f}s")
    
    # Average times
    print("\n" + "="*80)
    print("AVERAGE PERFORMANCE")
    print("="*80)
    
    method_times = {}
    for query_result in all_results:
        for result in query_result['results']:
            if 'error' not in result and 'time' in result:
                method = result['method']
                if method not in method_times:
                    method_times[method] = []
                method_times[method].append(result['time'])
    
    for method, times in method_times.items():
        avg_time = sum(times) / len(times)
        print(f"{method}: {avg_time:.3f}s average")
    
    print("\n" + "="*80)
    print("KEY DIFFERENCES")
    print("="*80)
    print("""
LangChain Retriever (Phase 1):
  ✓ Fast and simple
  ✓ Direct document retrieval
  ✓ Returns raw documents with scores
  ✗ No response synthesis
  ✗ Limited context handling
  
LlamaIndex Compact (Phase 3):
  ✓ Combines retrieval + synthesis
  ✓ Concatenates chunks efficiently
  ✓ Natural language responses
  ✗ Slower than pure retrieval
  
LlamaIndex Refine (Phase 3):
  ✓ Iterative refinement
  ✓ More detailed answers
  ✓ Better context integration
  ✗ Slowest (multiple LLM calls)
  ✗ Higher token usage
    """)


def main():
    """Run comparison."""
    compare_retrievers()
    
    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80)


if __name__ == "__main__":
    main()
