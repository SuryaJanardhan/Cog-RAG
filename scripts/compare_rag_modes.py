"""
Script to compare basic RAG vs agentic RAG behavior.
Shows the differences between Phase 1 and Phase 2 approaches.
"""
import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import create_rag_pipeline
from src.graph import create_agentic_rag


def compare_on_question(question: str):
    """Compare basic vs agentic RAG on a single question."""
    print("\n" + "="*60)
    print(f"QUESTION: {question}")
    print("="*60)
    
    # Test with basic RAG (Phase 1)
    print("\n--- Phase 1: Basic RAG ---")
    start = time.time()
    
    pipeline = create_rag_pipeline()
    basic_result = pipeline.query(question, return_sources=False)
    
    basic_time = time.time() - start
    
    print(f"Answer: {basic_result['answer'][:200]}...")
    print(f"Sources: {basic_result.get('num_sources', 0)}")
    print(f"Time: {basic_time:.2f}s")
    
    # Test with agentic RAG (Phase 2)
    print("\n--- Phase 2: Agentic RAG ---")
    start = time.time()
    
    graph = create_agentic_rag()
    agentic_result = graph.invoke(question)
    
    agentic_time = time.time() - start
    
    print(f"Answer: {agentic_result['answer'][:200]}...")
    print(f"Documents: {len(agentic_result.get('documents', []))}")
    print(f"Retrieval: {agentic_result['retrieval_attempted']}")
    print(f"Rewrites: {agentic_result['retry_count']}")
    print(f"Time: {agentic_time:.2f}s")
    
    # Comparison
    print("\n--- Comparison ---")
    print(f"Time difference: {abs(basic_time - agentic_time):.2f}s")
    print(f"Agentic benefits: Adaptive retrieval, query rewriting, relevance grading")


def run_comparison_suite():
    """Run comparison on multiple question types."""
    print("\n" + "="*60)
    print("BASIC RAG vs AGENTIC RAG COMPARISON")
    print("="*60)
    
    test_questions = [
        # Simple factual question
        "What is the capital of France?",
        
        # RAG-specific question
        "What are the key components of a RAG system?",
        
        # Question needing context
        "How do embeddings work in RAG?",
        
        # Ambiguous question
        "Explain the architecture",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_questions)}")
        compare_on_question(question)
        
        if i < len(test_questions):
            input("\nPress Enter to continue...")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print("\nKey differences:")
    print("• Phase 1 (Basic): Always retrieves, uses cache, straightforward")
    print("• Phase 2 (Agentic): Decides when to retrieve, rewrites queries, grades relevance")
    print("\nBoth approaches have their place:")
    print("• Basic RAG: Faster, predictable, good for known-good queries")
    print("• Agentic RAG: Smarter, handles unclear queries, more robust")


def main():
    """Main comparison script."""
    print("\n" + "="*60)
    print("RAG COMPARISON SCRIPT")
    print("="*60)
    print("\nCompare Phase 1 (Basic RAG) vs Phase 2 (Agentic RAG)")
    print("\nNote: Requires documents to be ingested first!")
    print("      Run: python scripts/ingest_documents.py")
    
    print("\nOptions:")
    print("  1. Run full comparison suite")
    print("  2. Single question comparison")
    
    choice = input("\nEnter choice (1-2) or press Enter for option 1: ").strip() or "1"
    
    if choice == "1":
        run_comparison_suite()
    elif choice == "2":
        question = input("\nEnter question: ").strip()
        if question:
            compare_on_question(question)
        else:
            print("No question provided")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
