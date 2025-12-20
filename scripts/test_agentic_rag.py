"""
Script to test agentic RAG behavior with LangGraph.
Demonstrates adaptive retrieval and query rewriting.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph import create_agentic_rag


def test_simple_question():
    """Test question that doesn't need retrieval."""
    print("\n" + "="*60)
    print("TEST 1: Simple Question (No Retrieval Needed)")
    print("="*60)
    
    graph = create_agentic_rag()
    
    question = "What is 2 + 2?"
    result = graph.invoke(question)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Retrieval Attempted: {result['retrieval_attempted']}")
    print(f"Retry Count: {result['retry_count']}")


def test_rag_question():
    """Test question that needs document retrieval."""
    print("\n" + "="*60)
    print("TEST 2: RAG Question (Needs Retrieval)")
    print("="*60)
    
    graph = create_agentic_rag()
    
    question = "What are the key components of a RAG system?"
    result = graph.invoke(question)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Documents Retrieved: {len(result.get('documents', []))}")
    print(f"Retrieval Attempted: {result['retrieval_attempted']}")
    print(f"Retry Count: {result['retry_count']}")


def test_ambiguous_question():
    """Test question that might trigger query rewriting."""
    print("\n" + "="*60)
    print("TEST 3: Ambiguous Question (Might Trigger Rewrite)")
    print("="*60)
    
    graph = create_agentic_rag()
    
    question = "How does it work?"
    result = graph.invoke(question)
    
    print(f"\nOriginal Question: {question}")
    print(f"Final Question: {result['question']}")
    print(f"Question Rewritten: {question != result['question']}")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Retry Count: {result['retry_count']}")


def test_multiple_questions():
    """Test multiple questions in sequence."""
    print("\n" + "="*60)
    print("TEST 4: Multiple Questions")
    print("="*60)
    
    graph = create_agentic_rag()
    
    questions = [
        "What is RAG?",
        "Explain the benefits of RAG systems",
        "What are typical use cases?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i}/{len(questions)} ---")
        print(f"Q: {question}")
        
        result = graph.invoke(question)
        
        print(f"A: {result['answer'][:150]}...")
        print(f"Docs: {len(result.get('documents', []))}, Retries: {result['retry_count']}")


def interactive_mode():
    """Interactive agentic RAG mode."""
    print("\n" + "="*60)
    print("AGENTIC RAG - INTERACTIVE MODE")
    print("="*60)
    print("\nThis mode uses LangGraph for adaptive RAG behavior:")
    print("  • Decides if retrieval is needed")
    print("  • Rewrites unclear questions")
    print("  • Grades document relevance")
    print("\nType 'exit' to quit, 'test' for predefined tests")
    print("-" * 60)
    
    graph = create_agentic_rag()
    
    while True:
        question = input("\nQuestion: ").strip()
        
        if not question:
            continue
        
        if question.lower() == 'exit':
            print("\nGoodbye!")
            break
        
        if question.lower() == 'test':
            test_simple_question()
            test_rag_question()
            test_ambiguous_question()
            continue
        
        try:
            result = graph.invoke(question)
            
            print("\n" + "="*60)
            print("ANSWER")
            print("="*60)
            print(result['answer'])
            
            print("\n" + "="*60)
            print("EXECUTION METADATA")
            print("="*60)
            if result['question'] != question:
                print(f"✓ Question rewritten: {result['question']}")
            print(f"• Retrieval attempted: {result['retrieval_attempted']}")
            print(f"• Documents found: {len(result.get('documents', []))}")
            print(f"• Retry count: {result['retry_count']}")
            
            if result.get('error'):
                print(f"⚠ Error: {result['error']}")
            
        except Exception as e:
            print(f"\n✗ Error: {e}")


def main():
    """Main test script."""
    print("\n" + "="*60)
    print("AGENTIC RAG TEST SCRIPT (Phase 2)")
    print("="*60)
    print("\nThis script demonstrates LangGraph-based agentic RAG.")
    print("\nNote: Requires documents to be ingested first!")
    print("      Run: python scripts/ingest_documents.py")
    
    print("\nOptions:")
    print("  1. Interactive mode")
    print("  2. Run all predefined tests")
    print("  3. Test simple question only")
    print("  4. Test RAG question only")
    
    choice = input("\nEnter choice (1-4) or press Enter for option 1: ").strip() or "1"
    
    if choice == "1":
        interactive_mode()
    elif choice == "2":
        test_simple_question()
        test_rag_question()
        test_ambiguous_question()
        test_multiple_questions()
    elif choice == "3":
        test_simple_question()
    elif choice == "4":
        test_rag_question()
    else:
        print("Invalid choice. Running interactive mode...")
        interactive_mode()


if __name__ == "__main__":
    main()
