"""
Sample script to query the RAG system.
Demonstrates querying with various options.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import create_rag_pipeline


def query_interactive():
    """Interactive query mode."""
    print("\n" + "="*60)
    print("RAG INTERACTIVE QUERY")
    print("="*60)
    
    print("\nInitializing RAG pipeline...")
    pipeline = create_rag_pipeline(use_cache=True)
    print("✓ Pipeline ready!")
    
    print("\nEnter your questions (type 'exit' to quit, 'stats' for cache statistics)")
    print("-" * 60)
    
    while True:
        question = input("\nQuestion: ").strip()
        
        if not question:
            continue
        
        if question.lower() == 'exit':
            print("\nGoodbye!")
            break
        
        if question.lower() == 'stats':
            stats = pipeline.rag_chain.get_cache_stats()
            print("\n" + "="*60)
            print("CACHE STATISTICS")
            print("="*60)
            for key, value in stats.items():
                print(f"  {key}: {value}")
            continue
        
        print("\nProcessing query...")
        try:
            result = pipeline.query(
                question=question,
                user_id="cli_user",
                return_sources=True
            )
            
            print("\n" + "="*60)
            print("ANSWER")
            print("="*60)
            print(result['answer'])
            
            if result.get('sources'):
                print("\n" + "="*60)
                print(f"SOURCES ({result['num_sources']})")
                print("="*60)
                for i, source in enumerate(result['sources'], 1):
                    print(f"\n{i}. {source['source']}")
                    print(f"   Type: {source['type']}")
                    if 'page' in source:
                        print(f"   Page: {source['page']}")
            
            # Show cache info
            cache_stats = result['metadata']['cache_stats']
            cache_status = "HIT" if cache_stats['cache_hits'] > 0 else "MISS"
            print(f"\nCache: {cache_status} | Hit Rate: {cache_stats['hit_rate_percent']}%")
            
        except Exception as e:
            print(f"\n✗ Error: {e}")


def query_batch():
    """Batch query mode with predefined questions."""
    print("\n" + "="*60)
    print("RAG BATCH QUERY")
    print("="*60)
    
    questions = [
        "What is RAG?",
        "What are the key components of a RAG system?",
        "What are the benefits of using RAG?",
        "What are some use cases for RAG?",
    ]
    
    print(f"\nRunning {len(questions)} queries...")
    print("\nInitializing RAG pipeline...")
    pipeline = create_rag_pipeline(use_cache=True)
    
    for i, question in enumerate(questions, 1):
        print("\n" + "="*60)
        print(f"QUERY {i}/{len(questions)}")
        print("="*60)
        print(f"Question: {question}")
        
        try:
            result = pipeline.query(
                question=question,
                user_id="batch_user",
                return_sources=False
            )
            
            print(f"\nAnswer: {result['answer'][:200]}...")
            
            cache_stats = result['metadata']['cache_stats']
            cache_status = "HIT" if cache_stats['cache_hits'] > cache_stats['cache_misses'] else "MISS"
            print(f"Cache: {cache_status}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Final stats
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    stats = pipeline.rag_chain.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main():
    """Main query script."""
    print("\n" + "="*60)
    print("RAG QUERY SCRIPT")
    print("="*60)
    print("\nThis script demonstrates querying the RAG system.")
    print("\nNote: Make sure you've ingested documents first!")
    print("      Run: python scripts/ingest_documents.py")
    
    print("\nOptions:")
    print("  1. Interactive mode (ask questions)")
    print("  2. Batch mode (predefined questions)")
    
    choice = input("\nEnter choice (1-2) or press Enter for option 1: ").strip() or "1"
    
    if choice == "1":
        query_interactive()
    elif choice == "2":
        query_batch()
    else:
        print("Invalid choice. Running default (option 1)...")
        query_interactive()


if __name__ == "__main__":
    main()
