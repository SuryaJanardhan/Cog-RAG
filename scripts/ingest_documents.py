"""
Sample script to ingest documents into the RAG system.
Demonstrates loading various document types and storing them in the vector database.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import create_ingestion_pipeline
from src.processing import create_processing_pipeline


def ingest_sample_text():
    """Ingest a sample text document."""
    print("\n" + "="*60)
    print("SAMPLE DOCUMENT INGESTION")
    print("="*60)
    
    # Create sample text file
    sample_text = """
# Introduction to RAG Systems

Retrieval-Augmented Generation (RAG) is a powerful technique that combines 
information retrieval with text generation. 

## Key Components

1. Document Store: Contains the knowledge base
2. Retriever: Finds relevant documents
3. Generator: Creates answers using retrieved context

## Benefits

- Reduces hallucination by grounding responses in real data
- Allows updating knowledge without retraining
- Provides source attribution for transparency

## Architecture

A typical RAG system consists of:
- Vector database for semantic search
- Embedding model for document representation
- Large language model for generation
- Caching layer for performance

## Use Cases

RAG is particularly useful for:
- Question answering over documents
- Customer support chatbots
- Research assistants
- Documentation search
"""
    
    # Save sample document
    sample_file = Path("./data/raw/sample_rag_intro.txt")
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text(sample_text, encoding='utf-8')
    print(f"\n✓ Created sample document: {sample_file}")
    
    # Initialize pipelines
    print("\n1. Initializing ingestion pipeline...")
    ingestion = create_ingestion_pipeline()
    
    print("2. Initializing processing pipeline...")
    processing = create_processing_pipeline()
    
    # Load document
    print(f"\n3. Loading document: {sample_file}")
    documents = ingestion.load_text_file(
        str(sample_file),
        metadata={"topic": "RAG Systems", "category": "tutorial"}
    )
    print(f"   Loaded {len(documents)} document(s)")
    
    # Normalize documents
    print("\n4. Normalizing documents...")
    documents = ingestion.normalize_documents(documents)
    
    # Process and store
    print("\n5. Processing and storing in vector database...")
    stats = processing.process_and_store(documents, use_cache=True)
    
    # Print statistics
    print("\n" + "="*60)
    print("INGESTION STATISTICS")
    print("="*60)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get ingestion stats
    print("\n" + "="*60)
    print("OVERALL INGESTION STATISTICS")
    print("="*60)
    ingestion_stats = ingestion.get_ingestion_stats()
    for key, value in ingestion_stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Sample document ingestion completed successfully!")
    print("\nYou can now query the system using the API or query script.")


def ingest_web_pages():
    """Ingest sample web pages."""
    print("\n" + "="*60)
    print("WEB PAGE INGESTION EXAMPLE")
    print("="*60)
    
    # Sample URLs (use public documentation)
    urls = [
        "https://python.langchain.com/docs/get_started/introduction",
    ]
    
    print("\nNote: Web scraping requires internet connection.")
    print("Uncomment and customize URLs as needed.\n")
    
    # Initialize pipelines
    ingestion = create_ingestion_pipeline()
    processing = create_processing_pipeline()
    
    try:
        # Load web pages
        print(f"Loading {len(urls)} web page(s)...")
        documents = ingestion.load_web_pages(urls, metadata={"source_type": "documentation"})
        
        # Process and store
        print("Processing and storing...")
        stats = processing.process_and_store(documents, use_cache=True)
        
        print("\n✓ Web pages ingested successfully!")
        print(f"  Chunks created: {stats['chunks_created']}")
        
    except Exception as e:
        print(f"\n✗ Error ingesting web pages: {e}")
        print("Make sure you have internet connection and the URLs are accessible.")


def main():
    """Main ingestion script."""
    print("\n" + "="*60)
    print("RAG DOCUMENT INGESTION SCRIPT")
    print("="*60)
    print("\nThis script demonstrates document ingestion into the RAG system.")
    print("\nOptions:")
    print("  1. Ingest sample text document (recommended for first run)")
    print("  2. Ingest web pages (requires internet)")
    print("  3. Both")
    
    choice = input("\nEnter choice (1-3) or press Enter for option 1: ").strip() or "1"
    
    if choice == "1":
        ingest_sample_text()
    elif choice == "2":
        ingest_web_pages()
    elif choice == "3":
        ingest_sample_text()
        ingest_web_pages()
    else:
        print("Invalid choice. Running default (option 1)...")
        ingest_sample_text()


if __name__ == "__main__":
    main()
