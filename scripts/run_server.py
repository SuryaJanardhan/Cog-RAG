"""
Script to run the FastAPI server.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from src.config import settings


def main():
    """Run the FastAPI server."""
    print("\n" + "="*60)
    print("STARTING RAG API SERVER")
    print("="*60)
    print(f"\nEnvironment: {settings.environment}")
    print(f"Host: {settings.api_host}")
    print(f"Port: {settings.api_port}")
    print(f"Vector DB: {settings.vector_db}")
    print(f"LLM Model: {settings.gemini_model}")
    print("\n" + "="*60)
    print("\nAPI Documentation will be available at:")
    print(f"  → http://{settings.api_host}:{settings.api_port}/docs")
    print(f"  → http://{settings.api_host}:{settings.api_port}/redoc")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
