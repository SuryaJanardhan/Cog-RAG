"""
FastAPI application for RAG API endpoints.
Provides REST API for querying the RAG system.
"""
from typing import Optional, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

from ..config import settings
from ..rag import create_rag_pipeline

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Production API",
    description="Production-ready Retrieval-Augmented Generation API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline (will be initialized on first request)
rag_pipeline = None


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="The question to ask", min_length=1)
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    return_sources: bool = Field(True, description="Whether to return source documents")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the main topic of the documents?",
                "user_id": "user123",
                "return_sources": True
            }
        }


class Source(BaseModel):
    """Source document information."""
    source: str
    type: str
    page: Optional[int] = None
    filename: Optional[str] = None
    url: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="The generated answer")
    sources: Optional[List[Source]] = Field(None, description="Source documents used")
    num_sources: Optional[int] = Field(None, description="Number of sources")
    metadata: dict = Field(..., description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The main topic is RAG systems.",
                "sources": [
                    {
                        "source": "docs/rag.pdf",
                        "type": "pdf",
                        "page": 1,
                        "filename": "rag.pdf"
                    }
                ],
                "num_sources": 1,
                "metadata": {
                    "user_id": "user123",
                    "cache_stats": {
                        "cache_hits": 0,
                        "cache_misses": 1,
                        "total_queries": 1,
                        "hit_rate_percent": 0.0
                    }
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    environment: str
    vector_db: str
    cache_enabled: bool


# Helper function to initialize pipeline
def get_rag_pipeline():
    """Get or initialize RAG pipeline."""
    global rag_pipeline
    if rag_pipeline is None:
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = create_rag_pipeline(use_cache=True)
        logger.info("RAG pipeline initialized successfully")
    return rag_pipeline


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Production API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and configuration.
    """
    return HealthResponse(
        status="healthy",
        environment=settings.environment,
        vector_db=settings.vector_db,
        cache_enabled=True
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest):
    """
    Query the RAG system.
    
    Args:
        request: Query request with question and optional user_id
        
    Returns:
        QueryResponse: Answer with sources and metadata
        
    Raises:
        HTTPException: If query processing fails
    """
    try:
        logger.info(f"Received query from user_id={request.user_id}: {request.query[:100]}...")
        
        # Get RAG pipeline
        pipeline = get_rag_pipeline()
        
        # Execute query
        result = pipeline.query(
            question=request.query,
            user_id=request.user_id,
            return_sources=request.return_sources
        )
        
        logger.info(f"Query processed successfully. Cache hit rate: {result['metadata']['cache_stats']['hit_rate_percent']}%")
        
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/query/batch", response_model=List[QueryResponse], tags=["RAG"])
async def batch_query_rag(queries: List[str], user_id: Optional[str] = None):
    """
    Process multiple queries in batch.
    
    Args:
        queries: List of questions
        user_id: Optional user identifier
        
    Returns:
        List[QueryResponse]: List of answers with sources
    """
    try:
        logger.info(f"Received batch query with {len(queries)} questions from user_id={user_id}")
        
        # Get RAG pipeline
        pipeline = get_rag_pipeline()
        
        # Process each query
        results = []
        for query in queries:
            result = pipeline.query(
                question=query,
                user_id=user_id,
                return_sources=True
            )
            results.append(QueryResponse(**result))
        
        logger.info(f"Batch query processed successfully")
        return results
    
    except Exception as e:
        logger.error(f"Error processing batch query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing batch query: {str(e)}"
        )


@app.get("/stats", tags=["Statistics"])
async def get_stats():
    """
    Get RAG system statistics.
    
    Returns cache stats and system information.
    """
    try:
        pipeline = get_rag_pipeline()
        cache_stats = pipeline.rag_chain.get_cache_stats()
        
        return {
            "cache_stats": cache_stats,
            "config": {
                "environment": settings.environment,
                "vector_db": settings.vector_db,
                "llm_model": settings.gemini_model,
                "retrieval_k": settings.retrieval_top_k,
                "chunk_size": settings.chunk_size
            }
        }
    
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching stats: {str(e)}"
        )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting RAG API server...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Vector DB: {settings.vector_db}")
    logger.info(f"LLM Model: {settings.gemini_model}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG API server...")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )
