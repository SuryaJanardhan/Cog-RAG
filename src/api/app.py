"""
FastAPI application for RAG API endpoints.
Provides REST API for querying the RAG system.
"""
from typing import Optional, List
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

from ..config import settings
from ..rag import create_rag_pipeline
from ..graph import create_agentic_rag

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Production API",
    description="Production-ready Retrieval-Augmented Generation API with Agentic Mode",
    version="2.0.0"
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
agentic_rag = None


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="The question to ask", min_length=1)
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    return_sources: bool = Field(True, description="Whether to return source documents")
    use_agentic: bool = Field(False, description="Use agentic RAG with LangGraph (Phase 2)")
    llm_provider: Optional[str] = Field("gemini", description="LLM provider: gemini, openai, groq")
    llm_api_key: Optional[str] = Field(None, description="Optional custom API key for the provider")
    llm_model: Optional[str] = Field(None, description="Optional custom model name")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the main topic of the documents?",
                "user_id": "user123",
                "return_sources": True,
                "use_agentic": False,
                "llm_provider": "gemini",
                "llm_api_key": None,
                "llm_model": None
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


def get_agentic_rag():
    """Get or initialize agentic RAG graph."""
    global agentic_rag
    if agentic_rag is None:
        logger.info("Initializing Agentic RAG (LangGraph)...")
        agentic_rag = create_agentic_rag()
        logger.info("Agentic RAG initialized successfully")
    return agentic_rag


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Production API with Agentic Mode",
        "version": "2.0.0",
        "docs": "/docs",
        "features": {
            "phase1": "Basic RAG with caching",
            "phase2": "Agentic RAG with LangGraph"
        }
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


def get_custom_llm(provider: Optional[str], api_key: Optional[str], model: Optional[str]):
    """Instantiate a custom LangChain LLM model based on provider, key, and model name."""
    if not provider:
        return None
    provider = provider.lower()
    
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            final_key = api_key or settings.openai_api_key
            if not final_key:
                raise ValueError("OpenAI API key not configured")
            return ChatOpenAI(
                model=model or "gpt-4o",
                openai_api_key=final_key,
                temperature=0.2
            )
        except Exception as e:
            logger.error(f"Failed to load OpenAI model: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to load OpenAI model: {str(e)}")
            
    elif provider == "groq":
        try:
            from langchain_groq import ChatGroq
            final_key = api_key or settings.groq_api_key
            if not final_key:
                raise ValueError("Groq API key not configured")
            return ChatGroq(
                model=model or "llama3-8b-8192",
                groq_api_key=final_key,
                temperature=0.2
            )
        except Exception as e:
            logger.error(f"Failed to load Groq model: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to load Groq model: {str(e)}")
            
    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            final_key = api_key or settings.gemini_api_key
            if not final_key:
                raise ValueError("Gemini API key not configured")
            return ChatGoogleGenerativeAI(
                model=model or settings.gemini_model,
                google_api_key=final_key,
                temperature=settings.gemini_temperature,
                max_output_tokens=settings.gemini_max_output_tokens,
            )
        except Exception as e:
            logger.error(f"Failed to load Gemini model: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to load Gemini model: {str(e)}")
            
    return None


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
        logger.info(f"Mode: {'Agentic (Phase 2)' if request.use_agentic else 'Basic (Phase 1)'}")
        
        # Get custom LLM if requested
        custom_llm = get_custom_llm(
            provider=request.llm_provider,
            api_key=request.llm_api_key,
            model=request.llm_model
        )
        
        if request.use_agentic:
            # Use agentic RAG with LangGraph
            if custom_llm:
                from ..graph.workflow import AgenticRAGGraph
                # Dynamically instantiate a local graph config for thread safety
                graph = AgenticRAGGraph()
                graph.nodes.llm = custom_llm
                if hasattr(graph.nodes, "llm_client") and graph.nodes.llm_client:
                    graph.nodes.llm_client._chat_model = custom_llm
            else:
                graph = get_agentic_rag()
                
            graph_result = graph.invoke(request.query)
            
            # Convert to standard response format
            result = {
                "answer": graph_result["answer"],
                "sources": [],  # Extract from documents
                "num_sources": len(graph_result.get("documents", [])),
                "metadata": {
                    "user_id": request.user_id,
                    "mode": "agentic",
                    "retry_count": graph_result.get("retry_count", 0),
                    "retrieval_attempted": graph_result.get("retrieval_attempted", False),
                    "question_rewritten": graph_result.get("question") != request.query
                }
            }
            
            # Extract sources if requested
            if request.return_sources and graph_result.get("documents"):
                from ..retrieval import create_retriever
                retriever = create_retriever()
                result["sources"] = retriever.get_sources(graph_result["documents"])
        else:
            # Use basic RAG pipeline
            if custom_llm:
                from ..rag.chain import RAGChain
                pipeline = get_rag_pipeline()
                # Dynamically compile a local chain to prevent thread collisions
                local_chain = RAGChain(retriever=pipeline.rag_chain.retriever, use_cache=pipeline.rag_chain.use_cache)
                local_chain.llm = custom_llm
                local_chain.chain = local_chain._build_chain()
                
                result = local_chain.query(request.query, return_sources=request.return_sources)
                result["metadata"] = {
                    "user_id": request.user_id,
                    "mode": "basic",
                    "cache_stats": local_chain.get_cache_stats()
                }
            else:
                pipeline = get_rag_pipeline()
                result = pipeline.query(
                    question=request.query,
                    user_id=request.user_id,
                    return_sources=request.return_sources
                )
        
        logger.info(f"Query processed successfully")
        return QueryResponse(**result)
        
    except HTTPException:
        raise
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


@app.post("/upload", tags=["Ingestion"])
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload documents and index them in the RAG vector store.
    """
    try:
        import os
        import shutil
        from ..ingestion.ingestion_sync import IncrementalIngestionManager
        
        os.makedirs("./data", exist_ok=True)
        ingest_manager = IncrementalIngestionManager()
        results = []
        for file in files:
            file_path = os.path.join("./data", file.filename)
            # Save the file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Run synchronization
            logger.info(f"Ingesting file: {file.filename}")
            processed = ingest_manager.sync_file(file_path)
            results.append({
                "filename": file.filename,
                "status": "indexed" if processed else "skipped (unchanged)",
                "path": file_path
            })
        return {"message": "Upload complete", "results": results}
    except Exception as e:
        logger.error(f"Failed to upload documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/documents", tags=["Ingestion"])
async def list_documents():
    """
    List all indexed documents in the RAG ingestion catalog.
    """
    try:
        import os
        import sqlite3
        catalog_path = "./data/ingestion_catalog.db"
        if not os.path.exists(catalog_path):
            return {"documents": []}
        conn = sqlite3.connect(catalog_path)
        cursor = conn.cursor()
        cursor.execute("SELECT filepath, file_hash, last_modified FROM file_catalog")
        rows = cursor.fetchall()
        conn.close()
        
        docs = []
        for row in rows:
            filepath, file_hash, mtime = row
            docs.append({
                "filename": os.path.basename(filepath),
                "filepath": filepath,
                "file_hash": file_hash,
                "last_modified": mtime
            })
        return {"documents": docs}
    except Exception as e:
        logger.error(f"Failed to fetch documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
