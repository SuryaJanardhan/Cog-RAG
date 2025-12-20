# Phase 1: Basic RAG Pipeline

## Overview

Phase 1 implements a complete RAG pipeline with LangChain, Gemini, and caching.

## Components

### 1. Document Ingestion (`src/ingestion/`)

- **DocumentIngestion**: Loads various document types
  - PDF files via PyPDFLoader
  - Web pages via WebBaseLoader
  - Text files and directories
  - Word documents and PowerPoints
- Metadata tracking and normalization
- Storage in `data/raw/` with metadata.json

### 2. Processing Pipeline (`src/processing/`)

- **ChunkingPipeline**: RecursiveCharacterTextSplitter
  - Configurable chunk_size (default: 1000)
  - Configurable chunk_overlap (default: 200)
- **EmbeddingPipeline**: Gemini embeddings with caching
  - Cache checking before generating embeddings
  - Batch embedding support
  - Cache statistics tracking
- **VectorStorePipeline**: Vector storage management
  - Automatic Qdrant/Chroma selection based on environment
  - Document addition with embedding caching

### 3. Retrieval (`src/retrieval/`)

- **CachedRetriever**: Configurable document retrieval
  - Top-K retrieval (default: 5)
  - Score threshold filtering (default: 0.7)
  - Similarity and MMR search modes
  - Document formatting and source extraction

### 4. RAG Chain (`src/rag/`)

- **RAGChain**: Complete RAG pipeline
  - Context injection with prompt templates
  - Gemini chat model integration
  - Response caching with query+doc_ids key
  - Cache hit/miss tracking
- **RAGPipeline**: High-level API wrapper
  - User ID tracking
  - Metadata injection
  - Custom chain creation

### 5. API (`src/api/`)

- **FastAPI Application**: REST API endpoints
  - `POST /query`: Single query with sources
  - `POST /query/batch`: Batch queries
  - `GET /health`: Health check
  - `GET /stats`: Cache and system statistics
  - CORS middleware
  - Error handling and logging

## Usage

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 2. Ingest Documents

```bash
python scripts/ingest_documents.py
```

This creates a sample document and stores it in the vector database.

### 3. Query via CLI

```bash
python scripts/query_rag.py
```

Interactive or batch query mode.

### 4. Run API Server

```bash
python scripts/run_server.py
```

Access API at http://localhost:8000/docs

### 5. Query via API

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "user_id": "test_user",
    "return_sources": true
  }'
```

## Caching Architecture

### Embedding Cache

- **Before embedding**: Check cache by text hash
- **Storage**: SQLite (dev) or Redis (prod)
- **Key**: SHA256 hash of text content
- **Value**: Embedding vector as JSON

### Response Cache

- **Before LLM**: Build cache key from (query + retrieved_doc_ids)
- **Storage**: Redis or PostgreSQL
- **Key**: SHA256 hash of query + sorted doc IDs
- **Value**: Generated answer
- **TTL**: Configurable (default: 3600s)

## API Endpoints

### POST /query

Request:

```json
{
  "query": "What is RAG?",
  "user_id": "user123",
  "return_sources": true
}
```

Response:

```json
{
  "answer": "RAG is...",
  "sources": [
    {
      "source": "data/raw/sample.txt",
      "type": "text",
      "filename": "sample.txt"
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
```

## Configuration

Key settings in `.env`:

- `GEMINI_API_KEY`: Required
- `VECTOR_DB`: chroma (dev) or qdrant (prod)
- `CHUNK_SIZE`: Default 1000
- `CHUNK_OVERLAP`: Default 200
- `RETRIEVAL_TOP_K`: Default 5
- `RETRIEVAL_SCORE_THRESHOLD`: Default 0.7
- `EMBEDDING_CACHE_TYPE`: sqlite or redis
- `RESPONSE_CACHE_TYPE`: redis or postgres

## Testing Flow

1. **Verify Phase 0**:

   ```bash
   python main.py
   ```

2. **Ingest sample data**:

   ```bash
   python scripts/ingest_documents.py
   ```

3. **Test queries**:

   ```bash
   python scripts/query_rag.py
   ```

4. **Start API server**:

   ```bash
   python scripts/run_server.py
   ```

5. **Test API**:
   - Visit http://localhost:8000/docs
   - Try the interactive Swagger UI

## Next Steps

Phase 1 provides a solid foundation. Ready for:

- **Phase 2**: Add LangGraph for agentic RAG
- **Phase 3**: Integrate LlamaIndex for advanced retrieval
- **Phase 4**: Production hardening and scaling

## Key Features Implemented

✅ Document ingestion (PDF, web, text, Word)
✅ Chunking with RecursiveCharacterTextSplitter
✅ Gemini embeddings with caching
✅ Vector storage (Qdrant/Chroma)
✅ Retrieval with score thresholding
✅ RAG chain with context injection
✅ Response caching
✅ FastAPI REST endpoints
✅ Interactive query scripts
✅ Batch processing
✅ Health checks and statistics
