# Quick Start Guide

## Prerequisites

- Python 3.10+
- Google Gemini API key ([Get one here](https://ai.google.dev/))
- (Optional) Redis for caching
- (Optional) Qdrant Cloud account for production

## Installation

### 1. Setup Virtual Environment

```bash
# Windows
python -m venv myenv
myenv\Scripts\activate

# Linux/Mac
python -m venv myenv
source myenv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API key
# Required: GEMINI_API_KEY=your_api_key_here
```

Minimum `.env` configuration:

```env
GEMINI_API_KEY=your_gemini_api_key_here
ENVIRONMENT=dev
VECTOR_DB=chroma
```

## Phase 0: Verify Setup

```bash
python main.py
```

This verifies all components are properly configured.

## Phase 1: Basic RAG Pipeline

### Step 1: Ingest Documents

```bash
python scripts/ingest_documents.py
```

This will:

- Create a sample document about RAG systems
- Chunk it into smaller pieces
- Generate embeddings (with caching)
- Store in vector database

### Step 2: Query via CLI

```bash
python scripts/query_rag.py
```

Try questions like:

- "What is RAG?"
- "What are the key components of a RAG system?"
- "What are the benefits of using RAG?"

### Step 3: Run API Server

```bash
python scripts/run_server.py
```

Access the API:

- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

### Step 4: Query via API

Using curl:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "user_id": "test_user",
    "return_sources": true
  }'
```

Using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What is RAG?",
        "user_id": "test_user",
        "return_sources": True
    }
)

result = response.json()
print(result["answer"])
```

## Project Structure

```
RAG-Prod-Level/
├── src/
│   ├── config/          # Settings management
│   ├── db/              # Vector databases
│   ├── cache/           # Caching layers
│   ├── llm/             # Gemini client
│   ├── ingestion/       # Document loading
│   ├── processing/      # Chunking & embedding
│   ├── retrieval/       # Document retrieval
│   ├── rag/             # RAG chain
│   └── api/             # FastAPI app
├── scripts/
│   ├── ingest_documents.py   # Ingest sample data
│   ├── query_rag.py          # CLI queries
│   └── run_server.py         # Start API
├── data/
│   ├── raw/             # Original documents
│   └── processed/       # Processed chunks
├── cache/               # Local cache storage
├── .env                 # Your configuration
└── main.py             # Entry point
```

## Common Issues

### "GEMINI_API_KEY not found"

- Make sure you created `.env` file
- Add `GEMINI_API_KEY=your_key` to `.env`

### "No module named 'src'"

- Run scripts from project root directory
- Make sure virtual environment is activated

### "Connection refused" (Redis)

- If using Redis cache, make sure Redis is running
- Or switch to SQLite cache in `.env`:
  ```env
  EMBEDDING_CACHE_TYPE=sqlite
  RESPONSE_CACHE_TYPE=redis
  ```
- For dev, you can use SQLite for both:
  ```env
  EMBEDDING_CACHE_TYPE=sqlite
  # RESPONSE_CACHE_TYPE still needs Redis for now
  ```

### Chroma initialization issues

- Delete `data/chroma_db/` folder
- Re-run ingestion script

## Next Steps

1. **Add your own documents**:

   - Place PDFs in `data/raw/`
   - Update `scripts/ingest_documents.py`
   - Re-run ingestion

2. **Customize prompts**:

   - Edit `src/rag/chain.py`
   - Modify `DEFAULT_RAG_TEMPLATE`

3. **Adjust retrieval**:

   - Change `RETRIEVAL_TOP_K` in `.env`
   - Adjust `RETRIEVAL_SCORE_THRESHOLD`

4. **Production setup**:
   - Get Qdrant Cloud account
   - Setup Redis server
   - Update `.env` with production credentials
   - Change `ENVIRONMENT=prod`

## Documentation

- [Phase 1 Details](docs/phase1.md)
- [API Documentation](http://localhost:8000/docs) (when server is running)
- [Environment Variables](.env.example)

## Support

- Check logs for detailed error messages
- Verify `.env` configuration
- Ensure all dependencies are installed
- Make sure Gemini API key is valid
