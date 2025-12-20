# RAG Production-Level System

A production-ready Retrieval-Augmented Generation (RAG) system built with LangChain, LangGraph, LlamaIndex, and Google Gemini.

trying to build one last solution on llm and advn rag pipeline where it can be later used as a ready to prod level scalable and reliable system in any segment with custom config :::

## 🎯 Project Overview

This project implements a multi-phase RAG system with:

- **LangChain** for RAG primitives
- **LangGraph** for agentic control flow
- **LlamaIndex** for advanced retrieval experiments
- **Google Gemini** as the primary LLM
- **Qdrant** (prod) / **Chroma** (dev) for vector storage
- **Redis** / **SQLite** for caching

## 📋 Project Phases

### ✅ Phase 0: Tech Stack Setup (Complete)

- Framework configuration locked
- Environment management with Pydantic settings
- Vector database initialization (Qdrant/Chroma)
- Cache layer setup (Redis/SQLite/Postgres)
- Gemini LLM client wrapper

### ✅ Phase 1: Basic RAG Pipeline (Complete)

- Document ingestion and processing
- Chunking and embeddings with caching
- Vector retrieval with score thresholding
- RAG chain with context injection
- Response caching
- FastAPI REST endpoints
- CLI query tools

### 🔄 Phase 2: LangGraph Orchestration

- Agentic RAG with adaptive retrieval
- Multi-node graph workflow
- Tool integration
- Query rewriting and grading

### 🔄 Phase 3: LlamaIndex Integration

- Advanced indexes and routers
- Sub-question decomposition
- Config-driven retrieval

### 🔄 Phase 4: Production Hardening

- Multi-tenant support
- Live web search integration
- Observability and tracing
- Scalability patterns

## 🏗️ Project Structure

```
RAG-Prod-Level/
├── src/
│   ├── config/          # Settings and environment configuration
│   ├── db/              # Vector database clients (Qdrant, Chroma)
│   ├── cache/           # Embedding and response caching
│   ├── llm/             # Gemini LLM wrapper
│   ├── ingestion/       # Document loaders (PDF, web, text, Word)
│   ├── processing/      # Chunking and embedding pipeline
│   ├── retrieval/       # Document retrieval with caching
│   ├── rag/             # RAG chain implementation
│   └── api/             # FastAPI REST endpoints
├── scripts/
│   ├── ingest_documents.py   # Document ingestion script
│   ├── query_rag.py          # Interactive query CLI
│   └── run_server.py         # API server launcher
├── data/
│   ├── raw/             # Raw documents
│   └── processed/       # Processed chunks
├── cache/               # Local cache storage
├── docs/                # Documentation
├── tests/               # Test suite
├── .env.example         # Environment template
├── requirements.txt     # Python dependencies
└── main.py             # Application entry point
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Google Gemini API key
- (Optional) Qdrant Cloud account for production
- (Optional) Redis for caching

### Installation

1. **Clone and navigate to the project:**

```bash
cd RAG-Prod-Level
```

2. **Create and activate virtual environment:**

```bash
python -m venv myenv
# Windows
myenv\Scripts\activate
# Linux/Mac
source myenv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Configure environment:**

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Configuration

Edit `.env` file with your settings:

```env
# Required
GEMINI_API_KEY=your_api_key_here

# Choose environment
ENVIRONMENT=dev  # or prod

# Vector DB (dev uses Chroma, prod uses Qdrant)
VECTOR_DB=chroma  # or qdrant
```

### Running Phase 0 Verification

```bash
python main.py
```

This will verify all components are correctly configured.

### Running Phase 1: Basic RAG

#### 1. Ingest Documents

```bash
python scripts/ingest_documents.py
```

Creates a sample document and stores it in the vector database.

#### 2. Query via CLI

```bash
python scripts/query_rag.py
```

Interactive query mode with cache statistics.

#### 3. Run API Server

```bash
python scripts/run_server.py
```

Access API at:

- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Stats: http://localhost:8000/stats

#### 4. Query via API

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "user_id": "test", "return_sources": true}'
```

See [Quick Start Guide](docs/quickstart.md) for detailed instructions.

## 🔧 Tech Stack

| Component           | Development                      | Production           |
| ------------------- | -------------------------------- | -------------------- |
| **LLM**             | Gemini 1.5 Flash                 | Gemini 1.5 Flash/Pro |
| **Vector DB**       | Chroma (local)                   | Qdrant Cloud         |
| **Embedding Cache** | SQLite                           | Redis                |
| **Response Cache**  | Redis                            | Redis/Postgres       |
| **Frameworks**      | LangChain, LangGraph, LlamaIndex |

## 📝 Environment Variables

See [.env.example](.env.example) for all available configuration options.

Key variables:

- `GEMINI_API_KEY` - Google Gemini API key
- `VECTOR_DB` - Vector database choice (chroma/qdrant)
- `ENVIRONMENT` - Running mode (dev/prod)
- `QDRANT_URL` - Qdrant Cloud URL (for prod)
- `REDIS_HOST` - Redis server (for caching)

## 🧪 Testing

```bash
pytest tests/
```

## 📚 Documentation

- [Quick Start Guide](docs/quickstart.md)
- [Phase 1 Details](docs/phase1.md)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Gemini API Documentation](https://ai.google.dev/docs)

## 🛣️ Roadmap

- [x] Phase 0: Tech stack configuration
- [x] Phase 1: Basic RAG pipeline
- [ ] Phase 2: Agentic orchestration
- [ ] Phase 3: Advanced retrieval
- [ ] Phase 4: Production features

## 📄 License

MIT

---

**Current Status:** Phase 1 Complete ✅

**Next:** Phase 2 - LangGraph Orchestration (Agentic RAG)
