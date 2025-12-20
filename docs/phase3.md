# Phase 3: LlamaIndex Integration

## Overview

Phase 3 adds advanced retrieval capabilities through LlamaIndex integration, providing multiple query engines, router-based source selection, and sub-question decomposition for complex queries.

## Architecture

### Components

```
src/llamaindex/
├── __init__.py              # Module exports
├── index_manager.py         # VectorStoreIndex management
├── query_engines.py         # Advanced query engines
└── tools.py                 # LangChain tool wrappers
```

### Key Classes

#### 1. LlamaIndexManager
Manages VectorStoreIndex with Qdrant/Chroma backend.

**Features:**
- Create/load indices from documents
- Add documents to existing index
- Get retriever or query engine
- Shared vector store with Phase 1
- Automatic embedding with Gemini

**Methods:**
```python
manager = LlamaIndexManager()
manager.create_index(documents)            # Create from documents
manager.add_documents(new_docs)            # Add to existing
retriever = manager.get_retriever(k=5)     # Get retriever
query_engine = manager.get_query_engine()  # Get query engine
```

#### 2. Query Engines

**CompactEngine:**
- Concatenates retrieved chunks
- Single LLM call for synthesis
- Best for: Straightforward questions
- Response mode: `compact`, `refine`, `tree_summarize`, `simple_summarize`

```python
engine = CompactEngine(index, response_mode="compact")
response = engine.query("What is RAG?")
```

**RouterEngine:**
- Routes queries to multiple indices
- LLM-based source selection
- Best for: Multi-source knowledge bases
- Example: Technical docs vs general docs

```python
router = RouterEngine({
    "technical": tech_index,
    "general": gen_index,
})
response = router.query("Explain vector databases")
```

**SubQuestionEngine:**
- Breaks complex queries into sub-questions
- Answers each independently
- Synthesizes final response
- Best for: Multi-part questions

```python
subq_engine = SubQuestionEngine({
    "embeddings": embed_index,
    "search": search_index,
})
response = subq_engine.query("How do embeddings enable similarity search?")
```

**HybridEngine:**
- Automatic mode selection
- Combines compact, router, and sub-question
- Chooses best strategy based on query

```python
hybrid = HybridEngine(primary_index, secondary_indices)
response = hybrid.query(query, mode="auto")  # Auto-selects mode
```

### 3. LangChain Tool Wrappers

Enables LlamaIndex query engines to be used in LangGraph workflows.

**LlamaIndexTools:**
```python
tools = LlamaIndexTools(index_manager)
tool_list = tools.get_tools()  # Returns LangChain tools

# Available tools:
# - llamaindex_compact_query
# - llamaindex_refine_query
# - llamaindex_tree_query

# Optional advanced tools:
tools.add_router_tool(indices)
tools.add_subquestion_tool(indices)
tools.add_hybrid_tool(primary, secondary)
```

## Integration with LangGraph

### Updated Workflow

Phase 3 integrates with Phase 2's agentic RAG by enhancing the retrieval node:

```python
# Create agentic graph with LlamaIndex
graph = AgenticRAGGraph(use_llamaindex=True)

# The retrieve node now uses LlamaIndex retriever
# when use_llamaindex=True
```

**Retrieval Flow:**
```
classify_or_answer
    ↓ (needs_retrieval=True)
retrieve (uses LlamaIndex if enabled)
    ↓
grade_documents
    ↓
[relevant] → generate_answer
[not relevant] → rewrite_question → classify
```

### Node Updates

**RAGNodes.__init__:**
- Accepts `use_llamaindex` parameter
- Initializes LlamaIndexManager if enabled
- Falls back to LangChain on error

**RAGNodes.retrieve:**
- Checks `self.use_llamaindex` flag
- Uses LlamaIndex retriever or LangChain retriever
- Converts LlamaIndex nodes to document format

## Configuration

### Settings (.env)

```bash
# LlamaIndex Configuration
LLAMAINDEX_RESPONSE_MODE=compact  # compact, refine, tree_summarize
LLAMAINDEX_USE_ROUTER=false
LLAMAINDEX_USE_SUBQUESTION=false
LLAMAINDEX_ENABLE_HYBRID=true
```

### Config Fields (src/config/settings.py)

```python
llamaindex_response_mode: Literal["compact", "refine", "tree_summarize", "simple_summarize"]
llamaindex_use_router: bool
llamaindex_use_subquestion: bool
llamaindex_enable_hybrid: bool
```

## Usage Examples

### 1. Basic Compact Query

```python
from src.llamaindex import LlamaIndexManager
from src.llamaindex.query_engines import CompactEngine
from llama_index.core import Document

# Create documents
docs = [
    Document(text="RAG combines retrieval with generation..."),
    Document(text="Vector embeddings capture semantic meaning..."),
]

# Initialize and query
manager = LlamaIndexManager()
manager.create_index(docs)

engine = CompactEngine(manager.index)
response = engine.query("What is RAG?")
print(response)
```

### 2. Router for Multi-Source

```python
from src.llamaindex.query_engines import RouterEngine

# Create separate indices
tech_manager = LlamaIndexManager()
tech_index = tech_manager.create_index(technical_docs)

gen_manager = LlamaIndexManager()
gen_index = gen_manager.create_index(general_docs)

# Create router
router = RouterEngine({
    "technical": tech_index,
    "general": gen_index,
})

# Router automatically selects best source
response = router.query("Explain vector databases")
```

### 3. Sub-Question Decomposition

```python
from src.llamaindex.query_engines import SubQuestionEngine

# Create domain-specific indices
embed_index = ...  # Embeddings documentation
search_index = ...  # Search documentation

# Create engine
subq_engine = SubQuestionEngine({
    "embeddings": embed_index,
    "search": search_index,
})

# Ask complex question
response = subq_engine.query(
    "How do embeddings enable similarity search and what metrics are used?"
)
# Breaks into:
# 1. "How do embeddings enable similarity search?" → embeddings index
# 2. "What metrics are used for similarity?" → search index
# Synthesizes final answer
```

### 4. Hybrid with Auto Mode

```python
from src.llamaindex.query_engines import HybridEngine

# Create hybrid engine
hybrid = HybridEngine(
    primary_index=main_index,
    secondary_indices={
        "technical": tech_index,
        "research": research_index,
    }
)

# Auto-selects best mode
response = hybrid.query(query, mode="auto")
# Simple questions → compact mode
# Multi-part questions → subquestion mode
# Multi-source questions → router mode
```

### 5. LangGraph Integration

```python
from src.graph.workflow import AgenticRAGGraph

# Create graph with LlamaIndex
graph = AgenticRAGGraph(use_llamaindex=True)

# Query flows through LlamaIndex retriever
response = graph.query("What is RAG?")
```

### 6. LangChain Tools for LangGraph

```python
from src.llamaindex.tools import LlamaIndexTools

# Create tools
manager = LlamaIndexManager()
manager.create_index(docs)

tools = LlamaIndexTools(manager)
tool_list = tools.get_tools()

# Use in LangGraph nodes
# Tools can be called from call_tools node
for tool in tool_list:
    print(f"{tool.name}: {tool.description}")
```

## Response Synthesis Modes

### Compact Mode (Default)
- **How it works:** Concatenates chunks, single LLM call
- **Pros:** Fast, efficient token usage
- **Cons:** May miss nuances if too many chunks
- **Best for:** Clear, factual questions

### Refine Mode
- **How it works:** Iteratively refines answer across chunks
- **Pros:** Detailed, comprehensive answers
- **Cons:** Slower, higher token usage (multiple LLM calls)
- **Best for:** Complex explanations, analysis

### Tree Summarize Mode
- **How it works:** Builds hierarchical summary tree
- **Pros:** Good for summarization tasks
- **Cons:** More complex, slower
- **Best for:** Overviews, summaries

### Simple Summarize Mode
- **How it works:** Truncates chunks to fit context
- **Pros:** Very fast, simple
- **Cons:** May lose information
- **Best for:** Quick answers with limited context

## Testing

### Test Script
```bash
python scripts/test_llamaindex.py
```

**Tests:**
1. Basic indexing and compact query
2. Response synthesis modes (compact, refine, tree)
3. Router engine with multiple indices
4. Sub-question query decomposition
5. Hybrid engine with auto mode selection
6. LangChain tool wrappers

**Interactive Mode:**
After tests, enter interactive mode to ask questions.

### Comparison Script
```bash
python scripts/compare_retrievers.py
```

Compares:
- LangChain retriever (Phase 1)
- LlamaIndex compact engine
- LlamaIndex refine engine

Metrics:
- Retrieval time
- Number of sources/documents
- Response quality
- Token usage

## API Integration

### Using LlamaIndex in API

The API can use LlamaIndex through the agentic RAG graph:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "use_agentic": true
  }'
```

When `use_agentic=true` and `LLAMAINDEX_ENABLE_HYBRID=true`, the retrieve node uses LlamaIndex.

### Configuration Priority

1. `use_llamaindex=True` in `AgenticRAGGraph()` → Always use LlamaIndex
2. `LLAMAINDEX_ENABLE_HYBRID=true` in .env → Use when agentic mode enabled
3. Default → Use LangChain retriever

## Performance Considerations

### Speed Comparison
- **LangChain retriever:** Fastest (pure retrieval)
- **LlamaIndex compact:** Moderate (retrieval + synthesis)
- **LlamaIndex refine:** Slowest (multiple LLM calls)

### When to Use Each

**LangChain Retriever:**
- Need raw documents
- Building custom synthesis
- Performance critical

**LlamaIndex Compact:**
- Want natural language responses
- Balanced speed/quality
- General use cases

**LlamaIndex Refine:**
- Need detailed explanations
- Quality over speed
- Complex analysis

**Router Engine:**
- Multiple knowledge sources
- Domain-specific routing
- Clear topic boundaries

**Sub-Question Engine:**
- Complex multi-part questions
- Cross-domain queries
- Need decomposition

## Debugging

### Enable Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

**1. LlamaIndex not initializing:**
```
[INIT] LlamaIndex initialization failed: ...
```
- Check `llama-index` packages installed
- Verify `GEMINI_API_KEY` set
- Ensure vector store accessible

**2. Empty responses:**
- Check if index has documents
- Verify embedding model working
- Check retrieval_top_k setting

**3. Slow queries:**
- Use compact mode instead of refine
- Reduce similarity_top_k
- Check vector store performance

### Verbose Output

Nodes print execution info:
```
[INIT] LlamaIndex retriever initialized
[RETRIEVE] Using LlamaIndex retriever
[RETRIEVE] Found 5 documents
```

## Comparison: LangChain vs LlamaIndex

| Feature | LangChain (Phase 1) | LlamaIndex (Phase 3) |
|---------|---------------------|----------------------|
| **Retrieval** | ✓ Fast, simple | ✓ Advanced strategies |
| **Synthesis** | Manual (RAG chain) | ✓ Built-in modes |
| **Multi-source** | Manual routing | ✓ Router engine |
| **Sub-questions** | Not supported | ✓ Decomposition |
| **Speed** | ✓ Fastest | Moderate |
| **Ease of use** | Moderate | ✓ High-level APIs |
| **Flexibility** | ✓ Full control | Opinionated |

## Next Steps

**Phase 4: Production Hardening**
- Multi-tenant support
- Background ingestion workers
- Rate limiting
- Advanced monitoring
- Offline evaluation

## Files Created

- `src/llamaindex/__init__.py`
- `src/llamaindex/index_manager.py`
- `src/llamaindex/query_engines.py`
- `src/llamaindex/tools.py`
- `scripts/test_llamaindex.py`
- `scripts/compare_retrievers.py`
- `docs/phase3.md` (this file)

## Configuration Files Updated

- `src/config/settings.py` - Added LlamaIndex settings
- `.env.example` - Added LlamaIndex configuration
- `src/graph/nodes.py` - Added LlamaIndex retrieval path
- `src/graph/workflow.py` - Added use_llamaindex parameter
