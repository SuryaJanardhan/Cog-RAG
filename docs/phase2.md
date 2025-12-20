# Phase 2: LangGraph Orchestration (Agentic RAG)

## Overview

Phase 2 adds intelligent, adaptive RAG using LangGraph for orchestration. The system now decides when to retrieve, rewrites unclear queries, and grades document relevance.

## Key Concepts

### Agentic Behavior

Unlike Phase 1's linear pipeline, Phase 2 uses a **graph-based workflow** that:

- **Decides** if retrieval is needed
- **Adapts** to document quality by rewriting queries
- **Routes** through different paths based on context
- **Loops** when needed (with limits) to improve results

## Architecture

### Graph State (`src/graph/state.py`)

State that flows through the graph:

```python
{
    "question": str,              # Current question (may be rewritten)
    "documents": List[Document],   # Retrieved documents
    "answer": str,                # Final answer
    "retry_count": int,           # Number of rewrites
    "needs_retrieval": bool,      # Whether to retrieve
    "retrieval_attempted": bool,  # If retrieval was tried
    "use_tools": bool,            # Whether to use external tools
    "tool_results": str,          # Results from tools
    "error": str                  # Any errors
}
```

### Nodes (`src/graph/nodes.py`)

#### 1. classify_or_answer

**Purpose**: Analyze question and decide next step

- Uses Gemini to classify query type
- Determines if retrieval needed
- Checks if external tools required
- Can answer directly for simple questions

**Outputs**: Sets `needs_retrieval` and `use_tools` flags

#### 2. retrieve

**Purpose**: Fetch relevant documents

- Uses Phase 1 retriever
- Performs vector similarity search
- Stores documents in state

**Outputs**: Updates `documents` and `retrieval_attempted`

#### 3. grade_documents

**Purpose**: Check document relevance

- Uses LLM to grade documents
- Decides if they're sufficient for answering
- Triggers rewrite if documents are poor

**Outputs**: Updates `needs_retrieval` based on quality

#### 4. rewrite_question

**Purpose**: Improve query for better retrieval

- Rephrases ambiguous questions
- Adds specificity and keywords
- Increments retry counter

**Outputs**: Updates `question` and `retry_count`

#### 5. generate_answer

**Purpose**: Create final response

- Uses retrieved context
- Falls back to general knowledge
- Formats final answer

**Outputs**: Sets `answer`

#### 6. call_tools (Phase 2+)

**Purpose**: Execute external tools

- Web search via Tavily
- Calculator for math
- HTTP fetch for URLs

**Outputs**: Sets `tool_results`

### Workflow (`src/graph/workflow.py`)

```
START
  ↓
classify_or_answer
  ├─→ (needs retrieval) → retrieve → grade_documents
  │                         ├─→ (relevant) → generate_answer → END
  │                         └─→ (not relevant & retries left) → rewrite_question → classify_or_answer
  │                         └─→ (max retries) → generate_answer → END
  ├─→ (needs tools) → call_tools → generate_answer → END
  └─→ (direct answer) → generate_answer → END
```

### Routing Logic

**After classification**:

- If `use_tools` = True → call_tools
- If `needs_retrieval` = True → retrieve
- Otherwise → generate_answer (direct)

**After grading**:

- If documents relevant → generate_answer
- If not relevant & retries < MAX → rewrite_question
- If max retries reached → generate_answer (with what we have)

**After rewriting**:

- If retries < MAX → classify_or_answer (try again)
- If max retries → generate_answer (give up)

## Tools (`src/tools/registry.py`)

### Available Tools

1. **Calculator**

   - Basic math operations
   - Supports: +, -, \*, /, sqrt, sin, cos, pow
   - Example: "2+2", "sqrt(16)"

2. **Web Search** (Tavily)

   - Requires TAVILY_API_KEY
   - Free tier: 1000 searches/month
   - Returns top 3 results

3. **HTTP Fetch**
   - Fetches webpage content
   - Basic text extraction
   - 2000 char limit

### Tool Integration

Tools are wrapped as LangChain Tools and can be:

- Called automatically by `call_tools` node
- Extended with custom tools
- Enabled/disabled via configuration

## Configuration

### New Environment Variables

```env
# Tool Configuration (Phase 2)
TAVILY_API_KEY=your_tavily_key_here
ENABLE_WEB_SEARCH=true
```

### Constants

- `MAX_RETRIES = 2`: Maximum query rewrites
- Prevents infinite loops
- Configurable in `src/graph/nodes.py`

## API Updates

### Enhanced `/query` Endpoint

**New Parameter**: `use_agentic`

```json
{
  "query": "What is RAG?",
  "user_id": "user123",
  "return_sources": true,
  "use_agentic": true // NEW: Use Phase 2 agentic mode
}
```

**Response Metadata** (Agentic Mode):

```json
{
  "answer": "...",
  "metadata": {
    "mode": "agentic",
    "retry_count": 1,
    "retrieval_attempted": true,
    "question_rewritten": true
  }
}
```

## Usage

### 1. Via CLI Script

```bash
python scripts/test_agentic_rag.py
```

Interactive mode with graph execution visualization.

### 2. Via Comparison Script

```bash
python scripts/compare_rag_modes.py
```

Compare Phase 1 vs Phase 2 behavior side-by-side.

### 3. Via API

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "How does RAG work?",
        "use_agentic": True  # Enable Phase 2
    }
)

result = response.json()
print(result["answer"])
print(f"Retries: {result['metadata']['retry_count']}")
```

### 4. Programmatically

```python
from src.graph import create_agentic_rag

graph = create_agentic_rag()
result = graph.invoke("What are RAG components?")

print(result["answer"])
print(f"Documents: {len(result['documents'])}")
print(f"Rewrites: {result['retry_count']}")
```

## Decision Flow Examples

### Example 1: Simple Question

**Q**: "What is 2+2?"

1. classify_or_answer → No retrieval needed
2. generate_answer → "4"

- **Path**: classify → answer
- **Nodes**: 2

### Example 2: Clear RAG Question

**Q**: "What are RAG components?"

1. classify_or_answer → Needs retrieval
2. retrieve → Found 5 documents
3. grade_documents → Documents relevant
4. generate_answer → Answer from context

- **Path**: classify → retrieve → grade → answer
- **Nodes**: 4

### Example 3: Ambiguous Question (Rewrite)

**Q**: "How does it work?"

1. classify_or_answer → Needs retrieval
2. retrieve → Found documents
3. grade_documents → Not specific enough
4. rewrite_question → "How does RAG system work?"
5. classify_or_answer → Needs retrieval
6. retrieve → Better documents
7. grade_documents → Relevant
8. generate_answer → Answer

- **Path**: classify → retrieve → grade → rewrite → classify → retrieve → grade → answer
- **Nodes**: 8, Retries: 1

## Advantages Over Phase 1

| Feature            | Phase 1 (Basic) | Phase 2 (Agentic)      |
| ------------------ | --------------- | ---------------------- |
| **Retrieval**      | Always          | Only when needed       |
| **Query Handling** | As-is           | Rewrites if unclear    |
| **Quality Check**  | None            | Grades relevance       |
| **Loops**          | Linear          | Adaptive with retries  |
| **Tools**          | No              | Yes (web search, etc.) |
| **Complexity**     | Simple          | Intelligent            |

## Performance Considerations

### When to Use Agentic Mode

✅ Unclear or ambiguous queries
✅ Need for adaptive behavior
✅ Complex multi-step reasoning
✅ When retrieval quality varies

### When to Use Basic Mode

✅ Well-formed queries
✅ Speed is critical
✅ Predictable patterns
✅ Simple Q&A

## Testing Scenarios

### Test 1: No Retrieval

```bash
python scripts/test_agentic_rag.py
# Choose option 3
```

Tests classification for simple questions.

### Test 2: With Retrieval

```bash
python scripts/test_agentic_rag.py
# Choose option 4
```

Tests full retrieval flow.

### Test 3: Comparison

```bash
python scripts/compare_rag_modes.py
```

Side-by-side Phase 1 vs Phase 2.

## Extending the Graph

### Adding New Nodes

```python
def custom_node(state: GraphState) -> GraphState:
    # Your logic
    state["custom_field"] = "value"
    return state

# In workflow.py
workflow.add_node("custom", custom_node)
workflow.add_edge("classify", "custom")
```

### Adding Tools

```python
# In tools/registry.py
class CustomTool:
    def run(self, input: str) -> str:
        return "custom result"

# Register in ToolRegistry._register_tools()
self.tools.append(Tool(
    name="custom_tool",
    description="Does custom thing",
    func=CustomTool().run
))
```

## Debugging

### Enable Verbose Logging

Nodes print their execution:

```
[CLASSIFY] Analyzing question...
[RETRIEVE] Fetching documents...
[GRADE] Evaluating 5 documents...
[REWRITE] Attempt 1/2
[GENERATE] Creating answer...
```

### Inspect Graph State

```python
# Stream execution to see each step
for step in graph.stream("What is RAG?"):
    print(step)
```

## Next Steps

Phase 2 provides intelligent RAG orchestration. Ready for:

- **Phase 3**: LlamaIndex integration for advanced retrieval
- **Phase 4**: Production hardening and scaling

## Key Features Implemented

✅ Graph state management
✅ Adaptive retrieval decisions
✅ Query classification
✅ Document relevance grading
✅ Query rewriting with limits
✅ Tool integration framework
✅ Calculator tool
✅ Web search (Tavily)
✅ HTTP fetch tool
✅ API support for agentic mode
✅ Comparison utilities
✅ Testing scripts
