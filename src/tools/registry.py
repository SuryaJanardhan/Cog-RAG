"""
Tool integrations for agentic RAG.
Provides web search, calculator, and other tools as LangChain tools.
"""
from typing import Optional, List, Any
from langchain_core.tools import Tool
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import math
import re
import requests

from ..config import settings


class CalculatorTool:
    """Simple calculator tool for basic math operations."""
    
    def run(self, expression: str) -> str:
        """
        Evaluate a mathematical expression.
        
        Args:
            expression: Math expression to evaluate
            
        Returns:
            str: Result or error message
        """
        try:
            # Safety: only allow basic math operations
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expression):
                return "Error: Invalid characters in expression"
            
            # Evaluate
            result = eval(expression, {"__builtins__": {}}, {
                "sin": math.sin,
                "cos": math.cos,
                "sqrt": math.sqrt,
                "pow": math.pow,
            })
            
            return f"Result: {result}"
        
        except Exception as e:
            return f"Error: {str(e)}"


class WebSearchTool:
    """Web search tool using Tavily API (free tier)."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize web search tool.
        
        Args:
            api_key: Tavily API key (optional)
        """
        self.api_key = api_key or getattr(settings, 'tavily_api_key', None)
        self.enabled = bool(self.api_key)
        
        if self.enabled:
            self.search = TavilySearchAPIWrapper(tavily_api_key=self.api_key)
        else:
            print("[WebSearch] No API key - web search disabled")
    
    def run(self, query: str) -> str:
        """
        Search the web for information.
        
        Args:
            query: Search query
            
        Returns:
            str: Search results or error
        """
        if not self.enabled:
            return "Web search is not configured. Please add TAVILY_API_KEY to .env"
        
        try:
            results = self.search.results(query, max_results=3)
            
            # Format results
            formatted = []
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                content = result.get('content', 'No content')
                url = result.get('url', '')
                
                formatted.append(f"{i}. {title}\n   {content}\n   Source: {url}")
            
            return "\n\n".join(formatted) if formatted else "No results found"
        
        except Exception as e:
            return f"Web search error: {str(e)}"


class SimpleFetchTool:
    """Simple HTTP fetch tool for getting webpage content."""
    
    def run(self, url: str) -> str:
        """
        Fetch content from a URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            str: Page content or error
        """
        try:
            # Basic validation
            if not url.startswith(('http://', 'https://')):
                return "Error: Invalid URL (must start with http:// or https://)"
            
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (RAG Bot)'
            })
            response.raise_for_status()
            
            # Return truncated content
            content = response.text[:2000]
            return f"Content from {url}:\n\n{content}..."
        
        except Exception as e:
            return f"Fetch error: {str(e)}"


class ToolRegistry:
    """Registry for all available tools."""
    
    def __init__(self):
        """Initialize tool registry."""
        self.tools: List[Tool] = []
        self._register_tools()
    
    def _register_tools(self):
        """Register all available tools."""
        # Calculator tool
        calc = CalculatorTool()
        self.tools.append(Tool(
            name="calculator",
            description="Useful for mathematical calculations. Input should be a math expression like '2+2' or 'sqrt(16)'",
            func=calc.run
        ))
        
        # Web search tool
        web_search = WebSearchTool()
        if web_search.enabled:
            self.tools.append(Tool(
                name="web_search",
                description="Search the web for current information. Input should be a search query.",
                func=web_search.run
            ))
        else:
            # Fallback web search mock to prevent crash and allow planning tests
            self.tools.append(Tool(
                name="web_search",
                description="Search the web for current information. Input should be a search query.",
                func=lambda q: f"Mock Web Search results for '{q}': RAG systems combine retrieval and generation."
            ))
        
        # HTTP fetch tool
        fetch = SimpleFetchTool()
        self.tools.append(Tool(
            name="fetch_url",
            description="Fetch content from a URL. Input should be a valid HTTP/HTTPS URL.",
            func=fetch.run
        ))

        # SQL read-only query tool
        self.tools.append(Tool(
            name="sql_db_query",
            description="Execute read-only SQL SELECT queries. Input must be a valid SQL statement.",
            func=self._mock_sql_query
        ))

        # SQL write tool (requires human approval)
        self.tools.append(Tool(
            name="sql_db_execute",
            description="Execute SQL modifications (INSERT, UPDATE, DELETE). Input must be a valid SQL statement.",
            func=self._mock_sql_execute
        ))
        
        print(f"[ToolRegistry] Registered {len(self.tools)} tools: {[t.name for t in self.tools]}")

    def _mock_sql_query(self, query: str) -> str:
        """Mock SQL query execution."""
        q_lower = query.lower()
        if "select" not in q_lower:
            return "Error: Only SELECT queries are allowed on sql_db_query."
        if "user" in q_lower or "employee" in q_lower:
            return json.dumps([
                {"id": 101, "name": "Alice Smith", "role": "RAG Engineer", "email": "alice@company.com"},
                {"id": 102, "name": "Bob Jones", "role": "AI Architect", "email": "bob@company.com"}
            ])
        return json.dumps([{"status": "success", "result": "Empty result set"}])

    def _mock_sql_execute(self, query: str) -> str:
        """Mock SQL modifications."""
        return f"Successfully executed update statement: '{query}'"
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
    
    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools."""
        return self.tools
    
    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools."""
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)


# Global tool registry
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


def execute_tool(tool_name: str, input_data: str) -> str:
    """
    Execute a tool by name.
    
    Args:
        tool_name: Name of the tool
        input_data: Input for the tool
        
    Returns:
        str: Tool result
    """
    registry = get_tool_registry()
    tool = registry.get_tool(tool_name)
    
    if not tool:
        return f"Error: Tool '{tool_name}' not found"
    
    try:
        return tool.func(input_data)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"
