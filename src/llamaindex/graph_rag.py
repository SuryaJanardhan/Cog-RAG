"""
GraphRAG implementation using LlamaIndex and Knowledge Graph Indexes.
Supports local file-based KG storage and production-grade Neo4j graph stores.
"""
import os
from typing import List, Dict, Any, Optional
from llama_index.core import (
    KnowledgeGraphIndex,
    StorageContext,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.schema import Document as LlamaDocument
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from langchain_core.documents import Document

from ..config import settings
from ..llm import get_gemini_client


class GraphRAGManager:
    """Manages LlamaIndex Knowledge Graph construction and query routing."""
    
    def __init__(self, persist_dir: str = "./data/kg_store"):
        self.persist_dir = persist_dir
        self.settings = settings
        self.index = None
        self._setup_llm()
        
    def _setup_llm(self) -> None:
        """Configure LLM for LlamaIndex operations."""
        # Configure Gemini as global LLM for LlamaIndex
        Settings.llm = Gemini(
            model=self.settings.gemini_model,
            api_key=self.settings.gemini_api_key,
            temperature=self.settings.gemini_temperature,
        )
        
    def get_graph_store(self) -> Any:
        """Get the configured graph store (Neo4j or Simple local graph store)."""
        # If Neo4j configuration is present, use it for production GraphRAG
        if self.settings.neo4j_url and "localhost" not in self.settings.neo4j_url:
            print(f"[GraphRAG] Initializing production Neo4jGraphStore on {self.settings.neo4j_url}")
            try:
                from llama_index.graph_stores.neo4j import Neo4jGraphStore
                return Neo4jGraphStore(
                    url=self.settings.neo4j_url,
                    username=self.settings.neo4j_username,
                    password=self.settings.neo4j_password,
                )
            except Exception as e:
                print(f"[GraphRAG] Failed to initialize Neo4j: {e}. Falling back to local simple graph store.")
                
        # Default local store
        print("[GraphRAG] Using local SimpleGraphStore.")
        return None
        
    def build_kg_index(self, documents: List[Document]) -> None:
        """Build a new Knowledge Graph index from LangChain documents."""
        print(f"[GraphRAG] Building Knowledge Graph index from {len(documents)} documents...")
        
        # Convert LangChain Documents to LlamaIndex Documents
        llama_docs = []
        for doc in documents:
            llama_docs.append(LlamaDocument(
                text=doc.page_content,
                metadata=doc.metadata
            ))
            
        # Set up storage context
        graph_store = self.get_graph_store()
        if graph_store:
            storage_context = StorageContext.from_defaults(graph_store=graph_store)
        else:
            storage_context = StorageContext.from_defaults()
            
        # Construct KG Index
        self.index = KnowledgeGraphIndex.from_documents(
            llama_docs,
            storage_context=storage_context,
            max_triplets_per_chunk=3,
            include_embeddings=True,
        )
        
        # Persist locally if using SimpleGraphStore
        if not graph_store:
            os.makedirs(self.persist_dir, exist_ok=True)
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            print(f"[GraphRAG] Persisted KG index to {self.persist_dir}")
            
    def load_index(self) -> bool:
        """Load persistent KG index from storage."""
        if self.index is not None:
            return True
            
        graph_store = self.get_graph_store()
        if graph_store:
            # Reconstruct from Neo4j
            try:
                storage_context = StorageContext.from_defaults(graph_store=graph_store)
                self.index = KnowledgeGraphIndex.from_documents(
                    [],
                    storage_context=storage_context,
                )
                return True
            except Exception as e:
                print(f"[GraphRAG] Reconstructing Neo4j index failed: {e}")
                return False
        else:
            # Reconstruct from local persistence files
            if os.path.exists(os.path.join(self.persist_dir, "index_store.json")):
                print(f"[GraphRAG] Loading local index from {self.persist_dir}")
                try:
                    from llama_index.core import load_index_from_storage
                    storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                    self.index = load_index_from_storage(storage_context)
                    return True
                except Exception as e:
                    print(f"[GraphRAG] Loading local index failed: {e}")
                    return False
            return False
            
    def query(self, query_str: str, similarity_top_k: int = 3) -> List[Document]:
        """Query the Knowledge Graph index and return matching triplet information as Documents."""
        if not self.load_index():
            print("[GraphRAG] Knowledge Graph Index not built or loaded yet.")
            return []
            
        print(f"[GraphRAG] Querying KG index: '{query_str}'")
        query_engine = self.index.as_query_engine(
            include_text=True,
            response_mode="tree_summarize",
            similarity_top_k=similarity_top_k,
        )
        
        response = query_engine.query(query_str)
        
        # Return relationship facts format as Document metadata
        source_nodes = response.source_nodes
        documents = []
        
        # Create a document containing the compiled answer from KG facts
        documents.append(Document(
            page_content=str(response),
            metadata={"source": "LlamaIndex KnowledgeGraphRAG", "type": "graph_triplets"}
        ))
        
        # Also append text snippet nodes
        for node in source_nodes:
            documents.append(Document(
                page_content=node.node.get_content(),
                metadata=node.node.metadata
            ))
            
        return documents
