"""
Parent-Document retriever implementation.
Splits documents into parent and child chunks, stores child embeddings in the vector DB,
and retrieves the larger parent chunks when a child chunk matches.
"""
import sqlite3
import json
import hashlib
import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..config import settings


class ParentDocumentStore:
    """SQLite-based key-value store for parent document chunks."""
    
    def __init__(self, db_path: str = "./data/parent_documents.db"):
        self.db_path = db_path
        self._initialize_db()
        
    def _initialize_db(self) -> None:
        """Create parent chunks table if it does not exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parent_chunks (
                parent_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        
    def get(self, parent_id: str) -> Optional[Document]:
        """Retrieve a parent document by its unique identifier."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT content, metadata FROM parent_chunks WHERE parent_id = ?", (parent_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Document(page_content=row[0], metadata=json.loads(row[1]))
        return None
        
    def set(self, parent_id: str, document: Document) -> None:
        """Store a parent document mapped by its unique identifier."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO parent_chunks (parent_id, content, metadata) VALUES (?, ?, ?)",
            (parent_id, document.page_content, json.dumps(document.metadata))
        )
        conn.commit()
        conn.close()


class ParentDocumentSplitter:
    """Splits documents into parent/child hierarchy and indexes parents in SQLite store."""
    
    def __init__(
        self,
        parent_size: int = 2000,
        parent_overlap: int = 400,
        child_size: int = 400,
        child_overlap: int = 50
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap
        )
        self.store = ParentDocumentStore()
        
    def split_and_store(self, documents: List[Document]) -> List[Document]:
        """
        Splits original documents into parent chunks (indexed locally) and child chunks (returned for vector DB).
        """
        child_docs = []
        for i, doc in enumerate(documents):
            parent_chunks = self.parent_splitter.split_documents([doc])
            for j, p_chunk in enumerate(parent_chunks):
                # Generate unique parent ID
                content_bytes = p_chunk.page_content.encode("utf-8")
                p_content_hash = hashlib.sha256(content_bytes).hexdigest()
                parent_id = f"parent_{p_content_hash}_{i}_{j}"
                
                # Store parent document in key-value store
                self.store.set(parent_id, p_chunk)
                
                # Split parent chunk into child chunks
                child_chunks = self.child_splitter.split_documents([p_chunk])
                for child in child_chunks:
                    child.metadata["parent_id"] = parent_id
                    # Keep source metadata and link parent
                    child.metadata["source"] = doc.metadata.get("source", "unknown")
                    if "page" in doc.metadata:
                        child.metadata["page"] = doc.metadata["page"]
                    child_docs.append(child)
                    
        print(f"Hierarchical splitting: {len(documents)} docs -> {len(child_docs)} child chunks")
        return child_docs
