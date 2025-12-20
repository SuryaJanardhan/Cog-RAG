"""
Document ingestion module for loading various document types.
Supports PDFs, web pages, text files, and more using LangChain loaders.
"""
import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)

from ..config import settings


class DocumentIngestion:
    """Handles document loading and normalization."""
    
    def __init__(self, raw_data_dir: str = None):
        """
        Initialize document ingestion.
        
        Args:
            raw_data_dir: Directory to store raw documents
        """
        self.raw_data_dir = raw_data_dir or "./data/raw"
        Path(self.raw_data_dir).mkdir(parents=True, exist_ok=True)
        self.metadata_file = os.path.join(self.raw_data_dir, "metadata.json")
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load existing metadata from file."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self) -> None:
        """Save metadata to file."""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _add_document_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Add document metadata to tracking file."""
        self.metadata[doc_id] = {
            **metadata,
            "ingested_at": datetime.now().isoformat(),
        }
        self._save_metadata()
    
    def load_pdf(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Load a PDF file.
        
        Args:
            file_path: Path to the PDF file
            metadata: Additional metadata to attach to documents
            
        Returns:
            List[Document]: Loaded documents with metadata
        """
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add custom metadata
        doc_id = os.path.basename(file_path)
        custom_metadata = {
            "source": file_path,
            "type": "pdf",
            "filename": doc_id,
            **(metadata or {})
        }
        
        for doc in documents:
            doc.metadata.update(custom_metadata)
        
        self._add_document_metadata(doc_id, custom_metadata)
        print(f"Loaded {len(documents)} pages from PDF: {file_path}")
        return documents
    
    def load_web_pages(self, urls: List[str], metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Load web pages from URLs.
        
        Args:
            urls: List of URLs to load
            metadata: Additional metadata to attach to documents
            
        Returns:
            List[Document]: Loaded documents with metadata
        """
        loader = WebBaseLoader(urls)
        documents = loader.load()
        
        # Add custom metadata to each document
        for i, doc in enumerate(documents):
            url = urls[i] if i < len(urls) else "unknown"
            doc_id = f"web_{hash(url)}"
            custom_metadata = {
                "source": url,
                "type": "web",
                "url": url,
                **(metadata or {})
            }
            doc.metadata.update(custom_metadata)
            self._add_document_metadata(doc_id, custom_metadata)
        
        print(f"Loaded {len(documents)} web pages")
        return documents
    
    def load_text_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Load a text file.
        
        Args:
            file_path: Path to the text file
            metadata: Additional metadata to attach to documents
            
        Returns:
            List[Document]: Loaded documents with metadata
        """
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        
        doc_id = os.path.basename(file_path)
        custom_metadata = {
            "source": file_path,
            "type": "text",
            "filename": doc_id,
            **(metadata or {})
        }
        
        for doc in documents:
            doc.metadata.update(custom_metadata)
        
        self._add_document_metadata(doc_id, custom_metadata)
        print(f"Loaded text file: {file_path}")
        return documents
    
    def load_directory(
        self, 
        directory_path: str, 
        glob_pattern: str = "**/*.txt",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Load all documents from a directory matching a pattern.
        
        Args:
            directory_path: Path to the directory
            glob_pattern: Pattern to match files (e.g., "**/*.pdf")
            metadata: Additional metadata to attach to documents
            
        Returns:
            List[Document]: Loaded documents with metadata
        """
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()
        
        # Add custom metadata
        for doc in documents:
            doc_id = os.path.basename(doc.metadata.get("source", "unknown"))
            custom_metadata = {
                "type": "directory_load",
                "directory": directory_path,
                "filename": doc_id,
                **(metadata or {})
            }
            doc.metadata.update(custom_metadata)
        
        print(f"Loaded {len(documents)} documents from directory: {directory_path}")
        return documents
    
    def load_word_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Load a Word document (.docx).
        
        Args:
            file_path: Path to the Word document
            metadata: Additional metadata to attach to documents
            
        Returns:
            List[Document]: Loaded documents with metadata
        """
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()
        
        doc_id = os.path.basename(file_path)
        custom_metadata = {
            "source": file_path,
            "type": "word",
            "filename": doc_id,
            **(metadata or {})
        }
        
        for doc in documents:
            doc.metadata.update(custom_metadata)
        
        self._add_document_metadata(doc_id, custom_metadata)
        print(f"Loaded Word document: {file_path}")
        return documents
    
    def normalize_documents(self, documents: List[Document]) -> List[Document]:
        """
        Normalize documents by ensuring consistent metadata structure.
        
        Args:
            documents: List of documents to normalize
            
        Returns:
            List[Document]: Normalized documents
        """
        normalized = []
        
        for doc in documents:
            # Ensure all documents have required metadata fields
            if "source" not in doc.metadata:
                doc.metadata["source"] = "unknown"
            if "type" not in doc.metadata:
                doc.metadata["type"] = "unknown"
            
            normalized.append(doc)
        
        return normalized
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get statistics about ingested documents.
        
        Returns:
            Dict with ingestion statistics
        """
        stats = {
            "total_documents": len(self.metadata),
            "by_type": {},
            "metadata_file": self.metadata_file
        }
        
        for doc_id, meta in self.metadata.items():
            doc_type = meta.get("type", "unknown")
            stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + 1
        
        return stats


def create_ingestion_pipeline() -> DocumentIngestion:
    """Factory function to create document ingestion pipeline."""
    return DocumentIngestion()
